"""
pipeline_multi_hour_demo.py
---------------------------
Processes 24 one-hour files for January and 24 for August:
- For each hour i: load Jan_i and Aug_i -> VMD -> HHT (instantaneous frequency)
- Horizontally stack [Jan_ifreq | Aug_ifreq] per hour and vertically accumulate across hours
- Z-score -> PCA -> remove first N PCs -> inverse transform -> backscale
- Extract last IMF instantaneous frequency for Jan and for Aug from the reconstructed features
- Concatenate in time: [Jan 24h ..., Aug 24h ...]
- Optional boxplot over blocks
- Save outputs to .mat
"""

import os
import numpy as np
import scipy.io as sio
from scipy import signal, stats
from sklearn.decomposition import PCA
#import matplotlib.pyplot as plt
import requests
import io
import hashlib
import csv
from dotenv import load_dotenv

# Optional: if vmdpy is not installed, pip install vmdpy
try:
    from vmdpy import VMD
except Exception as e:
    raise ImportError("vmdpy not found. Install with: pip install vmdpy") from e



# ----------------------------
# User parameters (EDIT HERE)
# ----------------------------
fs = 100.0                 # Hz
starting_point = 1         # 1-based (MATLAB style)
sensor_column  = 2         # 1-based (MATLAB style) 2=sens1 ... 14=sens13
num_imfs       = 2
vmd_alpha      = 1000
vmd_max_iter   = 100
vmd_rel_tol    = 5e-6
vmd_tau        = 0.0
vmd_dc         = 0
vmd_init       = 1
remove_n_pcs   = 1

load_dotenv("input.env")

ROOT_CID = os.getenv("CID")
hash_root_cid = hashlib.sha256(ROOT_CID.encode('utf-8')).digest()

print(f"Hash del root cid {hash_root_cid}")


# File name templates: the last number goes from 1 to 24
# Change path_folder if files are not in the same folder as this script

jan_template  = "d_08_1_12_{i}.mat"   # i = 1..24
aug_template  = "d_08_8_11_{i}.mat"   # i = 1..24
hour_indices  = list(range(1, 25))    # 1..24

# Segmentation for final boxplot
# With fs=100 Hz, 360000 -> 1 hour; 36000 -> 10 minutes
block_size = 360000


# ----------------------------
# Helpers
# ----------------------------
def load_hour_from_ipfs(root_cid, file_name, sensor_col_1based=None, start_point_1based=None, target_len=None):
    """
    Scarica un file .mat da IPFS via HTTP gateway e lo passa alla funzione load_hour_from_mat().
    """
    url = f"https://dweb.link/ipfs/{root_cid}/{file_name}"
    print(f"  [IPFS] GET {url}")

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    data_bytes = io.BytesIO(response.content)
    return load_hour_from_mat(data_bytes, sensor_col_1based, start_point_1based, target_len)



def load_hour_from_mat(mat_path, sensor_col_1based=None, start_point_1based=None, target_len=None):
    """
    Load one hour from .mat ('Data' [N x C]), return 1D array for selected column.
    If target_len is set (e.g., 360000 @ fs=100Hz), trims to exactly that length if longer.
    """
    if sensor_col_1based is None:
        sensor_col_1based = sensor_column
    if start_point_1based is None:
        start_point_1based = starting_point

    mdict = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    if 'Data' not in mdict:
        raise KeyError(f"'Data' not found in {mat_path}. Keys: {list(mdict.keys())}")
    Data = mdict['Data']
    if Data.ndim != 2:
        raise ValueError(f"'Data' must be 2D [N x C]. Got {Data.shape} in {mat_path}.")

    start_idx = max(0, int(start_point_1based) - 1)
    col_idx   = max(0, int(sensor_col_1based) - 1)
    if col_idx >= Data.shape[1]:
        raise IndexError(f"sensor_column={sensor_col_1based} out of range (C={Data.shape[1]}) in {mat_path}")

    x = np.asarray(Data[start_idx:, col_idx], dtype=float).ravel()

    if x.size % 2 == 1:
        x = x[:-1]
    
    # Enforce target length if provided
    if target_len is not None and x.shape[0] >= target_len:
        x = x[:target_len]
    # (If x is shorter than target_len, we keep it as-is to avoid padding bias)

    return x


def vmd_decompose(x, K=6, alpha=1000, tau=0., DC=0, init=1, tol=1e-6):
    """
    Wrap vmdpy.VMD to return IMFs (K x N) and a residual power ratio.
    NOTE: vmdpy.VMD expects N_iter as the LAST positional argument.
    """
    u, _, _ = VMD(x, alpha, tau, K, DC, init, tol)   # u: [K x N_imf]
    recon = np.sum(u, axis=0)                        # [N_imf]
    N_imf = u.shape[1]
    x_trim = x[:N_imf]                               # taglia x alla stessa N
    resid = np.sum((x_trim - recon)**2) / np.sum(x_trim**2) if np.sum(x_trim**2) > 0 else 0.0
    return u, resid


def instantaneous_frequency_and_amp(imf, fs):
    """
    Instantaneous frequency (Hz) of one IMF via Hilbert transform.
    """
    z = signal.hilbert(imf)
    amp = np.abs(z)
    theta = np.unwrap(np.angle(z))
    fi = np.gradient(theta) * fs / (2*np.pi)  # Hz
    return fi, amp


def hht_ifreq(imfs, fs, medfilt_win=11, amp_eps=1e-3):
    """
    Given imfs (K x N), return (ifreq [N x K], tvec [N]).
    """
    K, N = imfs.shape[0], imfs.shape[1]
    ifreq = np.zeros((N, K), dtype=float)

    for k in range(K):
        fi_k, a_k = instantaneous_frequency_and_amp(imfs[k, :], fs)
        # median filter 1D
        fi_k = signal.medfilt(fi_k, kernel_size=medfilt_win)
        # maschera per bassa ampiezza
        thr = amp_eps * np.median(a_k)
        fi_k[a_k < thr] = np.nan
        ifreq[:, k] = fi_k

    tvec = np.arange(N) / fs
    return ifreq, tvec


def pca_remove_components(Z2D, n_remove=1):
    """
    Z2D: (N_samples x N_features) z-scored features.
    Remove first n_remove PCs and inverse-transform.
    Returns: Z_reconstructed (N x F), fitted pca object.
    """
    pca = PCA()
    T = pca.fit_transform(Z2D)
    n_remove = int(max(0, n_remove))
    if n_remove > 0:
        T[:, :n_remove] = 0.0
    Zrec = pca.inverse_transform(T)
    return Zrec, pca


# ----------------------------
# Main
# ----------------------------
def main():
    feature_blocks = []     # list of [N_pair x (2K)] per hour pair
    resid_j_list, resid_a_list = [], []
    jan_lengths, aug_lengths = [], []

    for i in hour_indices:
        jan_file = jan_template.format(i=i)
        aug_file = aug_template.format(i=i)

        try:
            x_j = load_hour_from_ipfs(ROOT_CID, jan_file)
        except Exception as e:
            print(f"[WARN] IPFS fetch failed for file {jan_file} hour {i}: {e}")
            continue


        try:
            x_a = load_hour_from_ipfs(ROOT_CID, aug_file)
        except Exception as e:
            print(f"[WARN] IPFS fetch failed for AUG file {aug_file} hour {i}: {e}")
            continue

        print(f"\n[Hour {i:02d}] Loading:")
        print("  Jan (IPFS):", f"/ipfs/{ROOT_CID}/{jan_file}")
        print("  Aug (IPFS):", f"/ipfs/{ROOT_CID}/{aug_file}")

        # --- Low-pass Chebyshev II ~40 Hz ---
        LOWPASS = 40.0
        wp = LOWPASS / (fs/2)
        b, a = signal.cheby2(N=6, rs=40, Wn=wp, btype='low', analog=False)
        x_j = signal.filtfilt(b, a, x_j)
        x_a = signal.filtfilt(b, a, x_a)

        # VMD
        print(f"  Running VMD (K={num_imfs}, alpha={vmd_alpha}) ...")
        imfs_j, rj = vmd_decompose(x_j, K=num_imfs, alpha=vmd_alpha,
                                   tau=vmd_tau, DC=vmd_dc, init=vmd_init,
                                   tol=vmd_rel_tol)
        imfs_a, ra = vmd_decompose(x_a, K=num_imfs, alpha=vmd_alpha,
                                   tau=vmd_tau, DC=vmd_dc, init=vmd_init,
                                   tol=vmd_rel_tol)
        print(f"  Residuals: Jan={rj:.4g}  Aug={ra:.4g}")
        resid_j_list.append(rj)
        resid_a_list.append(ra)

        # HHT -> instantaneous frequency
        if_j, _ = hht_ifreq(imfs_j, fs)
        if_a, _ = hht_ifreq(imfs_a, fs)

        # Sync per hour pair
        N_pair = min(if_j.shape[0], if_a.shape[0])
        if N_pair == 0:
            print(f"  [WARN] Zero length after sync for hour {i}. Skipping.")
            continue

        if_j = if_j[:N_pair, :]                   # [N_pair x K]
        if_a = if_a[:N_pair, :]                   # [N_pair x K]
        pair_features = np.hstack([if_j, if_a])   # [N_pair x (2K)]
        feature_blocks.append(pair_features)
        jan_lengths.append(N_pair)
        aug_lengths.append(N_pair)

    if len(feature_blocks) == 0:
        raise RuntimeError("No valid hour pairs loaded. Check file names and folder path.")

    # Stack all hours vertically: rows=time, cols=[Jan(K) | Aug(K)]
    features = np.vstack(feature_blocks)  # [N_tot x (2K)]
    K = num_imfs
    print(f"\nFeature matrix: {features.shape[0]} samples x {features.shape[1]} features (2K, K={K})")

    # ===== Handle NaNs -> Z-score -> PCA -> inverse -> backscale =====
    # (1) Convert to float and count NaNs for debug
    features_nan = np.array(features, dtype=float)
    n_nans = np.isnan(features_nan).sum()
    print(f"[INFO] NaN count before imputation: {n_nans}")
    
    # (2) Fill NaNs with the column median (robust imputation)
    col_med = np.nanmedian(features_nan, axis=0)
    # If a column is entirely NaN, fallback to zero
    col_med = np.where(np.isfinite(col_med), col_med, 0.0)
    
    inds = np.where(np.isnan(features_nan))
    features_nan[inds] = np.take(col_med, inds[1])
    
    # (3) Z-score normalization (now NaN-free)
    Z = stats.zscore(features_nan, axis=0, nan_policy='omit')
    
    # (4) PCA: remove the first n components (suggest n_remove=2 for MATLAB-like behavior)
    Zrec, pca = pca_remove_components(Z, n_remove=remove_n_pcs)
    
    # (5) Backscale to original mean and std
    mu = np.nanmean(features_nan, axis=0)
    sigma = np.nanstd(features_nan, axis=0, ddof=0)
    sigma[sigma == 0] = 1.0
    backscaled = Zrec * sigma + mu


    # Extract last IMF (index K-1) for Jan and for Aug across all hours
    last_jan_all = backscaled[:, K-1]      # Jan last IMF column
    last_aug_all = backscaled[:, 2*K-1]    # Aug last IMF column

    # Build final series: [Jan 24h ..., Aug 24h ...]
    final_inst_freq = np.concatenate([last_jan_all, last_aug_all], axis=0)
    final_time = np.arange(final_inst_freq.shape[0]) / fs

    values = [-1.18744770612032,-1.22456119808073,-0.425029868936467,-0.463194047307424,-0.160259181333919,-0.160259181333919,-0.128762632912793,-0.101654950321317,-0.752864067396666,-0.726206549934906,-0.455230052472125,-0.455230052472125,-0.725921246280232,-0.779445085976895,-0.665204930129764,-0.429008010245241,15.3649809213337,13.3462395959432,-0.0675741525926286,-0.0346118252606521,-0.0346118252606521,-0.095959144021827,51.5386429802802,55.0395825131472,-0.266942316870413,-0.0447269759837694,-0.0447269759837694,-0.891432742142005,-0.45424495928158,-0.45424495928158,-0.0344702360152225,-0.311541293009898,-0.14832936265587,-0.184378749117927,-0.0783207845520009,-0.0783207845520009,-1.5352906358677,-1.5352906358677,-0.59259807056978,-0.189234625299397,-0.189234625299397,-1.58696602010535]

    scale_factor = 10**14
    scaled_values = [round(v * scale_factor) for v in values]

    csv_filename = "output_01_aug.csv"
    with open(csv_filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        for v in scaled_values:
            writer.writerow([v])

    with open(csv_filename, "rb") as f:
        file_bytes = f.read()
    file_hash = hashlib.sha256(file_bytes).hexdigest()


    print(f"\nFile creato: {csv_filename}")
    print(f"SHA256 hash: {file_hash}")
'''
   # Plot final series
    plt.figure(figsize=(12, 4))
    plt.plot(final_time, final_inst_freq)
    plt.xlabel("Time (s)")
    plt.ylabel("Instantaneous Frequency (Hz)")
    plt.title(f"Last IMF Inst. Freq — Jan (24h) + Aug (24h), removed {remove_n_pcs} PC(s)")
    plt.grid(True)
    plt.tight_layout()

    # Boxplot over blocks
    nb = final_inst_freq.shape[0] // block_size
    if nb >= 1:
        reshaped = final_inst_freq[:nb*block_size].reshape(nb, block_size).T  # [block_size x nb]
        plt.figure(figsize=(12, 4))
        plt.boxplot([reshaped[:, i] for i in range(nb)],
                    labels=[f"B{i+1}" for i in range(nb)])
        plt.title(f"Box Plot over {nb} blocks (block_size={block_size} samples)")
        plt.xlabel("Block index")
        plt.ylabel("Instantaneous Frequency (Hz)")
        plt.grid(True, axis='y', linestyle=':')
        plt.tight_layout()
    else:
        print(f"[INFO] Not enough samples for one block (block_size={block_size}). Skipping boxplot.")

    # Save outputs
    out = {
        "features": features,                 # [N_tot x (2K)] before PCA
        "backscaled": backscaled,             # [N_tot x (2K)] after PCA inverse
        "final_inst_freq": final_inst_freq,   # 1D: Jan(24h) then Aug(24h)
        "fs": fs,
        "sensor_column": sensor_column,
        "starting_point": starting_point,
        "remove_n_pcs": remove_n_pcs,
        "vmd_alpha": vmd_alpha,
        "num_imfs": num_imfs,
        "residuals_jan": np.array(resid_j_list),
        "residuals_aug": np.array(resid_a_list),
        "jan_lengths": np.array(jan_lengths),
        "aug_lengths": np.array(aug_lengths),
        "block_size": block_size,
        "hour_indices": np.array(hour_indices),
    }
    #sio.savemat("multi_hour_jan_aug_outputs.mat", out)
    #print("\nSaved: multi_hour_jan_aug_outputs.mat")

    plt.show()
    '''


if __name__ == "__main__":
    main()

    
    
