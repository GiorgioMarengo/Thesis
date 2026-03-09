import struct
import binascii
import sys
import os

def parse_quote(filepath):
    if not os.path.exists(filepath):
        print(f"Errore: File non trovato: {filepath}")
        return

    with open(filepath, "rb") as f:
        data = f.read()

    print(f"\n--- Analisi Quote: {os.path.basename(filepath)} ---")

    # Cerca l'inizio del REPORT BODY
    # In DCAP v3, l'header è 48 byte.
    if len(data) < 432:
        print("Errore: File troppo corto.")
        return

    # Estraiamo il Body (offset 48, lunghezza 384)
    report_body = data[48:48+384]

    # REPORT DATA (Offset 320 nel body, lungo 64 bytes)
    # Qui c'è il tuo hash!
    report_data = report_body[320:384]

    hex_data = binascii.hexlify(report_data).decode('utf-8')

    print(f"\n[REPORT_DATA] (Il tuo binding hash):")
    print(f"{hex_data}")
    print("\nSe i primi 64 caratteri corrispondono al tuo SHA256, è fatta!")

if __name__ == "__main__":
    # Cerca automaticamente il file .bin nella cartella corrente se non specificato
    files = [f for f in os.listdir('.') if f.endswith('.bin') and 'quote' in f]

    if len(sys.argv) > 1:
        parse_quote(sys.argv[1])
    elif files:
        print(f"Trovato file quote: {files[0]}")
        parse_quote(files[0])
    else:
        print("Nessun file quote .bin trovato nella cartella.")
