import asyncio
import dcap_qvl
import struct

async def verify_and_parse_quote():
    quote = open("quote_2026-02-01.bin", "rb").read()

    # Verifica
    result = await dcap_qvl.get_collateral_and_verify(quote)
    print(f"Status TCB: {result.status}")
    print(f"Advisory IDs: {result.advisory_ids if result.advisory_ids else 'Nessuno'}")
    print()

    # Parsing
    mrenclave = quote[112:144].hex()
    mrsigner = quote[176:208].hex()
    report_data = quote[368:432]
    isv_prod_id = struct.unpack('<H', quote[304:306])[0]
    isv_svn = struct.unpack('<H', quote[306:308])[0]
    attributes = struct.unpack('<Q', quote[96:104])[0]
    debug_mode = bool(attributes & 0x2)

    print(f"MRENCLAVE: {mrenclave}")
    print(f"MRSIGNER: {mrsigner}")
    print(f"Report Data: {report_data.hex()}")
    print(f"ISV Product ID: {isv_prod_id}")
    print(f"ISV SVN: {isv_svn}")
    print(f"Debug Mode: {debug_mode}")

asyncio.run(verify_and_parse_quote())
