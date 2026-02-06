try:
    import bitsandbytes as bnb
    print("bitsandbytes version:", getattr(bnb, "__version__", "unknown"))
except Exception as e:
    import traceback; traceback.print_exc()
    print("bitsandbytes import failed:", e)
