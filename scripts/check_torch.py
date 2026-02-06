import torch, sys
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    try:
        print("device name:", torch.cuda.get_device_name(0))
        print("device count:", torch.cuda.device_count())
    except Exception as e:
        print("device info error:", e)
