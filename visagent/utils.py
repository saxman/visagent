import torch


def get_pytorch_device():
    device = torch.accelerator.current_accelerator()
    if torch.accelerator.is_available():
        print(f"using PyTorch GPU device: {device}")
    else:
        print(f"using PyTorck CPU device: {device}")
    return device
