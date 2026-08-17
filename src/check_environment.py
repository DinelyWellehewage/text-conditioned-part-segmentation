import torch
import torchvision
import numpy as np
from PIL import Image
import matplotlib

print("Environment check")
print("-----------------")
print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("NumPy:", np.__version__)
print("Pillow:", Image.__version__)
print("Matplotlib:", matplotlib.__version__)

print()
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
else:
    print("Running without CUDA")