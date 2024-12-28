import torch
import onnx
from torchvision import transforms
from includes.source.model import CNN

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])