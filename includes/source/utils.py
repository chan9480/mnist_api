import torch
import onnx
from torchvision import transforms
from includes.source.model import CNN

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def convert_to_onnx(model_path):
    model = CNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    dummy_input = torch.randn(1, 1, 28, 28)
    onnx_path = './includes/models/onnx/mnist_model.onnx'
    torch.onnx.export(model, dummy_input, onnx_path)

    return onnx_path
