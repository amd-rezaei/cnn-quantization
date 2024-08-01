import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.insert(0, parent_dir)

from utils.mllog import MLlogger
from pytorch_quantizer.quantization.inference.inference_quantization_manager import QuantizationManagerInference as QM
from pathlib import Path
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch
import argparse


# Argument parser setup
parser = argparse.ArgumentParser(description='PyTorch MNIST Quantization')
parser.add_argument('--data', metavar='DIR', default='./data',
                    help='path to dataset')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--device', default='cuda',
                    help='device assignment ("cpu" or "cuda")')
parser.add_argument('--eval_precision', '-ep', action='store_true', default=False,
                    help='Evaluate different precisions, to csv.')
parser.add_argument('--arch', default='custom_mnist_model',
                    help='Model architecture')
args = parser.parse_args()

# Set deterministic behavior
torch.backends.cudnn.deterministic = True
torch.manual_seed(12345)


class SimpleMNISTModel(nn.Module):
    def __init__(self):
        super(SimpleMNISTModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28 * 1, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def get_params():
    qparams = {
        'int': {
            'clipping': 'laplace',  # Clipping type: laplace
            'stats_kind': 'mean',  # Statistics kind: mean
            'true_zero': False,  # Preserve zero during quantization
            'kld': False,  # KLD threshold
            'pcq_weights': True,  # Per channel quantization of weights
            'pcq_act': True,  # Per channel quantization of activations
            'bit_alloc_act': True,  # Optimal bit allocation for activations
            'bit_alloc_weight': True,  # Optimal bit allocation for weights
            'bit_alloc_rmode': 'round',  # Bit allocation rounding mode: round
            'bit_alloc_prior': 'gaus',  # Bit allocation prior: gaus
            'bit_alloc_target_act': None,  # Target value for bit allocation quota of activations
            # Target value for bit allocation quota of weights
            'bit_alloc_target_weight': None,
            'bcorr_act': False,  # Bias correction for activations
            'bcorr_weight': True,  # Bias correction for weights
            'vcorr_weight': False,  # Variance correction for weights
            'logger': None,  # Logger
            'measure_entropy': False,  # Measure entropy of activations
            'mtd_quant': False  # Mid thread quantization
        },
        'qmanager': {
            'rho_act': None,  # Rho parameter for activations
            'rho_weight': None  # Rho parameter for weights
        }
    }
    return qparams


def set_default_args(args):
    default_values = {
        'batch_size': 512,
        'qtype': 'int4',
        'qweight': 'int4',
        'q_off': False,
        'shuffle': True,
        'per_channel_quant_weights': True,
        'per_channel_quant_act': True,
        'bit_alloc_act': True,
        'bit_alloc_weight': True,
        'bias_corr_act': False,
        'bias_corr_weight': True,
        'var_corr_weight': False,
        'clipping': 'laplace',
        'arch': 'custom_mnist_model',
        'true_zero': False,
        'kld_threshold': False,
        'bit_alloc_rmode': 'round',
        'bit_alloc_prior': 'gaus',
        'bit_alloc_target_act': None,
        'bit_alloc_target_weight': None,
        'logger': None,
        'measure_entropy': False,
        'mtd_quant': False,
        'rho_act': None,
        'rho_weight': None,
        'stats_folder': None,
        'stats_mode': 'no',
        'stats_batch_avg': False,
        'measure_stats': False
    }
    for key, value in default_values.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args


def main():
    global args
    args = set_default_args(args)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Create and load the model
    model = SimpleMNISTModel().to(device)

    # Initialize Quantization Manager with required arguments
    quant_manager = QM(args, get_params())
    quant_manager.quantize_model(model)

    # Data loading code
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    val_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.data, train=False,
                       transform=transform, download=True),
        batch_size=args.batch_size, shuffle=args.shuffle, num_workers=0, pin_memory=True)

    # Define loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Validate and save the quantized model
    validate(val_loader, model, criterion)
    save_quantized_model(model)


def validate(val_loader, model, criterion):
    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input, target = input.to(args.device), target.to(args.device)
            output = model(input)
            loss = criterion(output, target)

            if i % args.print_freq == 0:
                print(f'Test: [{i}/{len(val_loader)}]\tLoss {loss.item():.4f}')


def save_quantized_model(model):
    state_dict = model.state_dict()
    torch.save(state_dict, 'quantized_model_neuton.pth')
    print("Quantized model saved to 'quantized_model_neuton.pth'")

    dummy_input = torch.randn(1, 1, 28, 28).to(args.device)
    onnx_path = 'quantized_model.onnx'
    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Quantized model saved to '{onnx_path}'")


if __name__ == '__main__':
    main()
