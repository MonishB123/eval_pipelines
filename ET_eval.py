"""
GraphSSM Time Series Forecasting Evaluation Script

This script is designed to work with the following directory structure:
- eval_forecasting.py: /data/
- gg_ssms repository: /workspace/
- ETDataset: /data/datasets/ETDataset/
- Model checkpoints: /data/checkpoints/
- Results: /data/results/

The script imports GraphSSM from the gg_ssms repository and uses MambaTS
data providers for time series forecasting evaluation on the ETT dataset.
"""

import argparse
import math
import os
import random
import sys
import time
from typing import Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity

# Add MambaTS to path to use their data providers
# Since eval_forecasting.py is now in ~/data and gg_ssms repo is in ~/workspace
gg_ssms_path = os.path.expanduser("/workspace")
mamba_ts_path = os.path.join(gg_ssms_path, "MambaTS")

# Check if paths exist and provide helpful error messages
if not os.path.exists(gg_ssms_path):
    print(f"ERROR: GG_SSMS repository not found at {gg_ssms_path}")
    print("Please ensure the gg_ssms repository is located at /workspace")
    sys.exit(1)

if not os.path.exists(mamba_ts_path):
    print(f"ERROR: MambaTS not found at {mamba_ts_path}")
    print("Please ensure MambaTS is located in the gg_ssms repository")
    sys.exit(1)

sys.path.append(mamba_ts_path)

from data_provider.data_factory import data_provider
from utils.tools import set_seed

# Import GraphSSM directly from main.py in the gg_ssms repo
main_py_path = os.path.join(gg_ssms_path, "core", "graph_ssm", "main.py")
if not os.path.exists(main_py_path):
    print(f"ERROR: main.py not found at {main_py_path}")
    print("Please ensure the core/graph_ssm/main.py file exists in the gg_ssms repository")
    sys.exit(1)

sys.path.append(os.path.join(gg_ssms_path, "core", "graph_ssm"))
from main import GraphSSM


# Remove the synthetic dataset class - we'll use MambaTS data providers instead


class TFLOPSCalculator:
    """Utility class for calculating TFLOPS during model inference"""
    
    def __init__(self):
        self.total_flops = 0
        self.total_time = 0.0
        self.batch_count = 0
        self.start_time = None
        
    def start_timing(self):
        """Start timing for a batch"""
        self.start_time = time.time()
        
    def end_timing(self):
        """End timing for a batch and accumulate time"""
        if self.start_time is not None:
            batch_time = time.time() - self.start_time
            self.total_time += batch_time
            self.batch_count += 1
            self.start_time = None
            return batch_time
        return 0.0
    
    def add_flops(self, flops: int):
        """Add FLOPs for a batch"""
        self.total_flops += flops
        
    def calculate_tflops(self) -> float:
        """Calculate TFLOPS from accumulated data"""
        if self.total_time > 0:
            return (self.total_flops / 1e12) / self.total_time
        return 0.0
    
    def reset(self):
        """Reset counters"""
        self.total_flops = 0
        self.total_time = 0.0
        self.batch_count = 0
        self.start_time = None


def count_linear_flops(input_shape: tuple, output_features: int, bias: bool = True) -> int:
    """Count FLOPs for a linear layer"""
    batch_size, seq_len, input_features = input_shape
    flops_per_output = input_features * 2  # multiply + add
    if bias:
        flops_per_output += 1  # bias addition
    total_outputs = batch_size * seq_len * output_features
    return total_outputs * flops_per_output


def count_normalization_flops(input_shape: tuple) -> int:
    """Count FLOPs for normalization operations"""
    batch_size, seq_len, features = input_shape
    mean_flops = batch_size * features * seq_len
    variance_flops = batch_size * seq_len * features * 3  # subtract, square, add
    sqrt_flops = batch_size * features
    div_flops = batch_size * seq_len * features
    return mean_flops + variance_flops + sqrt_flops + div_flops


def estimate_graphssm_flops(input_shape: tuple, d_model: int, d_state: int, d_conv: int, expand: int) -> int:
    """Estimate FLOPs for GraphSSM operations"""
    batch_size, seq_len, _ = input_shape
    input_proj_flops = batch_size * seq_len * d_model * d_model * 2
    state_flops = batch_size * seq_len * d_state * d_state * 2
    conv_flops = batch_size * seq_len * d_model * d_conv * 2
    output_proj_flops = batch_size * seq_len * d_model * d_model * 2
    total_flops = (input_proj_flops + state_flops + conv_flops + output_proj_flops) * expand
    return int(total_flops)


def count_model_flops(model: nn.Module, input_shape: tuple, args: argparse.Namespace) -> int:
    """Count total FLOPs for the TimeSeriesForecaster model"""
    batch_size, seq_len, enc_in = input_shape
    total_flops = 0
    input_embedding_flops = count_linear_flops(input_shape, args.d_model, bias=False)
    total_flops += input_embedding_flops
    norm_flops = count_normalization_flops(input_shape) * 2
    total_flops += norm_flops
    graphssm_input_shape = (batch_size, seq_len, args.d_model)
    graphssm_flops = estimate_graphssm_flops(
        graphssm_input_shape, args.d_model, args.d_state, args.d_conv, args.expand
    )
    total_flops += graphssm_flops
    output_shape = (batch_size, args.pred_len, args.d_model)
    output_proj_flops = count_linear_flops(output_shape, args.c_out, bias=False)
    total_flops += output_proj_flops
    return total_flops


class TimeSeriesForecaster(nn.Module):
    def __init__(
        self,
        enc_in: int,
        c_out: int,
        seq_len: int,
        pred_len: int,
        d_model: int = 512,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        distance_metric: str = "cosine",  # NEW: distance formula argument
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.c_out = c_out
        self.d_model = d_model
        self.distance_metric = distance_metric  # Store distance formula
        
        # Simple embedding layer
        self.input_embedding = nn.Linear(enc_in, d_model)
        
        # Core GraphSSM with distance formula
        self.graph_ssm = GraphSSM(
            d_model=d_model, 
            d_state=d_state, 
            d_conv=d_conv, 
            expand=expand,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            use_fast_path=True,
            distance_metric=self.distance_metric  # Pass to GraphSSM
        )
        
        self.output_projection = nn.Linear(d_model, c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        b, seq_len, enc_in = x_enc.shape
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc /= stdev
        embedded = self.input_embedding(x_enc)
        context_len = min(seq_len, 4)
        processed = self.graph_ssm(embedded, context_len)
        if processed.shape[1] >= self.pred_len:
            processed = processed[:, -self.pred_len:, :]
        else:
            last_timestep = processed[:, -1:, :].repeat(1, self.pred_len, 1)
            processed = last_timestep
        output = self.output_projection(processed)
        output = output * (stdev[:, [0], :].repeat(1, self.pred_len, 1))
        output = output + (means[:, [0], :].repeat(1, self.pred_len, 1))
        return output


def load_pretrained_model(model_path: str, args: argparse.Namespace, device: torch.device):
    """Load a pre-trained model from checkpoint"""
    model = TimeSeriesForecaster(
        enc_in=args.enc_in,
        c_out=args.c_out,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        d_model=args.d_model,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        distance_metric=args.distance,  # Pass command-line distance
    ).to(device)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded pre-trained model from {model_path}")
        else:
            model.load_state_dict(checkpoint)
            print(f"Loaded pre-trained model from {model_path}")
    else:
        print(f"No pre-trained model found at {model_path}")
        print("Using randomly initialized model for demonstration...")
    
    model.eval()
    return model


def build_argparser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate core/graph_ssm on MambaTS time series forecasting")
    
    # -- snip: all existing arguments here --
    
    # GraphSSM specific
    parser.add_argument("--d_state", type=int, default=16, help="GraphSSM state size")
    parser.add_argument("--d_conv", type=int, default=4, help="GraphSSM conv kernel size")
    parser.add_argument("--expand", type=int, default=2, help="Expansion ratio in GraphSSM")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    
    # NEW: distance metric CLI argument
    parser.add_argument("--distance", type=str, default="cosine",
                        choices=["cosine", "euclidean", "manhattan", "gaussian", "norm2"],
                        help="Distance formula to use in GraphSSM")
    
    # Inference specific
    parser.add_argument("--model_path", type=str, default=os.path.expanduser("~/data/checkpoints/best_model.pth"), help="Path to pre-trained model")
    parser.add_argument("--save_predictions", action="store_true", help="Save predictions to file")
    parser.add_argument("--mode", type=str, default="inference", choices=["inference", "train", "example"], help="Mode: inference, train, or example")
    parser.add_argument("--example", action="store_true", help="Run simple example like main.py")
    
    # TFLOPS profiling specific
    parser.add_argument("--warmup_batches", type=int, default=5, help="Number of warmup batches for accurate timing")
    parser.add_argument("--save_metrics", action="store_true", help="Save performance metrics to file")
    parser.add_argument("--profile_detailed", action="store_true", help="Enable detailed profiling with PyTorch profiler")
    parser.add_argument("--profile_steps", type=int, default=10, help="Number of steps to profile in detailed mode")
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    args = build_argparser()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    print("=" * 60)
    print("GraphSSM Time Series Forecasting")
    print("=" * 60)
    print(f"Mode: {args.mode.upper()}")
    print(f"Dataset: {args.data}")
    print(f"Data path: {os.path.join(args.root_path, args.data_path)}")
    print(f"Sequence length: {args.seq_len}, Prediction length: {args.pred_len}")
    print(f"Model: {args.model}")
    print(f"Distance formula: {args.distance}")  # NEW
    print("=" * 60)
    
    # Check if data file exists
    data_file_path = os.path.join(args.root_path, args.data_path)
    if not os.path.exists(data_file_path):
        print(f"ERROR: Data file not found at {data_file_path}")
        print("Please download the ETT dataset and place it in the correct location.")
        exit(1)
    
    if args.mode == "inference" or args.example:
        if args.example:
            print("\nRunning simple example...")
            run_example()
        else:
            print("\nStarting inference...")
            predictions, targets = inference(args)
            print("\nInference completed!")
    else:
        print("\nStarting training...")
        print("Training mode not implemented in this inference-only version.")