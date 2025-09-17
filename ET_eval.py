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
    # Each output element requires input_features multiplications and additions
    # Plus bias addition if present
    flops_per_output = input_features * 2  # multiply + add
    if bias:
        flops_per_output += 1  # bias addition
    
    total_outputs = batch_size * seq_len * output_features
    return total_outputs * flops_per_output


def count_normalization_flops(input_shape: tuple) -> int:
    """Count FLOPs for normalization operations"""
    batch_size, seq_len, features = input_shape
    # Mean calculation: seq_len additions per feature
    mean_flops = batch_size * features * seq_len
    
    # Variance calculation: (x - mean)^2 for each element
    variance_flops = batch_size * seq_len * features * 3  # subtract, square, add
    
    # Square root: 1 operation per feature
    sqrt_flops = batch_size * features
    
    # Division: 1 operation per element
    div_flops = batch_size * seq_len * features
    
    return mean_flops + variance_flops + sqrt_flops + div_flops


def estimate_graphssm_flops(input_shape: tuple, d_model: int, d_state: int, d_conv: int, expand: int) -> int:
    """Estimate FLOPs for GraphSSM operations"""
    batch_size, seq_len, _ = input_shape
    
    # This is a rough estimate - GraphSSM involves complex operations
    # We'll estimate based on typical SSM operations
    
    # Input projection (if any)
    input_proj_flops = batch_size * seq_len * d_model * d_model * 2
    
    # State space operations (simplified)
    # Each timestep processes d_state states
    state_flops = batch_size * seq_len * d_state * d_state * 2
    
    # Convolution operations
    conv_flops = batch_size * seq_len * d_model * d_conv * 2
    
    # Output projection
    output_proj_flops = batch_size * seq_len * d_model * d_model * 2
    
    # Expansion factor
    total_flops = (input_proj_flops + state_flops + conv_flops + output_proj_flops) * expand
    
    return int(total_flops)


def count_model_flops(model: nn.Module, input_shape: tuple, args: argparse.Namespace) -> int:
    """Count total FLOPs for the TimeSeriesForecaster model"""
    batch_size, seq_len, enc_in = input_shape
    total_flops = 0
    
    # Input embedding FLOPs
    input_embedding_flops = count_linear_flops(input_shape, args.d_model, bias=False)
    total_flops += input_embedding_flops
    
    # Normalization FLOPs (before and after)
    norm_flops = count_normalization_flops(input_shape) * 2  # normalize + denormalize
    total_flops += norm_flops
    
    # GraphSSM FLOPs (estimated)
    graphssm_input_shape = (batch_size, seq_len, args.d_model)
    graphssm_flops = estimate_graphssm_flops(
        graphssm_input_shape, 
        args.d_model, 
        args.d_state, 
        args.d_conv, 
        args.expand
    )
    total_flops += graphssm_flops
    
    # Output projection FLOPs
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
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.c_out = c_out
        self.d_model = d_model
        
        # Simple embedding layer to project input features to d_model
        self.input_embedding = nn.Linear(enc_in, d_model)
        
        # Core GraphSSM from main.py - using the same initialization pattern as main.py
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
            use_fast_path=True
        )
        
        # Output projection to get predictions (per timestep)
        self.output_projection = nn.Linear(d_model, c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: [B, seq_len, enc_in]
        b, seq_len, enc_in = x_enc.shape
        
        # Normalize (similar to MambaTS)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc /= stdev
        
        # Embed input features: [B, seq_len, enc_in] -> [B, seq_len, d_model]
        embedded = self.input_embedding(x_enc)
        
        # Pass through GraphSSM following the pattern from main.py
        # The main.py example shows: output = model(x, context_len)
        # where x is [batch_size, seq_len, d_model] and context_len is an integer
        context_len = min(seq_len, 4)  # Use a reasonable context length like in main.py example
        processed = self.graph_ssm(embedded, context_len)
        
        # Take only the last pred_len timesteps for prediction
        # This ensures we predict the future, not the past
        if processed.shape[1] >= self.pred_len:
            # Take the last pred_len timesteps
            processed = processed[:, -self.pred_len:, :]
        else:
            # If sequence is shorter than pred_len, repeat the last timestep
            last_timestep = processed[:, -1:, :].repeat(1, self.pred_len, 1)
            processed = last_timestep
        
        # Project to output: [B, pred_len, d_model] -> [B, pred_len, c_out]
        output = self.output_projection(processed)
        
        # De-normalize
        output = output * (stdev[:, [0], :].repeat(1, self.pred_len, 1))
        output = output + (means[:, [0], :].repeat(1, self.pred_len, 1))
        
        return output


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion) -> Tuple[float, dict]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_trues = []
    
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            batch_x = batch_x.to(device).float()
            batch_y = batch_y.to(device).float()
            batch_x_mark = batch_x_mark.to(device).float()
            batch_y_mark = batch_y_mark.to(device).float()
            
            # Create decoder input (similar to MambaTS)
            dec_inp = torch.zeros_like(batch_y).float().to(device)
            
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
            loss = criterion(outputs, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            
            all_preds.append(outputs.detach().cpu().numpy())
            all_trues.append(batch_y.detach().cpu().numpy())
    
    # Compute metrics (similar to MambaTS)
    preds = np.concatenate(all_preds, axis=0)
    trues = np.concatenate(all_trues, axis=0)
    
    mae = np.mean(np.abs(preds - trues))
    mse = np.mean((preds - trues) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((preds - trues) / (trues + 1e-8)))
    
    metrics = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
    }
    
    return total_loss / max(len(loader.dataset), 1), metrics


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


def inference(args: argparse.Namespace) -> None:
    """Run inference on ETT dataset using pre-trained GraphSSM model"""
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    
    print("=" * 50)
    print("GraphSSM Inference on ETT Dataset")
    print("=" * 50)
    
    # Load test data
    print("Loading test data...")
    try:
        test_data, test_loader = data_provider(args, flag='test')
        print(f"Test dataset loaded: {len(test_data)} samples")
    except Exception as e:
        print(f"Error loading test data: {e}")
        return
    
    # Load pre-trained model
    model_path = args.checkpoints + "best_model.pth"
    model = load_pretrained_model(model_path, args, device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Using device: {device}")
    
    # Run detailed profiling if requested
    if args.profile_detailed:
        detailed_profiling(model, test_loader, args, device)
    
    # Initialize TFLOPS calculator
    tflops_calc = TFLOPSCalculator()
    
    # Run inference
    print("\nRunning inference...")
    all_predictions = []
    all_targets = []
    total_loss = 0.0
    criterion = nn.MSELoss()
    
    # Warmup runs for accurate timing
    if args.warmup_batches > 0:
        print(f"Running {args.warmup_batches} warmup batches...")
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                if i >= args.warmup_batches:
                    break
                batch_x = batch_x.to(device).float()
                batch_y = batch_y.to(device).float()
                batch_x_mark = batch_x_mark.to(device).float()
                batch_y_mark = batch_y_mark.to(device).float()
                dec_inp = torch.zeros_like(batch_y).float().to(device)
                _ = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    
    # Main inference loop with TFLOPS calculation
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.to(device).float()
            batch_y = batch_y.to(device).float()
            batch_x_mark = batch_x_mark.to(device).float()
            batch_y_mark = batch_y_mark.to(device).float()
            
            # Create decoder input (zeros for inference)
            dec_inp = torch.zeros_like(batch_y).float().to(device)
            
            # Start timing for this batch
            tflops_calc.start_timing()
            
            # Forward pass
            predictions = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
            # End timing for this batch
            batch_time = tflops_calc.end_timing()
            
            # Calculate FLOPs for this batch
            batch_flops = count_model_flops(model, batch_x.shape, args)
            tflops_calc.add_flops(batch_flops)
            
            # Extract only the prediction part from batch_y (remove label_len overlap)
            # batch_y contains [label_len + pred_len], we only want the pred_len part
            if batch_y.shape[1] > args.pred_len:
                # Remove the label_len part, keep only pred_len
                batch_y = batch_y[:, -args.pred_len:, :]
            
            # Debug: print shapes on first batch
            if i == 0:
                print(f"Debug - batch_x shape: {batch_x.shape}")
                print(f"Debug - batch_y shape (after trim): {batch_y.shape}")
                print(f"Debug - predictions shape: {predictions.shape}")
                print(f"Debug - Batch FLOPs: {batch_flops:,}")
                print(f"Debug - Batch time: {batch_time:.4f}s")
            
            # Calculate loss
            loss = criterion(predictions, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            
            # Store results
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
            
            if (i + 1) % 100 == 0:
                current_tflops = tflops_calc.calculate_tflops()
                print(f"Processed {i + 1}/{len(test_loader)} batches - Current TFLOPS: {current_tflops:.2f}")
    
    # Calculate final metrics
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    mae = np.mean(np.abs(predictions - targets))
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((predictions - targets) / (targets + 1e-8)))
    
    # Calculate final TFLOPS
    final_tflops = tflops_calc.calculate_tflops()
    avg_batch_time = tflops_calc.total_time / max(tflops_calc.batch_count, 1)
    total_flops = tflops_calc.total_flops
    
    print("\n" + "=" * 50)
    print("INFERENCE RESULTS")
    print("=" * 50)
    print(f"Test Loss (MSE): {total_loss / len(test_data):.6f}")
    print(f"MAE:  {mae:.6f}")
    print(f"MSE:  {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAPE: {mape:.6f}")
    print("=" * 50)
    
    print("\n" + "=" * 50)
    print("PERFORMANCE METRICS")
    print("=" * 50)
    print(f"Total FLOPs: {total_flops:,}")
    print(f"Total time: {tflops_calc.total_time:.4f}s")
    print(f"Average batch time: {avg_batch_time:.4f}s")
    print(f"Batches processed: {tflops_calc.batch_count}")
    print(f"TFLOPS: {final_tflops:.2f}")
    
    # GPU utilization estimation
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")
        
        # Get theoretical peak TFLOPS (rough estimates for common GPUs)
        gpu_peak_tflops = {
            'RTX 4090': 83.0,
            'RTX 4080': 48.7,
            'RTX 4070': 29.0,
            'A100': 312.0,
            'V100': 125.0,
            'T4': 8.1
        }
        
        peak_tflops = None
        for gpu_key in gpu_peak_tflops:
            if gpu_key in gpu_name:
                peak_tflops = gpu_peak_tflops[gpu_key]
                break
        
        if peak_tflops:
            utilization = (final_tflops / peak_tflops) * 100
            print(f"Theoretical peak TFLOPS: {peak_tflops:.1f}")
            print(f"Utilization: {utilization:.1f}%")
    
    print("=" * 50)
    
    # Save predictions if requested
    if args.save_predictions:
        save_path = f"predictions_{args.data}_seq{args.seq_len}_pred{args.pred_len}.npy"
        np.save(save_path, predictions)
        print(f"Predictions saved to {save_path}")
    
    # Save performance metrics if requested
    if args.save_metrics:
        metrics = {
            'tflops': final_tflops,
            'total_flops': total_flops,
            'total_time': tflops_calc.total_time,
            'avg_batch_time': avg_batch_time,
            'batch_count': tflops_calc.batch_count,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape
        }
        metrics_path = f"metrics_{args.data}_seq{args.seq_len}_pred{args.pred_len}.npy"
        np.save(metrics_path, metrics)
        print(f"Performance metrics saved to {metrics_path}")
    
    return predictions, targets


def detailed_profiling(model: nn.Module, test_loader: DataLoader, args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    """Run detailed profiling with PyTorch profiler for advanced TFLOPS analysis"""
    print("\n" + "=" * 50)
    print("DETAILED PROFILING")
    print("=" * 50)
    
    model.eval()
    profiler_activities = [ProfilerActivity.CPU]
    if device.type == 'cuda':
        profiler_activities.append(ProfilerActivity.CUDA)
    
    with profile(
        activities=profiler_activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                if i >= args.profile_steps:
                    break
                    
                batch_x = batch_x.to(device).float()
                batch_y = batch_y.to(device).float()
                batch_x_mark = batch_x_mark.to(device).float()
                batch_y_mark = batch_y_mark.to(device).float()
                dec_inp = torch.zeros_like(batch_y).float().to(device)
                
                with record_function("model_forward"):
                    predictions = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    
    # Analyze profiler results
    print("Profiler results:")
    print(prof.key_averages().table(sort_by="cuda_time_total" if device.type == 'cuda' else "cpu_time_total", row_limit=20))
    
    # Extract key metrics
    key_averages = prof.key_averages()
    total_cuda_time = sum([event.cuda_time_total for event in key_averages if event.cuda_time_total > 0])
    total_cpu_time = sum([event.cpu_time_total for event in key_averages if event.cpu_time_total > 0])
    
    profiling_results = {
        'total_cuda_time': total_cuda_time,
        'total_cpu_time': total_cpu_time,
        'profile_steps': args.profile_steps
    }
    
    print(f"Total CUDA time: {total_cuda_time:.2f}ms")
    print(f"Total CPU time: {total_cpu_time:.2f}ms")
    print("=" * 50)
    
    return profiling_results


def run_example():
    """Example usage following the exact pattern from main.py"""
    print("Running GraphSSM example following main.py pattern...")
    
    # Example hyperparameters (exactly like main.py)
    d_model = 16
    seq_len = 12
    batch_size = 2
    context_len = 4  # Or pass in a list, e.g., [4, 4] for each sample
    
    # Create random input tensor (exactly like main.py)
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Instantiate the GraphSSM layer (exactly like main.py)
    model = GraphSSM(d_model=d_model)
    
    # Forward pass (exactly like main.py)
    output = model(x, context_len)
    
    print("Input shape:", x.shape)  # (B, L, d_model)
    print("Output shape:", output.shape)  # (B, L, d_model)
    print("Example completed!")


def build_argparser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate core/graph_ssm on MambaTS time series forecasting")
    
    # Basic config (MambaTS style)
    parser.add_argument("--task_name", type=str, default="long_term_forecast", help="task name")
    parser.add_argument("--is_training", type=int, default=1, help="status")
    parser.add_argument("--model_id", type=str, default="test", help="model id")
    parser.add_argument("--model", type=str, default="GraphSSM", help="model name")
    parser.add_argument("--seed", type=int, default=3047, help="random seed")
    
    # Data loader (MambaTS style)
    parser.add_argument("--data", type=str, default="ETTm1", help="dataset type")
    parser.add_argument("--root_path", type=str, default=os.path.expanduser("/data/eval_pipelines/datasets/ETT-small"), help="root path of the data file")
    parser.add_argument("--data_path", type=str, default="ETTm1.csv", help="data file")
    parser.add_argument("--features", type=str, default="M", help="forecasting task")
    parser.add_argument("--target", type=str, default="OT", help="target feature")
    parser.add_argument("--freq", type=str, default="t", help="freq for time features encoding")
    parser.add_argument("--checkpoints", type=str, default=os.path.expanduser("~/data/checkpoints/"), help="location of model checkpoints")
    parser.add_argument("--visualization", type=str, default=os.path.expanduser("~/data/test_results"), help="location of model checkpoints")
    parser.add_argument("--results", type=str, default=os.path.expanduser("~/data/results"), help="location of model checkpoints")
    
    # Forecasting task
    parser.add_argument("--seq_len", type=int, default=48, help="input sequence length")
    parser.add_argument("--label_len", type=int, default=24, help="start token length")
    parser.add_argument("--pred_len", type=int, default=24, help="prediction sequence length")
    parser.add_argument("--seasonal_patterns", type=str, default="Monthly", help="subset for M4")
    parser.add_argument("--inverse", action="store_true", help="inverse output data", default=False)
    
    # Model define
    parser.add_argument("--top_k", type=int, default=5, help="for TimesBlock")
    parser.add_argument("--num_kernels", type=int, default=6, help="for Inception")
    parser.add_argument("--enc_in", type=int, default=7, help="encoder input size")
    parser.add_argument("--dec_in", type=int, default=7, help="decoder input size")
    parser.add_argument("--c_out", type=int, default=7, help="output size")
    parser.add_argument("--d_model", type=int, default=512, help="dimension of model")
    parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
    parser.add_argument("--e_layers", type=int, default=2, help="num of encoder layers")
    parser.add_argument("--d_layers", type=int, default=1, help="num of decoder layers")
    parser.add_argument("--d_ff", type=int, default=2048, help="dimension of fcn")
    parser.add_argument("--moving_avg", type=int, default=25, help="window size of moving average")
    parser.add_argument("--factor", type=int, default=1, help="attn factor")
    parser.add_argument("--distil", action="store_false", help="whether to use distilling in encoder", default=True)
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument("--embed", type=str, default="timeF", help="time features encoding")
    parser.add_argument("--activation", type=str, default="gelu", help="activation")
    parser.add_argument("--output_attention", action="store_true", help="whether to output attention in ecoder")
    
    # Optimization
    parser.add_argument("--num_workers", type=int, default=0, help="data loader num workers")
    parser.add_argument("--itr", type=int, default=1, help="experiments times")
    parser.add_argument("--train_epochs", type=int, default=10, help="train epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size of train input data")
    parser.add_argument("--patience", type=int, default=3, help="early stopping patience")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="optimizer learning rate")
    parser.add_argument("--des", type=str, default="test", help="exp description")
    parser.add_argument("--loss", type=str, default="MSE", help="loss function")
    parser.add_argument("--lradj", type=str, default="type1", help="adjust learning rate")
    parser.add_argument("--use_amp", action="store_true", help="use automatic mixed precision training", default=False)
    parser.add_argument("--optimizer", type=str, default="adam", help="optimizer")
    parser.add_argument("--no_lradj", action="store_true")
    parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
    parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
    parser.add_argument("--lradj_by_iter", action="store_true")
    parser.add_argument("--warmup_steps", default=0.1, type=float, help="warmup")
    parser.add_argument("--iters_per_epoch", default=None, type=str, help="warmup")
    
    # GPU
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--use_multi_gpu", action="store_true", help="use multiple gpus", default=False)
    parser.add_argument("--devices", type=str, default="0,1,2,3", help="device ids of multile gpus")
    
    # De-stationary projector params
    parser.add_argument("--p_hidden_dims", type=int, nargs="+", default=[128, 128], help="hidden layer dimensions of projector")
    parser.add_argument("--p_hidden_layers", type=int, default=2, help="number of hidden layers in projector")
    
    # PatchTST
    parser.add_argument("--patch_len", type=int, default=16, help="patch length")
    parser.add_argument("--stride", type=int, default=8, help="stride")
    
    # GraphSSM specific
    parser.add_argument("--d_state", type=int, default=16, help="GraphSSM state size")
    parser.add_argument("--d_conv", type=int, default=4, help="GraphSSM conv kernel size")
    parser.add_argument("--expand", type=int, default=2, help="Expansion ratio in GraphSSM")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    
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
    print("=" * 60)
    
    # Check if data file exists
    data_file_path = os.path.join(args.root_path, args.data_path)
    if not os.path.exists(data_file_path):
        print(f"ERROR: Data file not found at {data_file_path}")
        print("Please download the ETT dataset and place it in the correct location.")
        print("Download from: https://github.com/zhouhaoyi/ETDataset")
        print("Place the ETT-small folder in ~/data/datasets/ETDataset/")
        exit(1)
    
    # Run based on mode
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
        # Note: Training function would need to be re-added if needed
        print("Training mode not implemented in this inference-only version.")
        print("Use a separate training script for model training.")
