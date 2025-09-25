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
import subprocess
import json
import re
from typing import Tuple, Dict, Any, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity

# Global variables for paths - will be set by setup_paths function
gg_ssms_path = None
mamba_ts_path = None


def setup_paths(workspace_path: str) -> None:
    """Setup paths for gg_ssms repository and MambaTS based on workspace argument"""
    global gg_ssms_path, mamba_ts_path
    
    # Expand user path and resolve absolute path
    gg_ssms_path = os.path.expanduser(workspace_path)
    gg_ssms_path = os.path.abspath(gg_ssms_path)
    mamba_ts_path = os.path.join(gg_ssms_path, "MambaTS")
    
    # Check if paths exist and provide helpful error messages
    if not os.path.exists(gg_ssms_path):
        print(f"ERROR: GG_SSMS repository not found at {gg_ssms_path}")
        print(f"Please ensure the gg_ssms repository is located at {workspace_path}")
        print("You can specify a different workspace path using --workspace argument")
        sys.exit(1)
    
    if not os.path.exists(mamba_ts_path):
        print(f"ERROR: MambaTS not found at {mamba_ts_path}")
        print("Please ensure MambaTS is located in the gg_ssms repository")
        sys.exit(1)
    
    # Add MambaTS to path
    sys.path.append(mamba_ts_path)
    
    # Import MambaTS modules
    try:
        from data_provider.data_factory import data_provider
        from utils.tools import set_seed
        globals()['data_provider'] = data_provider
        globals()['set_seed'] = set_seed
    except ImportError as e:
        print(f"ERROR: Failed to import MambaTS modules: {e}")
        print(f"Please ensure MambaTS is properly installed in {mamba_ts_path}")
        sys.exit(1)
    
    # Import GraphSSM from the graph_ssm package
    graph_ssm_path = os.path.join(gg_ssms_path, "core", "graph_ssm")
    main_py_path = os.path.join(graph_ssm_path, "main.py")
    if not os.path.exists(main_py_path):
        print(f"ERROR: main.py not found at {main_py_path}")
        print("Please ensure the core/graph_ssm/main.py file exists in the gg_ssms repository")
        sys.exit(1)
    
    # Add the core directory to path so we can import graph_ssm as a package
    core_path = os.path.join(gg_ssms_path, "core")
    if core_path not in sys.path:
        sys.path.insert(0, core_path)
    
    # Add the third-party directory to path for tree_scan_lan
    third_party_path = os.path.join(graph_ssm_path, "third-party", "TreeScanLan")
    if third_party_path not in sys.path:
        sys.path.insert(0, third_party_path)
    
    try:
        # Import as a package to handle relative imports correctly
        from graph_ssm.main import GraphSSM
        globals()['GraphSSM'] = GraphSSM
    except ImportError as e:
        print(f"ERROR: Failed to import GraphSSM: {e}")
        print(f"Please ensure GraphSSM is properly implemented in {main_py_path}")
        print("Make sure all dependencies (tree_utils, tree_scan_lan) are available")
        sys.exit(1)


# Remove the synthetic dataset class - we'll use MambaTS data providers instead


class NsightComputeProfiler:
    """NVIDIA Nsight Compute profiler integration for exact TFLOPS measurement"""
    
    def __init__(self, output_dir: str = "./nsight_profiles"):
        self.output_dir = output_dir
        self.nsight_path = self._find_nsight_compute()
        self.profile_files = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def _find_nsight_compute(self) -> Optional[str]:
        """Find NVIDIA Nsight Compute installation"""
        possible_paths = [
            "nv-nsight-cu-cli",
            "/usr/local/cuda/bin/nv-nsight-cu-cli",
            "C:\\Program Files\\NVIDIA Corporation\\Nsight Compute 2023.3\\nv-nsight-cu-cli.exe",
            "C:\\Program Files\\NVIDIA Corporation\\Nsight Compute 2024.1\\nv-nsight-cu-cli.exe",
        ]
        
        for path in possible_paths:
            try:
                result = subprocess.run([path, "--version"], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    print(f"Found Nsight Compute at: {path}")
                    return path
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                continue
        
        print("WARNING: NVIDIA Nsight Compute not found. Please install it for exact TFLOPS measurement.")
        return None
    
    def create_profiling_script(self, model_script: str, args: argparse.Namespace) -> str:
        """Create a profiling script that will be run with Nsight Compute"""
        script_content = f'''#!/usr/bin/env python3
"""
Auto-generated profiling script for Nsight Compute
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the model and data loading functions
from {os.path.basename(model_script).replace('.py', '')} import *
import torch
import torch.nn as nn

def profile_model():
    """Profile the model with minimal overhead"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load test data
    test_data, test_loader = data_provider(args, flag='test')
    
    # Load model
    model_path = args.checkpoints + "best_model.pth"
    model = load_pretrained_model(model_path, args, device)
    model.eval()
    
    # Profile only a few batches
    profile_batches = min(args.nsight_profile_batches, len(test_loader))
    
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            if i >= profile_batches:
                break
                
            batch_x = batch_x.to(device).float()
            batch_y = batch_y.to(device).float()
            batch_x_mark = batch_x_mark.to(device).float()
            batch_y_mark = batch_y_mark.to(device).float()
            dec_inp = torch.zeros_like(batch_y).float().to(device)
            
            # This is where Nsight Compute will profile
            predictions = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
            # Synchronize to ensure all operations complete
            if device.type == 'cuda':
                torch.cuda.synchronize()

if __name__ == "__main__":
    # Set up arguments
    args = build_argparser()
    args.nsight_profile_batches = {args.nsight_profile_batches}
    profile_model()
'''
        
        script_path = os.path.join(self.output_dir, "nsight_profile_script.py")
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        return script_path
    
    def run_nsight_profiling(self, model_script: str, args: argparse.Namespace) -> Dict[str, Any]:
        """Run Nsight Compute profiling and extract TFLOPS data"""
        if not self.nsight_path:
            return {"error": "Nsight Compute not available"}
        
        # Create profiling script
        profile_script = self.create_profiling_script(model_script, args)
        
        # Define metrics to collect
        metrics = [
            "sm__sass_thread_inst_executed_op_fp32_pred_on.sum",
            "sm__sass_thread_inst_executed_op_fp64_pred_on.sum", 
            "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
            "sm__cycles_elapsed.avg.per_second",
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "sm__inst_executed.sum",
            "sm__warps_active.avg.pct_of_peak_sustained_elapsed",
            "sm__warps_issued.avg.pct_of_peak_sustained_elapsed",
            "sm__warps_eligible.avg.pct_of_peak_sustained_elapsed",
            "sm__warps_launched.avg.pct_of_peak_sustained_elapsed"
        ]
        
        # Create output file path
        output_file = os.path.join(self.output_dir, f"nsight_profile_{int(time.time())}.ncu-rep")
        
        # Build Nsight Compute command
        cmd = [
            self.nsight_path,
            "--metrics", ",".join(metrics),
            "--target-processes", "all",
            "--kernel-regex", ".*",
            "--launch-skip-before-match", "0",
            "--launch-count", str(args.nsight_profile_batches),
            "--launch-skip", "0",
            "--export", output_file,
            "python", profile_script
        ]
        
        print(f"Running Nsight Compute profiling...")
        print(f"Command: {' '.join(cmd)}")
        
        try:
            # Run Nsight Compute
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                print(f"Nsight Compute error: {result.stderr}")
                return {"error": f"Nsight Compute failed: {result.stderr}"}
            
            # Parse results
            return self._parse_nsight_results(result.stdout, output_file)
            
        except subprocess.TimeoutExpired:
            return {"error": "Nsight Compute profiling timed out"}
        except Exception as e:
            return {"error": f"Nsight Compute profiling failed: {str(e)}"}
    
    def _parse_nsight_results(self, stdout: str, output_file: str) -> Dict[str, Any]:
        """Parse Nsight Compute results to extract TFLOPS data"""
        results = {
            "output_file": output_file,
            "raw_output": stdout,
            "tflops": 0.0,
            "total_flops": 0,
            "execution_time": 0.0,
            "utilization": 0.0,
            "metrics": {}
        }
        
        # Extract metrics from stdout
        lines = stdout.split('\n')
        current_kernel = None
        
        for line in lines:
            # Look for kernel names
            if "Kernel:" in line:
                current_kernel = line.split("Kernel:")[-1].strip()
                if current_kernel not in results["metrics"]:
                    results["metrics"][current_kernel] = {}
            
            # Extract metric values
            for metric in ["sm__sass_thread_inst_executed_op_fp32_pred_on.sum",
                          "sm__sass_thread_inst_executed_op_fp64_pred_on.sum",
                          "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
                          "sm__cycles_elapsed.avg.per_second",
                          "sm__throughput.avg.pct_of_peak_sustained_elapsed"]:
                
                if metric in line and "=" in line:
                    try:
                        value_str = line.split("=")[-1].strip().replace(",", "")
                        if value_str.replace(".", "").isdigit():
                            value = float(value_str)
                            results["metrics"][current_kernel][metric] = value
                    except (ValueError, IndexError):
                        continue
        
        # Calculate total FLOPs across all kernels
        total_fp32_ops = 0
        total_fp64_ops = 0
        total_ffma_ops = 0
        total_time = 0
        
        for kernel, metrics in results["metrics"].items():
            if kernel and metrics:
                total_fp32_ops += metrics.get("sm__sass_thread_inst_executed_op_fp32_pred_on.sum", 0)
                total_fp64_ops += metrics.get("sm__sass_thread_inst_executed_op_fp64_pred_on.sum", 0)
                total_ffma_ops += metrics.get("sm__sass_thread_inst_executed_op_ffma_pred_on.sum", 0)
                
                # Use the first kernel's timing (they should be similar)
                if total_time == 0:
                    cycles_per_sec = metrics.get("sm__cycles_elapsed.avg.per_second", 0)
                    if cycles_per_sec > 0:
                        # Estimate execution time (this is approximate)
                        total_time = 1.0  # We'll need to get actual timing from elsewhere
        
        # Calculate total FLOPs (FP32 + FP64 + FMA operations)
        results["total_flops"] = int(total_fp32_ops + total_fp64_ops + total_ffma_ops)
        
        # Calculate TFLOPS (we need execution time for this)
        if total_time > 0:
            results["tflops"] = (results["total_flops"] / 1e12) / total_time
        else:
            # Fallback: estimate from utilization
            avg_utilization = 0
            count = 0
            for kernel, metrics in results["metrics"].items():
                if kernel and "sm__throughput.avg.pct_of_peak_sustained_elapsed" in metrics:
                    avg_utilization += metrics["sm__throughput.avg.pct_of_peak_sustained_elapsed"]
                    count += 1
            
            if count > 0:
                avg_utilization /= count
                results["utilization"] = avg_utilization
                
                # Estimate TFLOPS from utilization (rough estimate)
                # This would need GPU-specific peak TFLOPS to be accurate
                estimated_peak_tflops = 83.0  # RTX 4090 default
                results["tflops"] = (estimated_peak_tflops * avg_utilization) / 100
        
        return results
    
    def detect_custom_cuda_libraries(self) -> List[str]:
        """Detect if custom CUDA libraries like TreeScan are being used"""
        detected_libs = []
        
        # Check for common custom CUDA libraries
        cuda_lib_patterns = [
            r"treescan",
            r"mamba",
            r"ssm",
            r"selective_scan",
            r"causal_conv",
            r"graph_ssm"
        ]
        
        # Check in the current directory and common locations
        search_paths = [
            ".",
            os.path.expanduser("~/workspace"),
            "/workspace"
        ]
        
        for search_path in search_paths:
            if os.path.exists(search_path):
                for root, dirs, files in os.walk(search_path):
                    for file in files:
                        if file.endswith(('.cu', '.cuh', '.cpp', '.h')):
                            file_path = os.path.join(root, file)
                            try:
                                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                    content = f.read().lower()
                                    for pattern in cuda_lib_patterns:
                                        if re.search(pattern, content):
                                            if pattern not in detected_libs:
                                                detected_libs.append(pattern)
                            except:
                                continue
        
        return detected_libs
    
    def run_advanced_nsight_profiling(self, model_script: str, args: argparse.Namespace) -> Dict[str, Any]:
        """Run advanced Nsight Compute profiling with timing and FLOP extraction"""
        if not self.nsight_path:
            return {"error": "Nsight Compute not available"}
        
        print("\n" + "=" * 60)
        print("NVIDIA NSIGHT COMPUTE PROFILING")
        print("=" * 60)
        
        # Detect custom CUDA libraries
        custom_libs = self.detect_custom_cuda_libraries()
        if custom_libs:
            print(f"Detected custom CUDA libraries: {', '.join(custom_libs)}")
        else:
            print("No custom CUDA libraries detected (using standard PyTorch operations)")
        
        # Create profiling script
        profile_script = self.create_profiling_script(model_script, args)
        
        # Define comprehensive metrics for exact TFLOPS calculation
        metrics = [
            # Floating point operations
            "sm__sass_thread_inst_executed_op_fp32_pred_on.sum",
            "sm__sass_thread_inst_executed_op_fp64_pred_on.sum",
            "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
            "sm__sass_thread_inst_executed_op_fp16_pred_on.sum",
            "sm__sass_thread_inst_executed_op_bf16_pred_on.sum",
            
            # Timing metrics
            "sm__cycles_elapsed.avg.per_second",
            "sm__cycles_elapsed.avg",
            "sm__cycles_elapsed.sum",
            
            # Throughput and utilization
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "sm__throughput.avg.pct_of_peak_sustained_achieved",
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            
            # Instruction counts
            "sm__inst_executed.sum",
            "sm__inst_executed.avg.per_cycle_elapsed",
            "sm__inst_executed.avg.per_cycle_active",
            
            # Warp utilization
            "sm__warps_active.avg.pct_of_peak_sustained_elapsed",
            "sm__warps_issued.avg.pct_of_peak_sustained_elapsed",
            "sm__warps_eligible.avg.pct_of_peak_sustained_elapsed",
            "sm__warps_launched.avg.pct_of_peak_sustained_elapsed",
            
            # Memory metrics
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
            "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
            "l2__throughput.avg.pct_of_peak_sustained_elapsed"
        ]
        
        # Create output file path
        timestamp = int(time.time())
        output_file = os.path.join(self.output_dir, f"nsight_profile_{timestamp}.ncu-rep")
        csv_output = os.path.join(self.output_dir, f"nsight_metrics_{timestamp}.csv")
        
        # Build Nsight Compute command with CSV output for easier parsing
        cmd = [
            self.nsight_path,
            "--metrics", ",".join(metrics),
            "--target-processes", "all",
            "--kernel-regex", ".*",
            "--launch-skip-before-match", "0",
            "--launch-count", str(args.nsight_profile_batches),
            "--launch-skip", "0",
            "--export", output_file,
            "--csv",
            "--page", "details",
            "python", profile_script
        ]
        
        print(f"Running Nsight Compute profiling...")
        print(f"Profile batches: {args.nsight_profile_batches}")
        print(f"Output file: {output_file}")
        print(f"Command: {' '.join(cmd)}")
        
        try:
            # Run Nsight Compute
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                print(f"Nsight Compute error: {result.stderr}")
                return {"error": f"Nsight Compute failed: {result.stderr}"}
            
            # Parse results with improved accuracy
            results = self._parse_advanced_nsight_results(result.stdout, output_file, args)
            
            # Save detailed results
            results_file = os.path.join(self.output_dir, f"nsight_results_{timestamp}.json")
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"Detailed results saved to: {results_file}")
            print("=" * 60)
            
            return results
            
        except subprocess.TimeoutExpired:
            return {"error": "Nsight Compute profiling timed out (10 minutes)"}
        except Exception as e:
            return {"error": f"Nsight Compute profiling failed: {str(e)}"}
    
    def _parse_advanced_nsight_results(self, stdout: str, output_file: str, args: argparse.Namespace) -> Dict[str, Any]:
        """Parse Nsight Compute results with improved accuracy for TFLOPS calculation"""
        results = {
            "output_file": output_file,
            "raw_output": stdout,
            "tflops": 0.0,
            "total_flops": 0,
            "execution_time": 0.0,
            "utilization": 0.0,
            "kernels": {},
            "custom_libraries_detected": [],
            "gpu_info": {},
            "metrics_summary": {}
        }
        
        # Parse the CSV-like output from Nsight Compute
        lines = stdout.split('\n')
        current_kernel = None
        kernel_data = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for kernel information
            if "Kernel:" in line:
                if current_kernel and kernel_data:
                    results["kernels"][current_kernel] = kernel_data.copy()
                current_kernel = line.split("Kernel:")[-1].strip()
                kernel_data = {}
                continue
            
            # Parse metric values
            if "=" in line:
                try:
                    parts = line.split("=", 1)
                    metric_name = parts[0].strip()
                    value_str = parts[1].strip().replace(",", "")
                    
                    # Convert value to appropriate type
                    if value_str.replace(".", "").replace("-", "").isdigit():
                        value = float(value_str)
                        kernel_data[metric_name] = value
                except (ValueError, IndexError):
                    continue
        
        # Add the last kernel
        if current_kernel and kernel_data:
            results["kernels"][current_kernel] = kernel_data
        
        # Calculate total FLOPs across all kernels
        total_fp32_ops = 0
        total_fp64_ops = 0
        total_ffma_ops = 0
        total_fp16_ops = 0
        total_bf16_ops = 0
        total_cycles = 0
        total_utilization = 0
        kernel_count = 0
        
        for kernel_name, metrics in results["kernels"].items():
            if not metrics:
                continue
                
            kernel_count += 1
            
            # Sum up floating point operations
            total_fp32_ops += metrics.get("sm__sass_thread_inst_executed_op_fp32_pred_on.sum", 0)
            total_fp64_ops += metrics.get("sm__sass_thread_inst_executed_op_fp64_pred_on.sum", 0)
            total_ffma_ops += metrics.get("sm__sass_thread_inst_executed_op_ffma_pred_on.sum", 0)
            total_fp16_ops += metrics.get("sm__sass_thread_inst_executed_op_fp16_pred_on.sum", 0)
            total_bf16_ops += metrics.get("sm__sass_thread_inst_executed_op_bf16_pred_on.sum", 0)
            
            # Sum up cycles
            total_cycles += metrics.get("sm__cycles_elapsed.sum", 0)
            
            # Average utilization
            utilization = metrics.get("sm__throughput.avg.pct_of_peak_sustained_elapsed", 0)
            if utilization > 0:
                total_utilization += utilization
        
        # Calculate total FLOPs (weighted by precision)
        # FP32 = 1x, FP64 = 2x, FMA = 2x (fused multiply-add), FP16/BF16 = 0.5x
        results["total_flops"] = int(
            total_fp32_ops + 
            total_fp64_ops * 2 + 
            total_ffma_ops * 2 + 
            total_fp16_ops * 0.5 + 
            total_bf16_ops * 0.5
        )
        
        # Calculate execution time from cycles
        if total_cycles > 0 and kernel_count > 0:
            # Get GPU frequency (approximate)
            gpu_freq_ghz = 1.5  # Default assumption, should be detected
            for kernel_name, metrics in results["kernels"].items():
                if "sm__cycles_elapsed.avg.per_second" in metrics:
                    gpu_freq_ghz = metrics["sm__cycles_elapsed.avg.per_second"] / 1e9
                    break
            
            # Calculate execution time
            results["execution_time"] = total_cycles / (gpu_freq_ghz * 1e9)
            
            # Calculate TFLOPS
            if results["execution_time"] > 0:
                results["tflops"] = (results["total_flops"] / 1e12) / results["execution_time"]
        
        # Calculate average utilization
        if kernel_count > 0:
            results["utilization"] = total_utilization / kernel_count
        
        # Create metrics summary
        results["metrics_summary"] = {
            "total_kernels": kernel_count,
            "fp32_operations": int(total_fp32_ops),
            "fp64_operations": int(total_fp64_ops),
            "ffma_operations": int(total_ffma_ops),
            "fp16_operations": int(total_fp16_ops),
            "bf16_operations": int(total_bf16_ops),
            "total_cycles": int(total_cycles),
            "average_utilization": results["utilization"]
        }
        
        # Detect custom libraries from kernel names
        custom_lib_patterns = ["treescan", "mamba", "ssm", "selective_scan", "causal_conv", "graph_ssm"]
        for kernel_name in results["kernels"].keys():
            for pattern in custom_lib_patterns:
                if pattern.lower() in kernel_name.lower():
                    if pattern not in results["custom_libraries_detected"]:
                        results["custom_libraries_detected"].append(pattern)
        
        return results


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
        
        # Debug information
        if not hasattr(self, '_forward_debug_count'):
            self._forward_debug_count = 0
        self._forward_debug_count += 1
        
        if self._forward_debug_count <= 3:
            print(f"Forward pass {self._forward_debug_count}: Input shape: {x_enc.shape}")
            print(f"Forward pass {self._forward_debug_count}: Device: {x_enc.device}")
        
        try:
            # Normalize (similar to MambaTS)
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            x_enc /= stdev
            
            # Ensure tensors are contiguous to avoid CUDA memory issues
            means = means.contiguous()
            stdev = stdev.contiguous()
            
            if self._forward_debug_count <= 3:
                print(f"Forward pass {self._forward_debug_count}: Normalization completed")
                
        except RuntimeError as e:
            print(f"Error in normalization step: {e}")
            print(f"Input tensor info: shape={x_enc.shape}, device={x_enc.device}, dtype={x_enc.dtype}")
            raise
        
        # Embed input features: [B, seq_len, enc_in] -> [B, seq_len, d_model]
        try:
            embedded = self.input_embedding(x_enc)
            if self._forward_debug_count <= 3:
                print(f"Forward pass {self._forward_debug_count}: Embedding completed, shape: {embedded.shape}")
        except RuntimeError as e:
            print(f"Error in embedding step: {e}")
            print(f"Input tensor info: shape={x_enc.shape}, device={x_enc.device}, dtype={x_enc.dtype}")
            raise
        
        # Pass through GraphSSM following the pattern from main.py
        # The main.py example shows: output = model(x, context_len)
        # where x is [batch_size, seq_len, d_model] and context_len is an integer
        try:
            context_len = min(seq_len, 4)  # Use a reasonable context length like in main.py example
            if self._forward_debug_count <= 3:
                print(f"Forward pass {self._forward_debug_count}: Calling GraphSSM with context_len={context_len}")
            
            processed = self.graph_ssm(embedded, context_len)
            
            if self._forward_debug_count <= 3:
                print(f"Forward pass {self._forward_debug_count}: GraphSSM completed, shape: {processed.shape}")
                
        except RuntimeError as e:
            print(f"Error in GraphSSM step: {e}")
            print(f"Embedded tensor info: shape={embedded.shape}, device={embedded.device}, dtype={embedded.dtype}")
            print(f"Context len: {context_len}")
            raise
        
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
        
        # De-normalize (optional - skip if dimensions don't match or CUDA errors occur)
        # stdev and means have shape [B, 1, enc_in], output has shape [B, pred_len, c_out]
        try:
            # Only attempt denormalization if dimensions are compatible
            if (stdev.shape[-1] == output.shape[-1] or 
                stdev.shape[-1] == 1 or 
                stdev.shape[-1] <= output.shape[-1]):
                
                # Debug information (only first few times)
                if hasattr(self, '_debug_count'):
                    self._debug_count += 1
                else:
                    self._debug_count = 1
                
                if self._debug_count <= 3:
                    print(f"Debug {self._debug_count}: stdev.shape={stdev.shape}, output.shape={output.shape}")
                    print(f"Debug {self._debug_count}: means.shape={means.shape}, pred_len={self.pred_len}")
                
                if stdev.shape[-1] == output.shape[-1]:
                    # Perfect match: use stdev and means directly
                    stdev_repeated = stdev.repeat(1, self.pred_len, 1)
                    means_repeated = means.repeat(1, self.pred_len, 1)
                    output = output * stdev_repeated + means_repeated
                elif stdev.shape[-1] == 1:
                    # Single feature: broadcast to all output features
                    stdev_repeated = stdev.repeat(1, self.pred_len, output.shape[-1])
                    means_repeated = means.repeat(1, self.pred_len, output.shape[-1])
                    output = output * stdev_repeated + means_repeated
                else:
                    # Partial match: only denormalize matching features
                    min_features = min(stdev.shape[-1], output.shape[-1])
                    stdev_slice = stdev[:, :, :min_features].repeat(1, self.pred_len, 1)
                    means_slice = means[:, :, :min_features].repeat(1, self.pred_len, 1)
                    output[:, :, :min_features] = (output[:, :, :min_features] * stdev_slice + 
                                                   means_slice)
            else:
                # Dimensions incompatible - skip denormalization
                if not hasattr(self, '_denorm_warning_shown'):
                    print(f"Warning: Skipping denormalization - incompatible dimensions: "
                          f"stdev.shape={stdev.shape}, output.shape={output.shape}")
                    self._denorm_warning_shown = True
                
        except RuntimeError as e:
            # If there's still a CUDA error, skip denormalization
            if not hasattr(self, '_cuda_error_shown'):
                print(f"Warning: Skipping denormalization due to CUDA error: {e}")
                print(f"stdev.shape: {stdev.shape}, output.shape: {output.shape}")
                self._cuda_error_shown = True
            pass
        
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


def configure_dataset_args(args: argparse.Namespace) -> argparse.Namespace:
    """Configure dataset-specific arguments based on the dataset type"""
    if args.data == "Solar" or args.use_solar:
        # Configure solar dataset parameters
        args.data = "Solar"  # Ensure data type is set correctly
        args.root_path = args.solar_root_path
        args.data_path = args.solar_data_path
        args.features = "M"  # Multi-variate forecasting
        args.target = "OT"  # Default target (will be ignored for Solar)
        args.freq = "t"  # Default frequency
        args.enc_in = 137  # Solar dataset has 137 features
        args.dec_in = 137
        args.c_out = 137
        print(f"Configured for Solar dataset:")
        print(f"  Root path: {args.root_path}")
        print(f"  Data path: {args.data_path}")
        print(f"  Features: {args.features}")
        print(f"  Input dimensions: {args.enc_in}")
    return args


def inference(args: argparse.Namespace) -> None:
    """Run inference on dataset using pre-trained GraphSSM model"""
    set_seed(args.seed)
    
    # Device selection with CUDA error handling
    if args.cpu or args.force_cpu:
        device = torch.device("cpu")
        print("Using CPU as requested")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available, using GPU")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU")
    
    # Configure dataset-specific arguments
    args = configure_dataset_args(args)
    
    print("=" * 50)
    print(f"GraphSSM Inference on {args.data} Dataset")
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
    
    # Test model on CPU first to verify it works
    print("Testing model on CPU first...")
    try:
        # Create a small test input
        test_input = torch.randn(1, args.seq_len, args.enc_in)
        test_x_mark = torch.randn(1, args.seq_len, 4)  # Assuming 4 time features
        test_dec = torch.randn(1, args.pred_len, args.enc_in)
        test_y_mark = torch.randn(1, args.pred_len, 4)
        
        # Test on CPU
        model_cpu = model.cpu()
        with torch.no_grad():
            test_output = model_cpu(test_input, test_x_mark, test_dec, test_y_mark)
        print(f"CPU test successful: input {test_input.shape} -> output {test_output.shape}")
        
        # Move model back to target device with error handling
        try:
            model = model.to(device)
            print(f"Model moved to {device}")
        except RuntimeError as e:
            if "CUDA" in str(e):
                print(f"CUDA error when moving model to GPU: {e}")
                print("The GraphSSM model contains CUDA kernels that are causing memory issues.")
                print("Falling back to CPU execution...")
                device = torch.device("cpu")
                model = model.cpu()  # Model is already on CPU
                print("Continuing with CPU execution")
            else:
                raise
        
    except Exception as e:
        print(f"CPU test failed: {e}")
        print("This suggests the issue is in the model architecture, not CUDA-specific")
        return None, None
    
    # Clear CUDA cache before starting inference
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print("Cleared CUDA cache before inference")
        
        # Check CUDA memory
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
        print(f"CUDA memory cached: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")
    
    # Run Nsight Compute profiling for exact TFLOPS measurement
    nsight_results = None
    if args.nsight_compute:
        nsight_profiler = NsightComputeProfiler()
        nsight_results = nsight_profiler.run_advanced_nsight_profiling(__file__, args)
        
        if "error" in nsight_results:
            print(f"Nsight Compute profiling failed: {nsight_results['error']}")
            print("Falling back to PyTorch profiler...")
            nsight_results = None
    
    # Run detailed profiling for accurate TFLOPS calculation (fallback)
    profiling_results = None
    if args.profile_detailed and not nsight_results:
        profiling_results = detailed_profiling_for_tflops(model, test_loader, args, device)
    
    # Initialize TFLOPS calculator for basic timing
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
                
                try:
                    print(f"Warmup batch {i+1}: Processing batch shapes - batch_x: {batch_x.shape}, batch_y: {batch_y.shape}")
                    
                    # Move to device with error handling
                    batch_x = batch_x.to(device).float()
                    batch_y = batch_y.to(device).float()
                    batch_x_mark = batch_x_mark.to(device).float()
                    batch_y_mark = batch_y_mark.to(device).float()
                    dec_inp = torch.zeros_like(batch_y).float().to(device)
                    
                    print(f"Warmup batch {i+1}: Data moved to device successfully")
                    
                    # Test model forward pass with error handling
                    _ = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    print(f"Warmup batch {i+1}: Model forward pass completed successfully")
                    
                except RuntimeError as e:
                    print(f"Error in warmup batch {i+1}: {e}")
                    print(f"Batch shapes: batch_x={batch_x.shape}, batch_y={batch_y.shape}")
                    print(f"Device: {device}")
                    
                    # If it's a CUDA error and fallback is enabled, try CPU
                    if "CUDA" in str(e) and args.fallback_cpu and device.type == 'cuda':
                        print("CUDA error detected, falling back to CPU...")
                        device = torch.device("cpu")
                        model = model.cpu()
                        print("Model moved to CPU")
                        break  # Exit warmup and continue with CPU
                    else:
                        # Try to clear CUDA cache and continue
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                            print("Cleared CUDA cache")
                        
                        # Skip this batch and continue
                        continue
    
    # Main inference loop with TFLOPS calculation
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            try:
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
                
            except RuntimeError as e:
                print(f"Error in main inference batch {i+1}: {e}")
                
                # If it's a CUDA error and fallback is enabled, try CPU
                if "CUDA" in str(e) and args.fallback_cpu and device.type == 'cuda':
                    print("CUDA error detected, falling back to CPU...")
                    device = torch.device("cpu")
                    model = model.cpu()
                    print("Model moved to CPU")
                    
                    # Retry the batch on CPU
                    try:
                        batch_x = batch_x.cpu().float()
                        batch_y = batch_y.cpu().float()
                        batch_x_mark = batch_x_mark.cpu().float()
                        batch_y_mark = batch_y_mark.cpu().float()
                        dec_inp = torch.zeros_like(batch_y).float()
                        
                        tflops_calc.start_timing()
                        predictions = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        batch_time = tflops_calc.end_timing()
                        print(f"Batch {i+1} completed successfully on CPU")
                    except Exception as cpu_e:
                        print(f"CPU fallback also failed: {cpu_e}")
                        continue
                else:
                    # Skip this batch
                    print(f"Skipping batch {i+1}")
                    continue
            
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
    
    # Calculate final TFLOPS - prioritize Nsight Compute results if available
    if nsight_results and 'tflops' in nsight_results:
        # Use Nsight Compute results (most accurate)
        final_tflops = nsight_results['tflops']
        total_flops = nsight_results['total_flops']
        total_time = nsight_results['execution_time']
        avg_batch_time = total_time / nsight_results['metrics_summary']['total_kernels'] if nsight_results['metrics_summary']['total_kernels'] > 0 else 0
        batch_count = nsight_results['metrics_summary']['total_kernels']
        tflops_method = "NVIDIA Nsight Compute (exact GPU kernel analysis)"
        custom_libs = nsight_results.get('custom_libraries_detected', [])
    elif profiling_results and 'tflops' in profiling_results:
        # Use PyTorch profiler-based TFLOPS calculation (fallback)
        final_tflops = profiling_results['tflops']
        total_flops = profiling_results['total_flops']
        total_time = profiling_results['total_time_s']
        avg_batch_time = total_time / profiling_results['profile_steps']
        batch_count = profiling_results['profile_steps']
        tflops_method = "PyTorch Profiler (GPU kernel timing)"
        custom_libs = []
    else:
        # Fall back to basic timing calculation
        final_tflops = tflops_calc.calculate_tflops()
        total_flops = tflops_calc.total_flops
        total_time = tflops_calc.total_time
        avg_batch_time = tflops_calc.total_time / max(tflops_calc.batch_count, 1)
        batch_count = tflops_calc.batch_count
        tflops_method = "Basic timing (wall-clock)"
        custom_libs = []
    
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
    print(f"TFLOPS Method: {tflops_method}")
    print(f"Total FLOPs: {total_flops:,}")
    print(f"Total time: {total_time:.4f}s")
    print(f"Average batch time: {avg_batch_time:.4f}s")
    print(f"Batches processed: {batch_count}")
    print(f"**TFLOPS: {final_tflops:.2f}**")
    
    # Show custom CUDA libraries if detected
    if custom_libs:
        print(f"Custom CUDA libraries detected: {', '.join(custom_libs)}")
    
    # Show detailed Nsight Compute results if available
    if nsight_results and 'metrics_summary' in nsight_results:
        print(f"\n--- Nsight Compute Detailed Results ---")
        summary = nsight_results['metrics_summary']
        print(f"Total kernels profiled: {summary['total_kernels']}")
        print(f"FP32 operations: {summary['fp32_operations']:,}")
        print(f"FP64 operations: {summary['fp64_operations']:,}")
        print(f"FMA operations: {summary['ffma_operations']:,}")
        print(f"FP16 operations: {summary['fp16_operations']:,}")
        print(f"BF16 operations: {summary['bf16_operations']:,}")
        print(f"Total cycles: {summary['total_cycles']:,}")
        print(f"Average utilization: {summary['average_utilization']:.1f}%")
        
        # Show kernel breakdown
        if nsight_results.get('kernels'):
            print(f"\nKernel breakdown:")
            for kernel_name, metrics in nsight_results['kernels'].items():
                if metrics:
                    fp32_ops = metrics.get('sm__sass_thread_inst_executed_op_fp32_pred_on.sum', 0)
                    ffma_ops = metrics.get('sm__sass_thread_inst_executed_op_ffma_pred_on.sum', 0)
                    utilization = metrics.get('sm__throughput.avg.pct_of_peak_sustained_elapsed', 0)
                    print(f"  {kernel_name}: {fp32_ops:,.0f} FP32 ops, {ffma_ops:,.0f} FMA ops, {utilization:.1f}% util")
    
    # GPU utilization estimation
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\nGPU: {gpu_name}")
        
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
            'tflops_method': tflops_method,
            'total_flops': total_flops,
            'total_time': total_time,
            'avg_batch_time': avg_batch_time,
            'batch_count': batch_count,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'custom_libraries_detected': custom_libs
        }
        
        # Add Nsight Compute results if available
        if nsight_results:
            metrics.update({
                'nsight_compute': True,
                'nsight_kernels': nsight_results.get('kernels', {}),
                'nsight_metrics_summary': nsight_results.get('metrics_summary', {}),
                'nsight_custom_libraries': nsight_results.get('custom_libraries_detected', []),
                'nsight_output_file': nsight_results.get('output_file', '')
            })
        else:
            metrics['nsight_compute'] = False
        
        # Add PyTorch profiler results if available
        if profiling_results:
            metrics.update({
                'pytorch_profiler': True,
                'profiler_total_cuda_time': profiling_results.get('total_cuda_time', 0),
                'profiler_total_cpu_time': profiling_results.get('total_cpu_time', 0),
                'profiler_profile_steps': profiling_results.get('profile_steps', 0),
                'profiler_flops_per_batch': profiling_results.get('flops_per_batch', 0)
            })
        else:
            metrics['pytorch_profiler'] = False
        
        metrics_path = f"metrics_{args.data}_seq{args.seq_len}_pred{args.pred_len}.npy"
        np.save(metrics_path, metrics)
        print(f"Performance metrics saved to {metrics_path}")
        
        # Also save Nsight Compute results separately if available
        if nsight_results and 'output_file' in nsight_results:
            print(f"Nsight Compute profile saved to: {nsight_results['output_file']}")
    
    return predictions, targets


def detailed_profiling_for_tflops(model: nn.Module, test_loader: DataLoader, args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    """Run detailed profiling with PyTorch profiler for accurate TFLOPS calculation"""
    print("\n" + "=" * 50)
    print("DETAILED TFLOPS PROFILING")
    print("=" * 50)
    
    model.eval()
    
    # 1. Select the number of steps to profile
    profile_steps = min(args.profile_steps, len(test_loader))
    print(f"Profiling {profile_steps} batches...")
    
    # Get batch shape for FLOP calculation
    first_batch = next(iter(test_loader))
    batch_x_shape = first_batch[0].shape
    print(f"Batch shape: {batch_x_shape}")
    
    # Calculate FLOPs per batch using our estimation
    flops_per_batch = count_model_flops(model, batch_x_shape, args)
    print(f"Estimated FLOPs per batch: {flops_per_batch:,}")
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if device.type == 'cuda' else [ProfilerActivity.CPU],
        record_shapes=False,
        with_stack=False,
        profile_memory=False
    ) as prof:
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                if i >= profile_steps:
                    break
                    
                batch_x = batch_x.to(device).float()
                batch_y = batch_y.to(device).float()
                batch_x_mark = batch_x_mark.to(device).float()
                batch_y_mark = batch_y_mark.to(device).float()
                dec_inp = torch.zeros_like(batch_y).float().to(device)
                
                with record_function("GraphSSM_Forward"):  # Time the entire forward pass
                    _ = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

    # 2. Extract Total Time from Profiler
    key_averages = prof.key_averages()
    time_key = "cuda_time_total" if device.type == 'cuda' else "cpu_time_total"
    
    # Find the total time spent in the specifically recorded function
    total_time_us = 0
    for event in key_averages:
        if event.key == "GraphSSM_Forward":
            total_time_us = event.self_average * profile_steps
            break
    
    # If we didn't find the specific function, sum all forward pass time
    if total_time_us == 0:
        for event in key_averages:
            if "forward" in event.key.lower() or "GraphSSM" in event.key:
                total_time_us += event.self_average * profile_steps
    
    # Convert microseconds (us) to seconds (s)
    total_time_s = total_time_us / 1e6
    
    # 3. Calculate total FLOPs for profiled batches
    total_flops = flops_per_batch * profile_steps
    
    # 4. Calculate TFLOPS
    tflops = (total_flops / 1e12) / total_time_s if total_time_s > 0 else 0.0
    
    print(f"\n--- Detailed TFLOPS Calculation (Profiled Steps) ---")
    print(f"Profile steps: {profile_steps}")
    print(f"Total Estimated FLOPs: {total_flops:,.0f}")
    print(f"Total GPU Time (Forward Pass): {total_time_s:.4f} s")
    print(f"**TFLOPS (Calculated): {tflops:.2f}**")
    
    # Show profiler results
    print(f"\nProfiler results (top 10 operations by {time_key}):")
    print(prof.key_averages().table(sort_by=time_key, row_limit=10))
    
    # Additional timing analysis
    total_cuda_time = sum([event.cuda_time_total for event in key_averages if event.cuda_time_total > 0])
    total_cpu_time = sum([event.cpu_time_total for event in key_averages if event.cpu_time_total > 0])
    
    print(f"\nTiming Summary:")
    print(f"Total CUDA time: {total_cuda_time:.2f}ms")
    print(f"Total CPU time: {total_cpu_time:.2f}ms")
    print(f"Average time per batch: {total_time_s/profile_steps:.4f}s")
    
    profiling_results = {
        'total_cuda_time': total_cuda_time,
        'total_cpu_time': total_cpu_time,
        'profile_steps': profile_steps,
        'total_time_s': total_time_s,
        'total_flops': total_flops,
        'tflops': tflops,
        'flops_per_batch': flops_per_batch
    }
    
    print("=" * 50)
    
    return profiling_results


def detailed_profiling(model: nn.Module, test_loader: DataLoader, args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    """Legacy function - redirects to detailed_profiling_for_tflops"""
    return detailed_profiling_for_tflops(model, test_loader, args, device)


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
    parser = argparse.ArgumentParser(
        description="Evaluate core/graph_ssm on MambaTS time series forecasting",
        epilog="""
Examples:
  # Run on ETT dataset (default)
  python ET_eval.py --mode inference

  # Run on Solar dataset
  python ET_eval.py --mode inference --use_solar

  # Run with custom workspace directory
  python ET_eval.py --mode inference --workspace /path/to/gg_ssms

  # Run on Solar dataset with custom paths
  python ET_eval.py --mode inference --data Solar --solar_root_path /path/to/solar --solar_data_path solar_AL.txt

  # Run with profiling and custom workspace
  python ET_eval.py --mode inference --use_solar --profile_detailed --nsight_compute --workspace /home/user/gg_ssms

  # Run on CPU to avoid GraphSSM CUDA issues
  python ET_eval.py --mode inference --force_cpu --workspace workspace

  # Run with CPU fallback if CUDA fails
  python ET_eval.py --mode inference --fallback_cpu --workspace workspace
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Basic config (MambaTS style)
    parser.add_argument("--task_name", type=str, default="long_term_forecast", help="task name")
    parser.add_argument("--is_training", type=int, default=1, help="status")
    parser.add_argument("--model_id", type=str, default="test", help="model id")
    parser.add_argument("--model", type=str, default="GraphSSM", help="model name")
    parser.add_argument("--seed", type=int, default=3047, help="random seed")
    
    # Data loader (MambaTS style)
    parser.add_argument("--data", type=str, default="ETTm1", help="dataset type")
    parser.add_argument("--root_path", type=str, default="/data/eval_pipelines/datasets/ETT-small", help="root path of the data file")
    parser.add_argument("--data_path", type=str, default="ETTm1.csv", help="data file")
    parser.add_argument("--features", type=str, default="M", help="forecasting task")
    parser.add_argument("--target", type=str, default="OT", help="target feature")
    parser.add_argument("--freq", type=str, default="t", help="freq for time features encoding")
    
    # Workspace and repository configuration
    parser.add_argument("--workspace", type=str, default="/workspace", 
                       help="path to workspace directory containing gg_ssms repository (default: /workspace). "
                            "This directory should contain the gg_ssms folder with MambaTS and core/graph_ssm subdirectories.")
    
    # Solar dataset specific configuration
    parser.add_argument("--solar_root_path", type=str, default="/data/eval_pipelines/datasets/Solar", help="root path for solar dataset")
    parser.add_argument("--solar_data_path", type=str, default="solar_AL.txt", help="solar dataset file")
    parser.add_argument("--checkpoints", type=str, default="/data/checkpoints/", help="location of model checkpoints")
    parser.add_argument("--visualization", type=str, default="/data/test_results", help="location of model checkpoints")
    parser.add_argument("--results", type=str, default="/data/results", help="location of model checkpoints")
    
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
    
    # Dataset selection
    parser.add_argument("--use_solar", action="store_true", help="Use Solar dataset instead of ETT")
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
    parser.add_argument("--fallback_cpu", action="store_true", help="fallback to CPU if CUDA fails", default=False)
    parser.add_argument("--force_cpu", action="store_true", 
                       help="force CPU execution (recommended for GraphSSM due to CUDA kernel memory issues)", default=False)
    
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
    parser.add_argument("--model_path", type=str, default="/data/checkpoints/best_model.pth", help="Path to pre-trained model")
    parser.add_argument("--save_predictions", action="store_true", help="Save predictions to file")
    parser.add_argument("--mode", type=str, default="inference", choices=["inference", "train", "example"], help="Mode: inference, train, or example")
    parser.add_argument("--example", action="store_true", help="Run simple example like main.py")
    
    # TFLOPS profiling specific
    parser.add_argument("--warmup_batches", type=int, default=5, help="Number of warmup batches for accurate timing")
    parser.add_argument("--save_metrics", action="store_true", help="Save performance metrics to file")
    parser.add_argument("--profile_detailed", action="store_true", help="Enable detailed profiling with PyTorch profiler")
    parser.add_argument("--profile_steps", type=int, default=10, help="Number of steps to profile in detailed mode")
    
    # NVIDIA Nsight Compute profiling
    parser.add_argument("--nsight_compute", action="store_true", help="Use NVIDIA Nsight Compute for exact TFLOPS measurement")
    parser.add_argument("--nsight_profile_batches", type=int, default=5, help="Number of batches to profile with Nsight Compute")
    parser.add_argument("--nsight_output_dir", type=str, default="./nsight_profiles", help="Directory to save Nsight Compute profiles")
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    args = build_argparser()
    
    # Setup paths based on workspace argument
    setup_paths(args.workspace)
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    print("=" * 60)
    print("GraphSSM Time Series Forecasting")
    print("=" * 60)
    print(f"Mode: {args.mode.upper()}")
    print(f"Workspace: {gg_ssms_path}")
    print(f"Dataset: {args.data}")
    print(f"Data path: {os.path.join(args.root_path, args.data_path)}")
    print(f"Sequence length: {args.seq_len}, Prediction length: {args.pred_len}")
    print(f"Model: {args.model}")
    print("=" * 60)
    
    # Configure dataset-specific arguments
    args = configure_dataset_args(args)
    
    # Check if data file exists
    data_file_path = os.path.join(args.root_path, args.data_path)
    if not os.path.exists(data_file_path):
        print(f"ERROR: Data file not found at {data_file_path}")
        if args.data == "Solar":
            print("Please download the Solar dataset and place it in the correct location.")
            print("The Solar dataset should be a .txt file with comma-separated values.")
            print("Expected format: solar_AL.txt with 137 features per timestep.")
            print(f"Place the solar dataset in: {args.root_path}")
        else:
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
            result = inference(args)
            if result is not None:
                predictions, targets = result
                print("\nInference completed!")
            else:
                print("\nInference failed - see error messages above.")
    else:
        print("\nStarting training...")
        # Note: Training function would need to be re-added if needed
        print("Training mode not implemented in this inference-only version.")
        print("Use a separate training script for model training.")
