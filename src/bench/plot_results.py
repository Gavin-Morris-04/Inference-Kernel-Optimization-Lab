"""
Utility script to plot benchmark results from CSV files.
Usage: python src/bench/plot_results.py
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path (two levels up from this file)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

results_dir = Path(__file__).parent.parent.parent / "results"
plots_dir = results_dir / "plots"


def plot_matmul_results():
    """Plot GEMM benchmark results."""
    csv_path = results_dir / "matmul_results.csv"
    if not csv_path.exists():
        print(f"Results file not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    # Filter to relevant columns
    if 'numpy' in df['kernel'].values and 'numba' in df['kernel'].values:
        # Plot latency comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        numpy_data = df[df['kernel'] == 'numpy']
        numba_data = df[df['kernel'] == 'numba']
        
        # Latency comparison
        axes[0].semilogy(numpy_data['M'], numpy_data['latency_ms'], 'o-', label='NumPy', marker='s')
        axes[0].semilogy(numba_data['M'], numba_data['latency_ms'], 'o-', label='Numba', marker='^')
        axes[0].set_xlabel('Matrix Dimension (M)')
        axes[0].set_ylabel('Latency (ms)')
        axes[0].set_title('GEMM Latency Comparison')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Throughput comparison
        axes[1].plot(numpy_data['M'], numpy_data['throughput_gflops'], 'o-', label='NumPy', marker='s')
        axes[1].plot(numba_data['M'], numba_data['throughput_gflops'], 'o-', label='Numba', marker='^')
        axes[1].set_xlabel('Matrix Dimension (M)')
        axes[1].set_ylabel('Throughput (GFLOPS)')
        axes[1].set_title('GEMM Throughput Comparison')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "matmul_results.png", dpi=150)
        print(f"Saved plot: {plots_dir / 'matmul_results.png'}")
        plt.close()


def plot_softmax_results():
    """Plot Softmax benchmark results."""
    csv_path = results_dir / "softmax_results.csv"
    if not csv_path.exists():
        print(f"Results file not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    if 'numpy' in df['kernel'].values and 'numba' in df['kernel'].values:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        numpy_data = df[df['kernel'] == 'numpy']
        numba_data = df[df['kernel'] == 'numba']
        
        # Create size identifier
        numpy_data['size'] = numpy_data['batch_size'] * numpy_data['seq_len']
        numba_data['size'] = numba_data['batch_size'] * numba_data['seq_len']
        
        # Latency comparison
        axes[0].semilogy(numpy_data['size'], numpy_data['latency_ms'], 'o-', label='NumPy', marker='s')
        axes[0].semilogy(numba_data['size'], numba_data['latency_ms'], 'o-', label='Numba', marker='^')
        axes[0].set_xlabel('Input Size (batch_size × seq_len)')
        axes[0].set_ylabel('Latency (ms)')
        axes[0].set_title('Softmax Latency Comparison')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Throughput comparison
        axes[1].plot(numpy_data['size'], numpy_data['throughput_mops'], 'o-', label='NumPy', marker='s')
        axes[1].plot(numba_data['size'], numba_data['throughput_mops'], 'o-', label='Numba', marker='^')
        axes[1].set_xlabel('Input Size (batch_size × seq_len)')
        axes[1].set_ylabel('Throughput (MOPS)')
        axes[1].set_title('Softmax Throughput Comparison')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "softmax_results.png", dpi=150)
        print(f"Saved plot: {plots_dir / 'softmax_results.png'}")
        plt.close()


def plot_layernorm_results():
    """Plot LayerNorm benchmark results."""
    csv_path = results_dir / "layernorm_results.csv"
    if not csv_path.exists():
        print(f"Results file not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    if 'numpy' in df['kernel'].values and 'numba' in df['kernel'].values:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        numpy_data = df[df['kernel'] == 'numpy']
        numba_data = df[df['kernel'] == 'numba']
        
        # Create size identifier
        numpy_data['size'] = numpy_data['batch_size'] * numpy_data['hidden_dim']
        numba_data['size'] = numba_data['batch_size'] * numba_data['hidden_dim']
        
        # Latency comparison
        axes[0].semilogy(numpy_data['size'], numpy_data['latency_ms'], 'o-', label='NumPy', marker='s')
        axes[0].semilogy(numba_data['size'], numba_data['latency_ms'], 'o-', label='Numba', marker='^')
        axes[0].set_xlabel('Input Size (batch_size × hidden_dim)')
        axes[0].set_ylabel('Latency (ms)')
        axes[0].set_title('LayerNorm Latency Comparison')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Throughput comparison
        axes[1].plot(numpy_data['size'], numpy_data['throughput_mops'], 'o-', label='NumPy', marker='s')
        axes[1].plot(numba_data['size'], numba_data['throughput_mops'], 'o-', label='Numba', marker='^')
        axes[1].set_xlabel('Input Size (batch_size × hidden_dim)')
        axes[1].set_ylabel('Throughput (MOPS)')
        axes[1].set_title('LayerNorm Throughput Comparison')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "layernorm_results.png", dpi=150)
        print(f"Saved plot: {plots_dir / 'layernorm_results.png'}")
        plt.close()


def plot_inference_block_results():
    """Plot end-to-end inference block results."""
    csv_path = results_dir / "inference_block_results.csv"
    if not csv_path.exists():
        print(f"Results file not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Total latency
    axes[0].semilogy(df['batch_size'], df['total_ms_mean'], 'o-', label='Total', marker='s')
    axes[0].semilogy(df['batch_size'], df['projection_ms_mean'], 'o-', label='Projection', marker='^')
    axes[0].semilogy(df['batch_size'], df['softmax_ms_mean'], 'o-', label='Softmax', marker='v')
    axes[0].semilogy(df['batch_size'], df['layernorm_ms_mean'], 'o-', label='LayerNorm', marker='d')
    axes[0].set_xlabel('Batch Size')
    axes[0].set_ylabel('Latency (ms)')
    axes[0].set_title('Inference Block Latency Breakdown')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Percentage breakdown
    df['proj_pct'] = 100 * df['projection_ms_mean'] / df['total_ms_mean']
    df['softmax_pct'] = 100 * df['softmax_ms_mean'] / df['total_ms_mean']
    df['layernorm_pct'] = 100 * df['layernorm_ms_mean'] / df['total_ms_mean']
    
    axes[1].plot(df['batch_size'], df['proj_pct'], 'o-', label='Projection', marker='^')
    axes[1].plot(df['batch_size'], df['softmax_pct'], 'o-', label='Softmax', marker='v')
    axes[1].plot(df['batch_size'], df['layernorm_pct'], 'o-', label='LayerNorm', marker='d')
    axes[1].set_xlabel('Batch Size')
    axes[1].set_ylabel('Percentage of Total Time (%)')
    axes[1].set_title('Inference Block Time Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "inference_block_results.png", dpi=150)
    print(f"Saved plot: {plots_dir / 'inference_block_results.png'}")
    plt.close()


if __name__ == "__main__":
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating plots from benchmark results...")
    plot_matmul_results()
    plot_softmax_results()
    plot_layernorm_results()
    plot_inference_block_results()
    print("Plot generation complete!")

