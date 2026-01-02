#!/bin/bash
# Script to run all benchmarks and generate results
# Usage: bash scripts/run_all_benchmarks.sh

set -e  # Exit on error

echo "=========================================="
echo "Running All Benchmarks"
echo "=========================================="

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate  # Linux/Mac
    # For Windows Git Bash: source venv/Scripts/activate
fi

# Create results directory
mkdir -p results/plots

echo ""
echo "1. Benchmarking GEMM..."
python src/bench/bench_matmul.py

echo ""
echo "2. Benchmarking Softmax..."
python src/bench/bench_softmax.py

echo ""
echo "3. Benchmarking LayerNorm..."
python src/bench/bench_layernorm.py

echo ""
echo "4. Benchmarking End-to-End Inference Block..."
python src/bench/inference_block.py

echo ""
echo "=========================================="
echo "All benchmarks complete!"
echo "Results saved in results/ directory"
echo "=========================================="

