@echo off
REM Script to run all benchmarks on Windows
REM Usage: scripts\run_all_benchmarks.bat

setlocal

echo ==========================================
echo Running All Benchmarks
echo ==========================================

REM Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Create results directory
if not exist results\plots mkdir results\plots

echo.
echo 1. Benchmarking GEMM...
python src\bench\bench_matmul.py

echo.
echo 2. Benchmarking Softmax...
python src\bench\bench_softmax.py

echo.
echo 3. Benchmarking LayerNorm...
python src\bench\bench_layernorm.py

echo.
echo 4. Benchmarking End-to-End Inference Block...
python src\bench\inference_block.py

echo.
echo ==========================================
echo All benchmarks complete!
echo Results saved in results\ directory
echo ==========================================

endlocal

