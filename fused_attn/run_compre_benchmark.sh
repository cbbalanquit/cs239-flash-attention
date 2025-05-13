#!/bin/bash

# Name of your compiled CUDA program
PROGRAM="./param_fused_attention"

# Compile the program
compile() {
    echo "Compiling..."
    nvcc -O3 -arch=sm_75 param_fused_attention.cu -o param_fused_attention
    if [ $? -ne 0 ]; then
        echo "Compilation failed!"
        exit 1
    fi
}

# Run a regular benchmark
run_benchmark() {
    local batch_size=$1
    local num_heads=$2
    local seq_len=$3
    local head_dim=$4
    local iters=$5
    
    echo "===== Benchmark: B=$batch_size, H=$num_heads, L=$seq_len, D=$head_dim ====="
    $PROGRAM $batch_size $num_heads $seq_len $head_dim $iters
    echo ""
}

# Run with Nsight Systems profiling
run_nsys() {
    local batch_size=$1
    local num_heads=$2
    local seq_len=$3
    local head_dim=$4
    local iters=10  # Fewer iterations for profiling
    
    echo "===== Nsight Systems Profile: B=$batch_size, H=$num_heads, L=$seq_len, D=$head_dim ====="
    nsys profile -o profile_B${batch_size}_H${num_heads}_L${seq_len}_D${head_dim} \
        --stats=true \
        $PROGRAM $batch_size $num_heads $seq_len $head_dim $iters
    echo ""
}

# Run with Nsight Compute profiling
run_ncu() {
    local batch_size=$1
    local num_heads=$2
    local seq_len=$3
    local head_dim=$4
    local iters=1  # Just one iteration for Compute profiling
    
    echo "===== Nsight Compute Profile: B=$batch_size, H=$num_heads, L=$seq_len, D=$head_dim ====="
    ncu --export profile_ncu_B${batch_size}_H${num_heads}_L${seq_len}_D${head_dim} \
        --set full \
        $PROGRAM $batch_size $num_heads $seq_len $head_dim $iters
    echo ""
}

# Main script

# Compile first
compile

# Default values
BATCH_SIZES=(1 2 4 8 16 32)
NUM_HEADS=(8)
SEQ_LENGTHS=(128 256 512 1024 2048)
HEAD_DIMS=(64)
ITERATIONS=10

# Check if we should run with Nsight
USE_NSYS=0
USE_NCU=0

# Parse command line arguments
for arg in "$@"; do
    case $arg in
        --nsys)
            USE_NSYS=1
            shift
            ;;
        --ncu)
            USE_NCU=1
            shift
            ;;
        --batch=*)
            BATCH_SIZES=(${arg#*=})
            shift
            ;;
        --heads=*)
            NUM_HEADS=(${arg#*=})
            shift
            ;;
        --seq=*)
            SEQ_LENGTHS=(${arg#*=})
            shift
            ;;
        --dim=*)
            HEAD_DIMS=(${arg#*=})
            shift
            ;;
        --iter=*)
            ITERATIONS=${arg#*=}
            shift
            ;;
    esac
done

# Print benchmark configuration
echo "==== Benchmark Configuration ===="
echo "Batch sizes: ${BATCH_SIZES[*]}"
echo "Number of heads: ${NUM_HEADS[*]}"
echo "Sequence lengths: ${SEQ_LENGTHS[*]}"
echo "Head dimensions: ${HEAD_DIMS[*]}"
echo "Iterations: $ITERATIONS"
echo "Use Nsight Systems: $USE_NSYS"
echo "Use Nsight Compute: $USE_NCU"
echo "=============================="
echo ""

# Create results directory
mkdir -p results

# Run benchmarks
for B in "${BATCH_SIZES[@]}"; do
    for H in "${NUM_HEADS[@]}"; do
        for L in "${SEQ_LENGTHS[@]}"; do
            for D in "${HEAD_DIMS[@]}"; do
                # Skip combinations where L*L is too large (memory constraints)
                if [ $((B * H * L * L)) -gt 100000000 ]; then
                    echo "Skipping B=$B, H=$H, L=$L, D=$D due to memory constraints"
                    continue
                fi
                
                # Run regular benchmark
                run_benchmark $B $H $L $D $ITERATIONS | tee -a results/benchmark_results.txt
                
                # Run Nsight Systems if requested
                if [ $USE_NSYS -eq 1 ]; then
                    run_nsys $B $H $L $D | tee -a results/nsys_results.txt
                fi
                
                # Run Nsight Compute if requested
                if [ $USE_NCU -eq 1 ]; then
                    run_ncu $B $H $L $D | tee -a results/ncu_results.txt
                fi
            done
        done
    done
done

echo "All benchmarks complete!"
echo "Results saved to results/ directory"