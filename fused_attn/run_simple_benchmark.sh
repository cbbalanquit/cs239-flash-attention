#!/bin/bash

# Name of your compiled CUDA program
PROGRAM="./param_fused_attention"

# Compile the program
echo "Compiling..."
nvcc -O3 -arch=sm_75 param_fused_attention.cu -o param_fused_attention
if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

# Create CSV header
echo "batch_size,num_heads,seq_len,head_dim,avg_ms,gflops,bandwidth_gb" > benchmark_results.csv

BATCH_SIZES=(1 2 4 8 16 32)
NUM_HEADS=(8)
SEQ_LENGTHS=(128 256 512 1024 2048)
HEAD_DIMS=(64)
ITERATIONS=10

# Run benchmarks
for batch_size in "${BATCH_SIZES[@]}"; do
    for num_heads in "${NUM_HEADS[@]}"; do
        for seq_len in "${SEQ_LENGTHS[@]}"; do
            for head_dim in "${HEAD_DIMS[@]}"; do
                # Skip combinations where L*L is too large (memory constraints)
                if [ $((batch_size * num_heads * seq_len * seq_len)) -gt 100000000 ]; then
                    echo "Skipping B=$batch_size, H=$num_heads, L=$seq_len, D=$head_dim due to memory constraints"
                    continue
                fi
                
                echo "Running B=$batch_size, H=$num_heads, L=$seq_len, D=$head_dim..."
                output=$($PROGRAM $batch_size $num_heads $seq_len $head_dim $ITERATIONS)
                
                # Extract results using grep and awk
                avg_ms=$(echo "$output" | grep "Average time" | awk '{print $5}')
                gflops=$(echo "$output" | grep "FLOPS" | awk '{print $2}')
                bandwidth=$(echo "$output" | grep "Memory Bandwidth" | awk '{print $3}')
                
                # Append to CSV
                echo "$batch_size,$num_heads,$seq_len,$head_dim,$avg_ms,$gflops,$bandwidth" >> benchmark_results.csv
            done
        done
    done
done

echo "Benchmarking complete! Results saved to benchmark_results.csv"

# Run with Nsight Systems if requested
if [ "$1" == "--nsys" ]; then
    echo "Running Nsight Systems profile for selected configurations..."
    # Select a few interesting configs based on the benchmark results
    for config in "4 8 256 64" "8 8 512 32" "16 8 256 16"; do
        read -r b h l d <<< "$config"
        echo "Profiling B=$b, H=$h, L=$l, D=$d with Nsight Systems..."
        nsys profile -o profile_B${b}_H${h}_L${l}_D${d} --stats=true $PROGRAM $b $h $l $d 10
    done
fi

# Run with Nsight Compute if requested
if [ "$1" == "--ncu" ]; then
    echo "Running Nsight Compute profile for selected configurations..."
    # Select a few interesting configs based on the benchmark results
    for config in "4 8 256 64" "8 8 512 32" "16 8 256 16"; do
        read -r b h l d <<< "$config"
        echo "Profiling B=$b, H=$h, L=$l, D=$d with Nsight Compute..."
        ncu --export profile_ncu_B${b}_H${h}_L${l}_D${d} --set full $PROGRAM $b $h $l $d 1
    done
fi