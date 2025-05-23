import numpy as np
import matplotlib.pyplot as plt
import time
import os

# Function to compute scaled dot-product attention
def scaled_dot_product_attention(Q, K, V):
    d_k = K.shape[-1]
    attn_scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)  # Scaled dot product
    attn_weights = np.exp(attn_scores - np.max(attn_scores, axis=-1, keepdims=True))  # Stability trick
    attn_weights /= np.sum(attn_weights, axis=-1, keepdims=True)  # Softmax normalization
    return np.matmul(attn_weights, V), attn_weights

# Multi-head attention function
def multi_head_attention(Q, K, V, num_heads):
    d_k = Q.shape[-1]
    d_v = V.shape[-1]

    # Step 1: Split Q, K, V into multiple heads
    Q_split = np.split(Q, num_heads, axis=-1)
    K_split = np.split(K, num_heads, axis=-1)
    V_split = np.split(V, num_heads, axis=-1)

    # Step 2: Apply scaled dot-product attention for each head
    outputs = []
    for i in range(num_heads):
        output, _ = scaled_dot_product_attention(Q_split[i], K_split[i], V_split[i])
        outputs.append(output)

    # Step 3: Concatenate outputs from all heads
    concat_output = np.concatenate(outputs, axis=-1)

    # Step 4: Apply final linear transformation (dummy for simplicity, no learned weights)
    final_output = concat_output  # In practice, this would be a learned linear transformation.

    return final_output

# Function to visualize attention weights
def visualize_attention(weights, title='Attention Weights', save_path=None):
    """
    Visualize attention weights as a heatmap.
    Args:
        weights: Attention weights with shape [batch_size, seq_len_q, seq_len_k]
        title: Title for the plot
        save_path: If provided, save the plot to this path instead of displaying
    """
    # Take the first sample from batch for visualization
    weights = weights[0]
    
    plt.figure(figsize=(10, 8))
    plt.imshow(weights, cmap='viridis')
    plt.title(title)
    plt.xlabel('Key sequence')
    plt.ylabel('Query sequence')
    plt.colorbar()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Attention visualization saved to {save_path}")
        plt.close()
    else:
        # Try to show, but don't fail if running in a terminal without display
        try:
            plt.show()
        except Exception as e:
            print(f"Could not display plot due to: {e}")
            print("Consider providing a save_path to save the visualization to a file.")

# New function to benchmark attention mechanisms with different sequence lengths
def benchmark_attention_mechanisms(seq_lengths=[512, 1024, 2048, 4096, 8192],
                                  n_runs=5, 
                                  batch_size=2, 
                                  d_model=64, 
                                  num_heads=8):
    """
    Benchmark the performance of attention mechanisms with varying sequence lengths.
    
    Args:
        seq_lengths: List of sequence lengths to test
        n_runs: Number of runs to average for each test
        batch_size: Batch size for the input tensors
        d_model: Dimensionality of the model (must be divisible by num_heads)
        num_heads: Number of attention heads for multi-head attention
        
    Returns:
        Dictionary containing timing results
    """
    print("\nBenchmarking attention mechanisms...")
    print("====================================")
    
    # Ensure d_model is divisible by num_heads
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
    
    results = {
        'seq_lengths': seq_lengths,
        'single_head_times': [],
        'multi_head_times': []
    }
    
    for seq_len in seq_lengths:
        print(f"\nTesting sequence length: {seq_len}")
        
        # Single-head attention timing
        single_head_times = []
        for run in range(n_runs):
            # Create random inputs
            Q = np.random.randn(batch_size, seq_len, d_model)
            K = np.random.randn(batch_size, seq_len, d_model)
            V = np.random.randn(batch_size, seq_len, d_model)
            
            # Time single-head attention
            start_time = time.time()
            _, _ = scaled_dot_product_attention(Q, K, V)
            end_time = time.time()
            
            single_head_times.append(end_time - start_time)
            
        avg_single_time = sum(single_head_times) / len(single_head_times)
        results['single_head_times'].append(avg_single_time)
        print(f"  Single-head attention: {avg_single_time:.4f} seconds (avg of {n_runs} runs)")
        
        # Multi-head attention timing
        multi_head_times = []
        for run in range(n_runs):
            # Create random inputs
            Q = np.random.randn(batch_size, seq_len, d_model)
            K = np.random.randn(batch_size, seq_len, d_model)
            V = np.random.randn(batch_size, seq_len, d_model)
            
            # Time multi-head attention
            start_time = time.time()
            _ = multi_head_attention(Q, K, V, num_heads)
            end_time = time.time()
            
            multi_head_times.append(end_time - start_time)
            
        avg_multi_time = sum(multi_head_times) / len(multi_head_times)
        results['multi_head_times'].append(avg_multi_time)
        print(f"  Multi-head attention: {avg_multi_time:.4f} seconds (avg of {n_runs} runs)")
    
    return results

# Function to visualize benchmark results
def visualize_benchmark_results(results, save_path=None):
    """
    Visualize the benchmark results.
    
    Args:
        results: Dictionary containing benchmark results
        save_path: If provided, save the plot to this path
    """
    plt.figure(figsize=(12, 8))
    
    # Create width-2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot execution times
    ax1.plot(results['seq_lengths'], results['single_head_times'], 'o-', label='Single-head Attention')
    ax1.plot(results['seq_lengths'], results['multi_head_times'], 's-', label='Multi-head Attention')
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_title('Attention Mechanism Execution Time')
    ax1.legend()
    ax1.grid(True)
    
    # Plot execution times with log scale for better visualization of growth
    ax2.plot(results['seq_lengths'], results['single_head_times'], 'o-', label='Single-head Attention')
    ax2.plot(results['seq_lengths'], results['multi_head_times'], 's-', label='Multi-head Attention')
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Execution Time (seconds)')
    ax2.set_title('Attention Mechanism Execution Time (Log Scale)')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Benchmark results visualization saved to {save_path}")
        plt.close()
    else:
        plt.show()

    # Create a separate plot to show the ratio
    plt.figure(figsize=(10, 6))
    ratios = [multi/single for multi, single in zip(results['multi_head_times'], results['single_head_times'])]
    plt.plot(results['seq_lengths'], ratios, 'o-')
    plt.xlabel('Sequence Length')
    plt.ylabel('Ratio (Multi-head / Single-head)')
    plt.title('Multi-head vs Single-head Attention Time Ratio')
    plt.grid(True)
    
    if save_path:
        ratio_path = save_path.replace('.png', '_ratio.png')
        plt.savefig(ratio_path)
        print(f"Ratio visualization saved to {ratio_path}")
        plt.close()
    else:
        plt.show()

# Function for unit testing with known inputs and expected outputs
def test_attention_mechanisms():
    """
    Test both the scaled dot-product attention and multi-head attention.
    """
    print("Running tests for attention mechanisms...")
    
    # Test 1: Simple test for scaled dot-product attention
    batch_size = 1
    seq_len = 128
    d_model = 64
    
    # Create simple test inputs where Q[0][0] should attend strongly to K[0][0]
    Q = np.zeros((batch_size, seq_len, d_model))
    K = np.zeros((batch_size, seq_len, d_model))
    V = np.zeros((batch_size, seq_len, d_model))
    
    # Set the first query and key to have high dot product
    Q[0, 0] = np.ones(d_model)
    K[0, 0] = np.ones(d_model)
    
    # Set values to be distinct
    for i in range(seq_len):
        V[0, i] = np.full(d_model, i)
    
    # Test scaled dot-product attention
    output, weights = scaled_dot_product_attention(Q, K, V)
    
    print("\nTest 1: Scaled Dot-Product Attention")
    print(f"Attention weights for first query: {weights[0, 0]}")
    print(f"Output for first query: {output[0, 0]}")
    
    # Visualize the attention weights
    visualize_attention(weights, 'Scaled Dot-Product Attention Weights', save_path='attention_single_test1.png')
    
    # Test 2: Multi-head attention with 2 heads
    num_heads = 2
    d_model = 8  # Must be divisible by num_heads
    
    # Create new test data
    Q = np.random.randn(batch_size, seq_len, d_model)
    K = np.random.randn(batch_size, seq_len, d_model)
    V = np.random.randn(batch_size, seq_len, d_model)
    
    # Test multi-head attention
    mha_output = multi_head_attention(Q, K, V, num_heads)
    
    print("\nTest 2: Multi-Head Attention")
    print(f"Input shape: {Q.shape}")
    print(f"Output shape: {mha_output.shape}")
    print(f"First output vector: {mha_output[0, 0]}")
    
    # Test 3: Compare single-head and multi-head attention
    print("\nTest 3: Comparing single-head vs. multi-head attention")
    
    # Generate more realistic sentence-like data
    batch_size = 1
    seq_len = 6
    d_model = 8
    
    # Create data with a clear pattern
    Q = np.zeros((batch_size, seq_len, d_model))
    K = np.zeros((batch_size, seq_len, d_model))
    V = np.zeros((batch_size, seq_len, d_model))
    
    # Make each position focus on different aspects
    for i in range(seq_len):
        Q[0, i] = np.random.randn(d_model)
        K[0, i] = np.random.randn(d_model)
        V[0, i] = np.random.randn(d_model)
    
    # Run with single head
    single_output, single_weights = scaled_dot_product_attention(Q, K, V)
    
    # Run with multiple heads
    multi_output = multi_head_attention(Q, K, V, 2)
    
    print(f"Single-head output shape: {single_output.shape}")
    print(f"Multi-head output shape: {multi_output.shape}")
    
    # Visualize single-head attention
    visualize_attention(single_weights, 'Single-Head Attention Weights', save_path='attention_single_test3.png')
    
    # Print summary of differences
    print("\nDifference between single-head and multi-head:")
    print(f"Mean absolute difference: {np.mean(np.abs(single_output - multi_output))}")
    
    return True

# Additional function to run a practical example with sentence-like data
def run_practical_example():
    """
    Demonstrates how attention works on sentence-like data
    """
    print("\nRunning practical example with sentence-like data...")
    
    # Simulate word embeddings for a sentence
    # Sentence: "The cat sat on the mat"
    batch_size = 1
    seq_len = 6  # 6 words
    d_model = 8  # 8-dimensional embeddings
    
    # Create synthetic word embeddings
    # In practice, these would come from an embedding layer
    embeddings = np.random.randn(batch_size, seq_len, d_model)
    
    # In self-attention, Q, K, and V are all derived from the same embeddings
    Q = embeddings
    K = embeddings
    V = embeddings
    
    # Run single-head attention
    output_single, weights_single = scaled_dot_product_attention(Q, K, V)
    
    # Run multi-head attention with 2 heads
    output_multi = multi_head_attention(Q, K, V, 2)
    
    # Visualize the attention pattern
    visualize_attention(weights_single, 'Self-Attention Weights for Sentence', save_path='attention_sentence_example.png')
    
    # Print attention pattern in text form
    print("\nAttention pattern (which words attend to which):")
    words = ["The", "cat", "sat", "on", "the", "mat"]
    for i in range(seq_len):
        attending_to = []
        for j in range(seq_len):
            if weights_single[0, i, j] > 0.2:  # Threshold for significant attention
                attending_to.append(f"{words[j]}({weights_single[0, i, j]:.2f})")
        print(f"{words[i]} attends to: {', '.join(attending_to)}")
    
    return True

# Function to save benchmark results as CSV
def save_benchmark_results_csv(results, output_path):
    """
    Save benchmark results to a CSV file.
    
    Args:
        results: Dictionary containing benchmark results
        output_path: Path to save the CSV file
    """
    import csv
    
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['sequence_length', 'single_head_time', 'multi_head_time', 'ratio']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for i, seq_len in enumerate(results['seq_lengths']):
            single_time = results['single_head_times'][i]
            multi_time = results['multi_head_times'][i]
            ratio = multi_time / single_time
            
            writer.writerow({
                'sequence_length': seq_len,
                'single_head_time': single_time,
                'multi_head_time': multi_time,
                'ratio': ratio
            })
    
    print(f"Benchmark results saved to CSV: {output_path}")

# Main function to run all tests and examples
def main():
    print("Testing and Benchmarking Multihead Attention Implementation")
    print("=========================================================")
    
    # Set up output directory for visualizations
    os.makedirs('attention_plots', exist_ok=True)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Test and benchmark multihead attention implementation')
    parser.add_argument('--no-vis', action='store_true', help='Disable visualizations completely')
    parser.add_argument('--output-dir', type=str, default='attention_plots', help='Directory to save visualizations')
    parser.add_argument('--skip-tests', action='store_true', help='Skip the basic tests and go straight to benchmarking')
    parser.add_argument('--seq-lengths', type=int, nargs='+', default=[512, 1024, 2048, 4096, 8192], 
                        help='Sequence lengths to benchmark')
    parser.add_argument('--n-runs', type=int, default=10, help='Number of runs to average for each benchmark')
    parser.add_argument('--d-model', type=int, default=64, help='Dimensionality of model')
    parser.add_argument('--num-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--csv-output', type=str, default='attention_benchmark_results.csv',
                        help='Filename for CSV output of benchmark results')
    args = parser.parse_args()
    
    # Update visualization paths based on arguments
    global visualize_attention
    original_visualize_attention = visualize_attention
    
    if args.no_vis:
        # Replace with dummy function
        def dummy_visualize(*args, **kwargs):
            print("Visualization disabled.")
        visualize_attention = dummy_visualize
    else:
        # Wrap to use the output directory
        def wrapped_visualize(weights, title, save_path=None):
            if save_path:
                save_path = os.path.join(args.output_dir, save_path)
            return original_visualize_attention(weights, title, save_path)
        visualize_attention = wrapped_visualize
    
    # Run the basic tests if not skipped
    if not args.skip_tests:
        test_attention_mechanisms()
        run_practical_example()
    
    # Run the benchmarks
    benchmark_results = benchmark_attention_mechanisms(
        seq_lengths=args.seq_lengths,
        n_runs=args.n_runs,
        d_model=args.d_model,
        num_heads=args.num_heads
    )
    
    # Visualize the benchmark results
    if not args.no_vis:
        visualize_benchmark_results(benchmark_results, 
                                   save_path=os.path.join(args.output_dir, 'attention_benchmark_results.png'))
    
    # Save benchmark results to CSV
    csv_path = os.path.join(args.output_dir, args.csv_output)
    save_benchmark_results_csv(benchmark_results, csv_path)
    
    # Print a summary of results
    print("\nBenchmark Results Summary:")
    print("=========================")
    print(f"{'Sequence Length':<15} {'Single-head (s)':<15} {'Multi-head (s)':<15} {'Ratio':<10}")
    print("-" * 55)
    for i, seq_len in enumerate(benchmark_results['seq_lengths']):
        single_time = benchmark_results['single_head_times'][i]
        multi_time = benchmark_results['multi_head_times'][i]
        ratio = multi_time / single_time
        print(f"{seq_len:<15} {single_time:<15.4f} {multi_time:<15.4f} {ratio:<10.2f}")
    
    print("\nAll tests and benchmarks completed!")
    if not args.no_vis:
        print(f"Visualizations saved to directory: {args.output_dir}")
    print(f"CSV results saved to: {csv_path}")

if __name__ == "__main__":
    main()