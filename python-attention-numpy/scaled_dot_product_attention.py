import numpy as np

def scaled_dot_product_attention(Q, K, V):
    d_k = K.shape[-1]

    attn_scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)

    attn_weights = np.exp(attn_scores - np.max(attn_scores, axis=-1, keepdims=True))  # Stability trick

    attn_weights /= np.sum(attn_weights, axis=-1, keepdims=True)  # Normalize using softmax

    return np.matmul(attn_weights, V), attn_weights

Q, K, V = np.random.rand(1, 3, 4), np.random.rand(1, 3, 4), np.random.rand(1, 3, 5)
output, attn_weights = scaled_dot_product_attention(Q, K, V)

print("+===+Test Q, K, V matrices.")
print("Q\n",Q)
print("K\n",K)
print("V\n",V)
print("+===+")

print("Output:", output)
print("Attention Weights:", attn_weights)
print("+===+")
print("+===+")