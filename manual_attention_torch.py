import math
import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load
from torch.profiler import profile, record_function, ProfilerActivity

# Load the CUDA kernel as a python module
minimal_attn = load(name='minimal_attn', sources=['minimal_attn/main.cpp', 'minimal_attn/flash.cu'], extra_cuda_cflags=['-O2'])

batch_size = 16
n_head = 12
seq_len = 64
head_embd = 64

q = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
k = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
v = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()

def manual_attn(q, k, v):
    att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y

# Perform some warm-up runs to ensure CUDA initialization is complete
for _ in range(10):
    _ = manual_attn(q, k, v)
    _ = minimal_attn.forward(q, k, v)

torch.cuda.synchronize()

# Using the Legacy Profiler
print('=== profiling manual attention with legacy profiler ===')

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    with record_function("manual_attention_legacy"):
        for _ in range(100):  # Run multiple times for more stable measurements
            manual_result_legacy = manual_attn(q, k, v)
        torch.cuda.synchronize()  # Ensure all operations complete

avg_time_legacy = prof.self_cpu_time_total / 100
print(f"Average time per iteration: {avg_time_legacy/1000:.3f} ms")
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

torch.cuda.synchronize()

print('=== profiling manual attention with new profiler ===')
profiler_activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
with profile(activities=profiler_activities, record_shapes=True) as prof:
    with record_function("manual_attention_new"):
        for _ in range(100):  # Run multiple times for more stable measurements
            manual_result_new = manual_attn(q, k, v)
        torch.cuda.synchronize()  # Ensure all operations complete

profiler_events = prof.key_averages()
cpu_time_sum = sum(event.cpu_time for event in profiler_events)
avg_time_new = cpu_time_sum / 100
print(f"Average time per iteration: {avg_time_new/1000:.3f} ms")
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

# Force synchronization once more
torch.cuda.synchronize()

print('=== profiling minimal flash attention === ')
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    with record_function("flash_attention"):
        for _ in range(100):  # Run multiple times for more stable measurements
            minimal_result = minimal_attn.forward(q, k, v)
        torch.cuda.synchronize()  # Ensure all operations complete
avg_time_flash = prof.self_cpu_time_total / 100
print(f"Average time per iteration: {avg_time_flash/1000:.3f} ms")
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print('=== Time comparison ===')
print(f"Legacy profiler: {avg_time_legacy/1000:.3f} ms")
print(f"New profiler: {avg_time_new/1000:.3f} ms")
print(f"Flash attention: {avg_time_flash/1000:.3f} ms")

print('attn values sanity check:', torch.allclose(minimal_result, manual_result_new, rtol=0, atol=1e-02))