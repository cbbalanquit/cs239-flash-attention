# Fused Attention

Code to profile using nsys
```bash
nsys profile \
-o profile_fused_attn \
--trace=nvtx,cuda,osrt \
--gpu-metrics-devices=all \
--cuda-memory-usage true \
--force-overwrite true \
./fused_attention
```