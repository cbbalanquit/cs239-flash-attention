#!/bin/bash

# Configuration
BINARY=./naive_attention
OUTPUT_DIR=profiling_reports
SEQLENS=(128 256 512 1024 2048)
BATCH=1
HEADS=4
HEADDIM=64

# Create output directory
mkdir -p $OUTPUT_DIR

# Loop through sequence lengths
for SEQLEN in "${SEQLENS[@]}"
do
    REPORT_NAME="naiveattn_b${BATCH}_h${HEADS}_d${HEADDIM}_s${SEQLEN}"
    echo "Profiling seq_len=$SEQLEN..."
    
    ncu --export "$OUTPUT_DIR/${REPORT_NAME}" \
        $BINARY $SEQLEN $BATCH $HEADS $HEADDIM 
done

echo "Profiling complete. Reports saved in $OUTPUT_DIR/"
