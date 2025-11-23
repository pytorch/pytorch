#!/bin/bash
# Quick test script for collective autotuning

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch-3.12

cd /data/users/tianren/pytorch

# Test with default 2 ranks
echo "========================================="
echo "Testing with default 2 ranks"
echo "========================================="
python test/inductor/test_collective_autotuning.py TestCollectiveAutotuning.test_allreduce_2ranks -v

# Test with multiple ranks if you have enough GPUs
# Uncomment to test:
# echo ""
# echo "========================================="
# echo "Testing with 2,4 ranks"
# echo "========================================="
# TEST_COLLECTIVE_WORLD_SIZES="2,4" python test/inductor/test_collective_autotuning.py -v
