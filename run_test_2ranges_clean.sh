#!/bin/bash
# Clean run script for ir.Conditional 2-range test

echo "Cleaning old logs and cache..."
rm -f test_ir_conditional_2ranges.log
rm -rf torch/_inductor/kernel/__pycache__/custom_op*.pyc

echo "Running ir.Conditional 2-range test..."
echo ""

# Enable logging
export TORCH_LOGS="+inductor"
export TORCHINDUCTOR_VERBOSE_PROGRESS=1

# Run test using full python path
conda activate pytorch-3.12
python test_ir_conditional_2ranges.py 2>&1 | tee test_ir_conditional_2ranges.log

echo ""
echo "Test complete! Log saved to: test_ir_conditional_2ranges.log"
