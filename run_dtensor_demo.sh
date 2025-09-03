#!/bin/bash
#
# PyTorch DTensor Demo Launch Commands
#
# This script shows different ways to launch the DTensor demonstration
# with different numbers of processes and configurations.
#

# Make the script executable
chmod +x dtensor_demo.py

echo "PyTorch DTensor Demo Launch Options"
echo "=================================="
echo ""

# Option 1: Single node, 2 processes (minimum for distributed)
echo "1. Launch with 2 processes (single node):"
echo "   torchrun --nproc_per_node=2 --nnodes=1 dtensor_demo.py"
echo ""

# Option 2: Single node, 4 processes (recommended for full demo)
echo "2. Launch with 4 processes (single node, enables 2D mesh):"
echo "   torchrun --nproc_per_node=4 --nnodes=1 dtensor_demo.py"
echo ""

# Option 3: Single node, 8 processes (for larger scale demo)
echo "3. Launch with 8 processes (single node, larger scale):"
echo "   torchrun --nproc_per_node=8 --nnodes=1 dtensor_demo.py"
echo ""

# Option 4: Multi-node setup example
echo "4. Multi-node setup (2 nodes, 4 processes each):"
echo "   # On node 0:"
echo "   torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \\"
echo "            --master_addr=<node0_ip> --master_port=29500 dtensor_demo.py"
echo ""
echo "   # On node 1:"
echo "   torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \\"
echo "            --master_addr=<node0_ip> --master_port=29500 dtensor_demo.py"
echo ""

# Option 5: CPU-only demo
echo "5. CPU-only demo (if CUDA not available):"
echo "   CUDA_VISIBLE_DEVICES= torchrun --nproc_per_node=2 --nnodes=1 dtensor_demo.py"
echo ""

# Option 6: With additional debugging
echo "6. With debugging output:"
echo "   TORCH_LOGS=+dtensor torchrun --nproc_per_node=4 --nnodes=1 dtensor_demo.py"
echo ""

echo "Prerequisites:"
echo "- PyTorch with distributed support"
echo "- CUDA GPUs for GPU demo (optional, falls back to CPU)"
echo "- Multiple processes/GPUs for full functionality"
echo ""

echo "Choose an option and run the corresponding command!"
echo ""

# Quick launch function
function quick_launch() {
    local nproc=${1:-4}
    echo "Launching DTensor demo with $nproc processes..."
    torchrun --nproc_per_node=$nproc --nnodes=1 dtensor_demo.py
}

# If script is sourced, provide the function
# If executed directly, show help
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "To quick launch with 4 processes, run:"
    echo "  source run_dtensor_demo.sh && quick_launch 4"
    echo ""
    echo "Or directly use one of the torchrun commands above."
fi
