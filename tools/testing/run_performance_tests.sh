#!/bin/bash

# PyTorch Performance Testing Script
# Automatically sets up environment and runs GPU vs CPU performance tests

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script configuration
PYTORCH_VENV_PATH="/Users/kayhewett/Downloads/pytorch/.venv"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${BLUE}🚀 PyTorch Performance Test Runner${NC}"
echo -e "${BLUE}====================================${NC}"
echo ""

# Function to check if PyTorch environment exists
check_pytorch_env() {
    if [ -d "$PYTORCH_VENV_PATH" ]; then
        echo -e "${GREEN}✅ Found PyTorch virtual environment at: $PYTORCH_VENV_PATH${NC}"
        return 0
    else
        echo -e "${RED}❌ PyTorch virtual environment not found at: $PYTORCH_VENV_PATH${NC}"
        return 1
    fi
}

# Function to activate PyTorch environment
activate_pytorch_env() {
    echo -e "${YELLOW}🔧 Activating PyTorch environment...${NC}"
    source "$PYTORCH_VENV_PATH/bin/activate"
    
    # Verify PyTorch is available
    if python -c "import torch; print(f'PyTorch {torch.__version__} loaded successfully')" 2>/dev/null; then
        echo -e "${GREEN}✅ PyTorch environment activated successfully${NC}"
        python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
        return 0
    else
        echo -e "${RED}❌ Failed to import PyTorch${NC}"
        return 1
    fi
}

# Function to run performance tests
run_tests() {
    echo ""
    echo -e "${BLUE}🏃 Running Performance Tests...${NC}"
    echo -e "${BLUE}===============================${NC}"
    
    # Change to temp directory to avoid import conflicts
    cd /tmp
    
    if [ "$1" = "basic" ]; then
        echo -e "${YELLOW}📋 Running basic PyTorch functionality test...${NC}"
        python "$SCRIPT_DIR/test_pytorch.py"
    elif [ "$1" = "gpu" ]; then
        echo -e "${YELLOW}⚡ Running GPU vs CPU performance comparison...${NC}"
        python "$SCRIPT_DIR/cuda_test.py"
    else
        echo -e "${YELLOW}📋 Running basic PyTorch functionality test...${NC}"
        python "$SCRIPT_DIR/test_pytorch.py"
        echo ""
        echo -e "${YELLOW}⚡ Running GPU vs CPU performance comparison...${NC}"
        python "$SCRIPT_DIR/cuda_test.py"
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [basic|gpu|all]"
    echo ""
    echo "Options:"
    echo "  basic  - Run only basic PyTorch functionality tests"
    echo "  gpu    - Run only GPU vs CPU performance comparison"
    echo "  all    - Run all tests (default)"
    echo ""
    echo "Examples:"
    echo "  $0           # Run all tests"
    echo "  $0 basic     # Run only basic tests"
    echo "  $0 gpu       # Run only GPU performance tests"
}

# Main execution
main() {
    local test_type="${1:-all}"
    
    # Show usage if help requested
    if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
        show_usage
        exit 0
    fi
    
    # Validate test type
    if [ "$test_type" != "basic" ] && [ "$test_type" != "gpu" ] && [ "$test_type" != "all" ]; then
        echo -e "${RED}❌ Invalid test type: $test_type${NC}"
        show_usage
        exit 1
    fi
    
    # Check if PyTorch environment exists
    if ! check_pytorch_env; then
        echo ""
        echo -e "${YELLOW}💡 To set up PyTorch environment:${NC}"
        echo "1. Install PyTorch: pip install torch"
        echo "2. Or use conda: conda install pytorch"
        echo "3. Update PYTORCH_VENV_PATH in this script if using different location"
        exit 1
    fi
    
    # Activate environment and run tests
    if activate_pytorch_env; then
        run_tests "$test_type"
        echo ""
        echo -e "${GREEN}🎉 Performance testing completed successfully!${NC}"
    else
        echo -e "${RED}❌ Failed to activate PyTorch environment${NC}"
        exit 1
    fi
}

# Run main function with all arguments
main "$@"
