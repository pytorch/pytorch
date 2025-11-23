#!/bin/bash
# Collective Op Autotuning - Package Script
# 用于打包所有文档和代码到一个tarball，方便迁移到其他机器

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTORCH_ROOT="$(dirname "$SCRIPT_DIR")"
PACKAGE_NAME="collective_op_autotuning_package_$(date +%Y%m%d_%H%M%S).tar.gz"
TEMP_DIR="/tmp/collective_op_package_$$"

echo "=== Collective Op Autotuning Packaging Script ==="
echo ""
echo "Creating package: $PACKAGE_NAME"
echo "PyTorch root: $PYTORCH_ROOT"
echo ""

# Create temporary directory
mkdir -p "$TEMP_DIR/collective_op_autotuning"

# Copy documentation
echo "📄 Copying documentation..."
cp -r "$SCRIPT_DIR"/* "$TEMP_DIR/collective_op_autotuning/docs/"

# Copy implementation code
echo "💻 Copying implementation code..."
mkdir -p "$TEMP_DIR/collective_op_autotuning/code"
cp "$PYTORCH_ROOT/torch/_inductor/runtime/collective_benchmarking.py" \
   "$TEMP_DIR/collective_op_autotuning/code/"

# Create README for the package
cat > "$TEMP_DIR/collective_op_autotuning/README_PACKAGE.md" << 'EOF'
# Collective Op Autotuning Package

This package contains all documentation and code for implementing Collective Operation Autotuning in PyTorch Inductor.

## 📦 Package Contents

```
collective_op_autotuning/
├── README_PACKAGE.md (this file)
├── docs/
│   ├── README.md - Documentation index
│   ├── MASTER_GUIDE.md - Main implementation guide ⭐
│   └── reference/ - Reference documents
│       ├── DESIGN_OVERVIEW.md
│       ├── V1_SIMPLE_APPROACH.md
│       ├── V2_ADVANCED_APPROACH.md
│       ├── FAQ.md
│       └── COLLECTIVE_OP_IMPLEMENTATION_SUMMARY.md
└── code/
    └── collective_benchmarking.py - Core implementation ✅

```

## 🚀 Quick Start

### Step 1: Extract Package
```bash
tar -xzf collective_op_autotuning_package_*.tar.gz
cd collective_op_autotuning
```

### Step 2: Read Documentation
```bash
# Start with the master guide
cat docs/MASTER_GUIDE.md | less

# Or open in your editor
vim docs/MASTER_GUIDE.md
```

### Step 3: Install Implementation
```bash
# Copy the implementation to your PyTorch checkout
cp code/collective_benchmarking.py <YOUR_PYTORCH>/torch/_inductor/runtime/

# Verify
ls <YOUR_PYTORCH>/torch/_inductor/runtime/collective_benchmarking.py
```

### Step 4: Follow Implementation Steps
Follow the steps in `docs/MASTER_GUIDE.md`:
1. Modify custom_op.py (Step 2)
2. Modify select_algorithm.py (Step 3-4)
3. Create tests (Phase 1-4)

## 📖 Recommended Reading Order

1. **Start here**: `docs/MASTER_GUIDE.md`
2. **If confused**: `docs/reference/FAQ.md`
3. **For details**: Other reference docs

## 🔧 Implementation Checklist

- [ ] Read MASTER_GUIDE.md
- [ ] Copy collective_benchmarking.py to PyTorch
- [ ] Modify custom_op.py (detection logic)
- [ ] Modify select_algorithm.py (integration)
- [ ] Write Phase 1 test
- [ ] Run Phase 1 test (2 ranks)
- [ ] Write Phase 2-4 tests
- [ ] Collect performance data
- [ ] Decide on V2 upgrade

## 💡 Key Points

- **V1 Goal**: Get collective op autotuning working (1-2 days)
- **V1 Features**: Timeout protection, cross-rank sync, compatible with existing system
- **V2 Optional**: If you have 3+ collective ops or need epilogue fusion
- **Testing**: Start with 2 ranks, gradually increase

## 🆘 Troubleshooting

### Issue: Import error
```python
ModuleNotFoundError: No module named 'torch._inductor.runtime.collective_benchmarking'
```
**Solution**: Make sure you copied `collective_benchmarking.py` to the right location:
```bash
cp code/collective_benchmarking.py <PYTORCH>/torch/_inductor/runtime/
```

### Issue: Distributed not initialized
```
RuntimeError: torch.distributed is not initialized
```
**Solution**: Initialize distributed before compiling:
```python
import torch.distributed as dist
dist.init_process_group(backend='nccl')
```

### Issue: Tests hang
**Solution**: V1 has timeout protection. Check logs for timeout warnings.

## 📞 Support

For questions or issues:
1. Check `docs/reference/FAQ.md`
2. Review troubleshooting section in MASTER_GUIDE.md
3. Contact PyTorch Inductor team

## 📝 Version

- Package Date: $(date)
- Version: 1.0 (V1 Implementation)
- Status: Ready for Implementation

---

**Ready to start? Open `docs/MASTER_GUIDE.md`!** 🚀
EOF

# Create installation script
cat > "$TEMP_DIR/collective_op_autotuning/install.sh" << 'EOF'
#!/bin/bash
# Installation script for collective op autotuning

if [ $# -eq 0 ]; then
    echo "Usage: ./install.sh <path_to_pytorch_checkout>"
    echo "Example: ./install.sh /home/user/pytorch"
    exit 1
fi

PYTORCH_DIR="$1"
TARGET_DIR="$PYTORCH_DIR/torch/_inductor/runtime"

if [ ! -d "$PYTORCH_DIR" ]; then
    echo "Error: PyTorch directory not found: $PYTORCH_DIR"
    exit 1
fi

if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Target directory not found: $TARGET_DIR"
    echo "Are you sure $PYTORCH_DIR is a PyTorch checkout?"
    exit 1
fi

echo "Installing collective_benchmarking.py to $TARGET_DIR ..."
cp code/collective_benchmarking.py "$TARGET_DIR/"

if [ $? -eq 0 ]; then
    echo "✅ Installation successful!"
    echo ""
    echo "Next steps:"
    echo "1. Read docs/MASTER_GUIDE.md"
    echo "2. Modify custom_op.py and select_algorithm.py"
    echo "3. Write and run tests"
else
    echo "❌ Installation failed!"
    exit 1
fi
EOF

chmod +x "$TEMP_DIR/collective_op_autotuning/install.sh"

# Create tarball
echo "📦 Creating tarball..."
cd "$TEMP_DIR"
tar -czf "$PACKAGE_NAME" collective_op_autotuning/

# Move to original directory
mv "$PACKAGE_NAME" "$PYTORCH_ROOT/"

# Cleanup
rm -rf "$TEMP_DIR"

echo ""
echo "✅ Package created successfully!"
echo ""
echo "Package location: $PYTORCH_ROOT/$PACKAGE_NAME"
echo "Package size: $(du -h "$PYTORCH_ROOT/$PACKAGE_NAME" | cut -f1)"
echo ""
echo "To use on another machine:"
echo "  1. scp $PACKAGE_NAME remote_machine:/path/to/destination/"
echo "  2. tar -xzf $PACKAGE_NAME"
echo "  3. cd collective_op_autotuning"
echo "  4. Read docs/MASTER_GUIDE.md"
echo ""
echo "Or use the install script:"
echo "  ./install.sh <path_to_pytorch>"
echo ""
