function pip_install() {
  # retry 3 times
  # old versions of pip don't have the "--progress-bar" flag
  pip install --progress-bar off "$@" || pip install --progress-bar off "$@" || pip install --progress-bar off "$@" ||\
  pip install "$@" || pip install "$@" || pip install "$@"
}

# export PYTORCH_ROCM_ARCH=gfx1030

TORCHVISION_COMMIT=e828eefa4c326f893ebdd07abae7adc873d6ab63
pip_install --user "git+https://github.com/pytorch/vision.git@$TORCHVISION_COMMIT"

# 0.10.0a0+e828eef
