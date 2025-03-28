set -x

PYTORCH_HOME=$PWD
PYTORCH_BUILD=$PYTORCH_HOME/build
PYTORCH_REMOTING_DEST=$PYTORCH_BUILD/aten/src/ATen/mps/remoting
what=$1

python -m torchgen.gen \
  --source-path $PYTORCH_HOME/aten/src/ATen \
  --install_dir $PYTORCH_REMOTING_DEST/$what \
  --per-operator-headers \
  --mps \
  --generate remoting_$what \
  --output-dependencies $PYTORCH_BUILD/generated_sources.$what.cmake
