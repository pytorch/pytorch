# WARGNING: this script assumes it's ran from repo's root

BASE_DIR=$(pwd)
cd torch/lib
INSTALL_DIR=$(pwd)/tmp_install
BASIC_FLAGS=" -DTH_INDEX_BASE=1 -I$INSTALL_DIR/include -I$INSTALL_DIR/include/TH -I$INSTALL_DIR/include/THC -L$INSTALL_DIR/lib "
FLAGS="$BASIC_FLAGS -Wl,-rpath,\$ORIGIN"
function build() {
  mkdir -p build/$1
  cd build/$1
  cmake ../../$1 -DCMAKE_MODULE_PATH="$BASE_DIR/cmake/FindCUDA" \
              -DTorch_FOUND="1" \
              -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
              -DCMAKE_C_FLAGS="$FLAGS" \
              -DCMAKE_CXX_FLAGS="$FLAGS" \
              -DCUDA_NVCC_FLAGS="$BASIC_FLAGS" \
              -DTH_INCLUDE_PATH="$INSTALL_DIR/include"
  make install -j$(getconf _NPROCESSORS_ONLN)
  cd ../..
}

mkdir -p tmp_install
build TH
build THNN

if [[ "$1" == "--with-cuda" ]]; then
    build THC
    build THCUNN
fi

cp $INSTALL_DIR/lib/* .
cp THNN/generic/THNN.h .
