# WARGNING: this script assumes it's ran from repo's root
set -e

BASE_DIR=$(pwd)
cd torch/lib
INSTALL_DIR=$(pwd)/tmp_install
BASIC_C_FLAGS=" -DTH_INDEX_BASE=0 -I$INSTALL_DIR/include -I$INSTALL_DIR/include/TH -I$INSTALL_DIR/include/THC "
LDFLAGS="-L$INSTALL_DIR/lib "
if [[ $(uname) == 'Darwin' ]]; then
    LDFLAGS="$LDFLAGS -Wl,-rpath,@loader_path"
else
    LDFLAGS="$LDFLAGS -Wl,-rpath,\$ORIGIN"
fi
C_FLAGS="$BASIC_C_FLAGS $LDFLAGS"
function build() {
  mkdir -p build/$1
  cd build/$1
  cmake ../../$1 -DCMAKE_MODULE_PATH="$BASE_DIR/cmake/FindCUDA" \
              -DTorch_FOUND="1" \
              -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
              -DCMAKE_C_FLAGS="$C_FLAGS" \
              -DCMAKE_CXX_FLAGS="$C_FLAGS $CPP_FLAGS" \
              -DCUDA_NVCC_FLAGS="$BASIC_C_FLAGS" \
              -DTH_INCLUDE_PATH="$INSTALL_DIR/include" \
              -DTH_LIB_PATH="$INSTALL_DIR/lib"
  make install -j$(getconf _NPROCESSORS_ONLN)
  cd ../..

  if [[ $(uname) == 'Darwin' ]]; then
    cd tmp_install/lib
    for lib in *.dylib; do
      echo "Updating install_name for $lib"
      install_name_tool -id @rpath/$lib $lib
    done
    cd ../..
  fi
}

mkdir -p tmp_install
build TH
build THNN

if [[ "$1" == "--with-cuda" ]]; then
    build THC
    build THCUNN
fi

CPP_FLAGS=" -std=c++11 "
build libshm


cp $INSTALL_DIR/lib/* .
cp THNN/generic/THNN.h .
cp THCUNN/THCUNN.h .
cp -r tmp_install/include .
cp $INSTALL_DIR/bin/* .
