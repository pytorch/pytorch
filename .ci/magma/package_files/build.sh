CUDA__VERSION=$(nvcc --version|sed -n 4p|cut -f5 -d" "|cut -f1 -d",")
if [ "$CUDA__VERSION" != "$DESIRED_CUDA" ]; then
    echo "CUDA Version is not $DESIRED_CUDA. CUDA Version found: $CUDA__VERSION"
    exit 1
fi

mkdir build
cd build
cmake .. -DUSE_FORTRAN=OFF -DGPU_TARGET="All" -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" -DCUDA_ARCH_LIST="$CUDA_ARCH_LIST"
make -j$(getconf _NPROCESSORS_CONF)
make install
cd ..
