CUDA__VERSION=$(nvcc --version|sed -n 4p|cut -f5 -d" "|cut -f1 -d",")
if [ "$CUDA__VERSION" != "$DESIRED_CUDA" ]; then
    echo "CUDA Version is not $DESIRED_CUDA. CUDA Version found: $CUDA__VERSION"
    exit 1
fi

mkdir build
cd build
cmake .. \
  -DBUILD_SHARED_LIBS=OFF \
  -DMAGMA_ENABLE_CUDA=ON \
  -DUSE_FORTRAN=OFF \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES"
cmake --build . --parallel "$(getconf _NPROCESSORS_CONF)"
cmake --install .
cd ..
