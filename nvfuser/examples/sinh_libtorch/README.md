# Build

```
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ..
make -j
```

# Test

```
./sinh_example
```
