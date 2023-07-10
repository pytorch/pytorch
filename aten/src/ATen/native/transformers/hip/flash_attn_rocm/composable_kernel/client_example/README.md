##
Client application links to CK library, and therefore CK library needs to be installed before building client applications.


## Build
```bash
mkdir -p client_example/build
cd client_example/build
```

```bash
cmake                                                                 \
-D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc                             \
-D CMAKE_PREFIX_PATH="/opt/rocm;${PATH_TO_CK_INSTALL_DIRECTORY}"      \
..
```

### Build client example
```bash
 make -j 
```
