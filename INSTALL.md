# Instructions

You need to move two files into the appropriate location, after downloading cuSPARSELt 0.4.0

```
mv ${CUSPARESLT_DIR}/lib64/libcuspareLt.so.0.4.0.0 ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcusparseLt.so
mv ${CUSPARESLT_DIR}/include/libcuparse_static.a ${CUDA_TOOLKIT_ROOT_DIR}/include/libcupsarse_static.a
```

After this you should be able to build normally. You should be able to use the build flags ive included in env.sh (for faster compilation)

```
source .env.sh
python setup.py develop
```

You can test if this works using
```
python -u benchmark_cusparselt.py --mode nvidia-bert
```

