#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "torch/csrc/cuda/generic/TensorCopyAsync.cpp"
#else

#ifndef THC_REAL_IS_HALF
void THCPTensor_(copyAsyncCPU)(PyObject *dst, PyObject *src)
{
  THCTensor_(copyAsyncCPU)(state, ((THCPTensor*)dst)->cdata, ((THPTensor*)src)->cdata);
}

void THPTensor_(copyAsyncGPU)(PyObject *dst, PyObject *src)
{
  THTensor_(copyAsyncCuda)(state, ((THPTensor*)dst)->cdata, ((THCPTensor*)src)->cdata);
}
#endif

#endif
