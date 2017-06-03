#ifndef THDP_COPY_UTILS_H
#define THDP_COPY_UTILS_H

extern THDTensorDescriptor* THDPModule_makeDescriptor(PyObject *obj);
template <typename TensorSrc>
void THDPInsertCopyFunctionFromWorker(
  THPCopyList& copyList,
  void (*copyFunc)(THDTensorDescriptor* x, TensorSrc *z))
{
  auto wrapper = [copyFunc](PyObject* dst_, PyObject* src_) {
    TensorSrc* src = THPTypeInfo<TensorSrc>::cdata(src_);

    PyThreadState *_save = NULL;
    try {
      Py_UNBLOCK_THREADS;
      copyFunc(LIBRARY_STATE THDPModule_makeDescriptor(dst_), src);
      Py_BLOCK_THREADS;
    } catch (...) {
      if (_save) {
        Py_BLOCK_THREADS;
      }
      throw;
    }
  };

  PyTypeObject* srcType = THPTypeInfo<TensorSrc>::pyType();
  copyList.push_back({ srcType, wrapper, false });
}

template <typename TensorDst>
void THDPInsertCopyFunctionFromMaster(
  THPCopyList& copyList,
  void (*copyFunc)(TensorDst *x, THDTensorDescriptor* z),
  PyTypeObject *srcType)
{
  auto wrapper = [copyFunc](PyObject* dst_, PyObject* src_) {
    TensorDst* dst = THPTypeInfo<TensorDst>::cdata(dst_);

    PyThreadState *_save = NULL;
    try {
      Py_UNBLOCK_THREADS;
      copyFunc(LIBRARY_STATE dst, THDPModule_makeDescriptor(src_));
      Py_BLOCK_THREADS;
    } catch (...) {
      if (_save) {
        Py_BLOCK_THREADS;
      }
      throw;
    }
  };

  copyList.push_back({ srcType, wrapper, false });
}

#endif
