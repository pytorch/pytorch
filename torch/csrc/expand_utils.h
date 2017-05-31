#ifndef THP_EXPAND_UTILS_H
#define THP_EXPAND_UTILS_H

#include <sstream>
#include <Python.h>

template <typename ExpandType>
ExpandType *newForExpand(LIBRARY_STATE_TYPE_NOARGS);

template <typename TensorType>
int expand(LIBRARY_STATE_TYPE TensorType *r, TensorType *tensor, THLongStorage *sizes, int raiseErrors);

template <typename TensorType>
int expand2(LIBRARY_STATE_TYPE TensorType *r1, TensorType *r2,
            TensorType *e1, TensorType *e2, int raiseErrors);

template <typename TensorType>
int expand3(LIBRARY_STATE_TYPE TensorType *r1, TensorType *r2, TensorType *r3,
            TensorType *e1, TensorType *e2, TensorType *e3, int raiseErrors);

template <typename ExpandType, typename TensorType>
void check_fallback(ExpandType *to_expand, TensorType *tensor,
                    char *to_expand_name, char *tensor_name, bool fallback,
                    ptrdiff_t to_expand_nElem, ptrdiff_t tensor_nElem, int to_expand_err) {
  if (fallback && getBackCompatBroadcastWarn()) {
    bool same_shape = THSize_isSameSizeAs(tensor->size, tensor->nDimension,
        to_expand->size, to_expand->nDimension);
    if (!same_shape && to_expand_err == 0 && (tensor_nElem == to_expand_nElem) && fallback) {
      std::ostringstream warn;
      warn << tensor_name << " and " << to_expand_name << " do not have the same shape, but are "
           << "broadcastable, and have the same number of elements.  Changing behavior in a backwards incompatible "
           << "manner to broadcasting rather than viewing as 1-dimensional.";
      PyErr_WarnEx(PyExc_UserWarning, warn.str().c_str(), 1);
    }
  }
}

template <typename ExpandType, typename TensorType>
int expand_inplace(LIBRARY_STATE_TYPE ExpandType *r, ExpandType *to_expand, TensorType *tensor,
                   char *to_expand_name, char *tensor_name, bool fallback,
                   THLongStorage *tensor_size,  ptrdiff_t to_expand_nElem, ptrdiff_t tensor_nElem, bool raise) {
  int ret = 0;

  int to_expand_err = expand<ExpandType>(LIBRARY_STATE r, to_expand, tensor_size, raise);
  if (to_expand_err != 0 && !raise) {
    ret = to_expand_err;
    std::ostringstream warn;
    warn << to_expand_name << " is not broadcastable to " << tensor_name
         << ", but they have the same number of elements.  Falling back to deprecated pointwise behavior.";
    PyErr_WarnEx(PyExc_UserWarning, warn.str().c_str(), 1);
  }

  return ret;
}

template <typename ExpandType, typename TensorType>
int expand_inplace1(LIBRARY_STATE_TYPE ExpandType *r, ExpandType *to_expand, TensorType *tensor,
                   char *to_expand_name, char *tensor_name, bool fallback) {
  ptrdiff_t to_expand_nElem = THSize_nElement(to_expand->nDimension, to_expand->size);
  ptrdiff_t tensor_nElem = THSize_nElement(tensor->nDimension, tensor->size);
  bool to_expand_raise = !fallback || (to_expand_nElem != tensor_nElem);
  THLongStoragePtr tensor_size(THLongStorage_newWithSize(tensor->nDimension));
  THLongStorage_rawCopy(tensor_size.get(), tensor->size);

  int ret = expand_inplace(LIBRARY_STATE r, to_expand, tensor, to_expand_name, tensor_name, fallback,
                           tensor_size, to_expand_nElem, tensor_nElem, to_expand_raise);

  check_fallback<ExpandType, TensorType>(to_expand, tensor, to_expand_name, tensor_name, fallback,
                                         to_expand_nElem, tensor_nElem, ret);

  return ret;
}

template <typename TensorType>
int expand_inplace2(LIBRARY_STATE_TYPE TensorType *r1, TensorType *r2,
                    TensorType *to_expand1, TensorType *to_expand2, TensorType *tensor,
                    char *to_expand1_name, char *to_expand2_name, char *tensor_name, bool fallback) {
  ptrdiff_t tensor_nElem = THSize_nElement(tensor->nDimension, tensor->size);
  ptrdiff_t to_expand1_nElem = THSize_nElement(to_expand1->nDimension, to_expand1->size);
  ptrdiff_t to_expand2_nElem = THSize_nElement(to_expand2->nDimension, to_expand2->size);
  bool to_expand1_raise = !fallback || (tensor_nElem != to_expand1_nElem);
  bool to_expand2_raise = !fallback || (tensor_nElem != to_expand2_nElem);
  THLongStoragePtr tensor_size(THLongStorage_newWithSize(tensor->nDimension));
  THLongStorage_rawCopy(tensor_size.get(), tensor->size);

  int ret = expand_inplace(LIBRARY_STATE r1, to_expand1, tensor, to_expand1_name, tensor_name, fallback,
                           tensor_size, to_expand1_nElem, tensor_nElem, to_expand1_raise || to_expand2_raise);

  int ret2 = 0;
  if (ret == 0) {
    ret2 = expand_inplace(LIBRARY_STATE r2, to_expand2, tensor, to_expand2_name, tensor_name, fallback,
                         tensor_size, to_expand2_nElem, tensor_nElem, to_expand1_raise || to_expand2_raise);
  }

  check_fallback<TensorType, TensorType>(to_expand1, tensor, to_expand1_name, tensor_name, fallback,
                                         to_expand1_nElem, tensor_nElem, ret);
  check_fallback<TensorType, TensorType>(to_expand2, tensor, to_expand2_name, tensor_name, fallback,
                                         to_expand2_nElem, tensor_nElem, ret2);

  return ret == 0 && ret2 == 0 ? 0 : -1;
}

template <typename TensorType>
int expand_outplace2(LIBRARY_STATE_TYPE TensorType *r1, TensorType *r2,
                    TensorType *to_expand1, TensorType *to_expand2,
                    char *to_expand1_name, char *to_expand2_name, bool fallback) {
  ptrdiff_t to_expand1_nElem = THSize_nElement(to_expand1->nDimension, to_expand1->size);
  ptrdiff_t to_expand2_nElem = THSize_nElement(to_expand2->nDimension, to_expand2->size);
  bool raise = !fallback || (to_expand1_nElem != to_expand2_nElem);

  int ret = 0;
  int err = expand2<TensorType>(LIBRARY_STATE r1, r2, to_expand1, to_expand2, raise);
  if (err != 0 && !raise) {
    ret = err;
    std::ostringstream warn;
    warn << to_expand1_name << " and " << to_expand2_name << " not broadcastable, but have the same number of "
        "elements.  Falling back to deprecated pointwise behavior.";
    PyErr_WarnEx(PyExc_UserWarning, warn.str().c_str(), 1);
  }
  check_fallback<TensorType, TensorType>(to_expand1, to_expand2, to_expand1_name, to_expand2_name, fallback,
                                         to_expand1_nElem, to_expand2_nElem, ret);
  return ret;
}

template <typename TensorType>
int expand_outplace3(LIBRARY_STATE_TYPE TensorType *r1, TensorType *r2, TensorType *r3,
                    TensorType *to_expand1, TensorType *to_expand2, TensorType *to_expand3,
                    char *to_expand1_name, char *to_expand2_name, char *to_expand3_name, bool fallback) {
  ptrdiff_t to_expand1_nElem = THSize_nElement(to_expand1->nDimension, to_expand1->size);
  ptrdiff_t to_expand2_nElem = THSize_nElement(to_expand2->nDimension, to_expand2->size);
  ptrdiff_t to_expand3_nElem = THSize_nElement(to_expand3->nDimension, to_expand3->size);
  bool to_expand2_raise = !fallback || (to_expand1_nElem != to_expand2_nElem);
  bool to_expand3_raise = !fallback || (to_expand1_nElem != to_expand2_nElem);

  int ret = 0;
  int err = expand3<TensorType>(LIBRARY_STATE r1, r2, r3,
                                to_expand1, to_expand2, to_expand3,
                                to_expand2_raise || to_expand3_raise);

  if (err != 0 && !to_expand2_raise && !to_expand3_raise) {
    ret = err;
    std::ostringstream warn;
    warn << to_expand1_name << ", " << to_expand2_name << ", and " << to_expand3_name << " not broadcastable,"
         << " but have the same number of elements.  Falling back to deprecated pointwise behavior.";
    PyErr_WarnEx(PyExc_UserWarning, warn.str().c_str(), 1);
  }

  check_fallback<TensorType, TensorType>(to_expand1, to_expand2, to_expand1_name, to_expand2_name, fallback,
                                         to_expand1_nElem, to_expand2_nElem, ret);
  check_fallback<TensorType, TensorType>(to_expand1, to_expand3, to_expand1_name, to_expand3_name, fallback,
                                         to_expand1_nElem, to_expand3_nElem, ret);
  return ret;
}

#endif
