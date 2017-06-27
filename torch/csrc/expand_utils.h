#ifndef THP_EXPAND_UTILS_H
#define THP_EXPAND_UTILS_H

#include <sstream>
#include <Python.h>

template <typename ExpandType>
ExpandType *newForExpand(LIBRARY_STATE_TYPE_NOARGS);

template <typename TensorType>
void expand(LIBRARY_STATE_TYPE TensorType *r, TensorType *tensor, THLongStorage *sizes);

template <typename TensorType1, typename TensorType2>
void expand2(LIBRARY_STATE_TYPE TensorType1 *r1, TensorType2 *r2,
             TensorType1 *e1, TensorType2 *e2,
             char *e1_name, char *e2_name) {
  if (e1->nDimension <= 0) {
    throw std::runtime_error(std::string("can't expand empty tensor ").append(e1_name));
  }
  if (e2->nDimension <= 0) {
    throw std::runtime_error(std::string("can't expand empty tensor ").append(e2_name));
  }
  THLongStoragePtr sizes(THLongStorage_new());
  char error_buffer[1024];
  int ret = THLongStorage_inferSize2(sizes,
                                     e1->size, e1->nDimension,
                                     e2->size, e2->nDimension,
                                     error_buffer, 1024);
  if (ret != 0) {
    throw std::runtime_error(error_buffer);
  }

  expand(LIBRARY_STATE r1, e1, sizes);
  expand(LIBRARY_STATE r2, e2, sizes);
}

template <typename TensorType1, typename TensorType2, typename TensorType3>
void expand3(LIBRARY_STATE_TYPE TensorType1 *r1, TensorType2 *r2, TensorType3 *r3,
             TensorType1 *e1, TensorType2 *e2, TensorType3 *e3,
             char *e1_name, char *e2_name, char *e3_name) {
  if (e1->nDimension <= 0) {
    throw std::runtime_error(std::string("can't expand empty tensor ").append(e1_name));
  }
  if (e2->nDimension <= 0) {
    throw std::runtime_error(std::string("can't expand empty tensor ").append(e2_name));
  }
  if (e3->nDimension <= 0) {
    throw std::runtime_error(std::string("can't expand empty tensor ").append(e3_name));
  }

  long *e_sizes[3];
  long e_dims[3];

  e_sizes[ 0 ] = e1->size;
  e_sizes[ 1 ] = e2->size;
  e_sizes[ 2 ] = e3->size;
  e_dims[ 0 ] = e1->nDimension;
  e_dims[ 1 ] = e2->nDimension;
  e_dims[ 2 ] = e3->nDimension;

  THLongStoragePtr sizes(THLongStorage_new());
  char error_buffer[1024];
  int ret = THLongStorage_inferSizeN(sizes,
                                     3,
                                     e_sizes,
                                     e_dims,
                                     error_buffer,
                                     1024);

  if(ret != 0) {
    throw std::runtime_error(error_buffer);
  }

  expand(LIBRARY_STATE r1, e1, sizes);
  expand(LIBRARY_STATE r2, e2, sizes);
  expand(LIBRARY_STATE r3, e3, sizes);
}

template <typename ExpandType, typename TensorType>
void check_backincompat_expand_warn(ExpandType *to_expand, TensorType *tensor,
                                    char *to_expand_name, char *tensor_name, bool fallback,
                                    ptrdiff_t to_expand_nElem, ptrdiff_t tensor_nElem) {
  if (fallback && getBackCompatBroadcastWarn()) {
    bool same_shape = THSize_isSameSizeAs(tensor->size, tensor->nDimension,
        to_expand->size, to_expand->nDimension);
    if (!same_shape && (tensor_nElem == to_expand_nElem)) {
      std::ostringstream warn;
      warn << tensor_name << " and " << to_expand_name << " do not have the same shape, but are "
           << "broadcastable, and have the same number of elements.  Changing behavior in a backwards incompatible "
           << "manner to broadcasting rather than viewing as 1-dimensional.";
      PyErr_WarnEx(PyExc_UserWarning, warn.str().c_str(), 1);
    }
  }
}

template <typename ExpandType, typename TensorType>
void expand_inplace(LIBRARY_STATE_TYPE ExpandType *r, ExpandType *to_expand, TensorType *tensor,
                    char *to_expand_name, char *tensor_name, bool fallback,
                    THLongStorage *tensor_size,  ptrdiff_t to_expand_nElem, ptrdiff_t tensor_nElem,
                    bool warn_pointwise_fallback) {
  try {
    expand<ExpandType>(LIBRARY_STATE r, to_expand, tensor_size);
  } catch (std::exception &e) {
    if (warn_pointwise_fallback) {
      std::ostringstream warn;
      warn << to_expand_name << " is not broadcastable to " << tensor_name
           << ", but they have the same number of elements.  Falling back to deprecated pointwise behavior.";
      PyErr_WarnEx(PyExc_UserWarning, warn.str().c_str(), 1);
    }
    throw;
  }
}

template <typename ExpandType, typename TensorType>
void expand_inplace1(LIBRARY_STATE_TYPE ExpandType *r, ExpandType *to_expand, TensorType *tensor,
                     char *to_expand_name, char *tensor_name, bool fallback) {
  ptrdiff_t to_expand_nElem = THSize_nElement(to_expand->nDimension, to_expand->size);
  ptrdiff_t tensor_nElem = THSize_nElement(tensor->nDimension, tensor->size);
  bool to_expand_warn = fallback && (to_expand_nElem == tensor_nElem) && to_expand_nElem != 0;
  THLongStoragePtr tensor_size(THLongStorage_newWithSize(tensor->nDimension));
  THLongStorage_rawCopy(tensor_size.get(), tensor->size);

  expand_inplace(LIBRARY_STATE r, to_expand, tensor, to_expand_name, tensor_name, fallback,
                 tensor_size, to_expand_nElem, tensor_nElem, to_expand_warn);
  check_backincompat_expand_warn<ExpandType, TensorType>(to_expand, tensor, to_expand_name, tensor_name, fallback,
                                                         to_expand_nElem, tensor_nElem);
}

template <typename TensorType>
void expand_inplace2(LIBRARY_STATE_TYPE TensorType *r1, TensorType *r2,
                     TensorType *to_expand1, TensorType *to_expand2, TensorType *tensor,
                     char *to_expand1_name, char *to_expand2_name, char *tensor_name, bool fallback) {
  ptrdiff_t tensor_nElem = THSize_nElement(tensor->nDimension, tensor->size);
  ptrdiff_t to_expand1_nElem = THSize_nElement(to_expand1->nDimension, to_expand1->size);
  ptrdiff_t to_expand2_nElem = THSize_nElement(to_expand2->nDimension, to_expand2->size);
  bool to_expand1_warn = fallback && (tensor_nElem == to_expand1_nElem) && tensor_nElem != 0;
  bool to_expand2_warn = fallback && (tensor_nElem == to_expand2_nElem) && tensor_nElem != 0;
  THLongStoragePtr tensor_size(THLongStorage_newWithSize(tensor->nDimension));
  THLongStorage_rawCopy(tensor_size.get(), tensor->size);

  expand_inplace(LIBRARY_STATE r1, to_expand1, tensor, to_expand1_name, tensor_name, fallback,
                 tensor_size, to_expand1_nElem, tensor_nElem, to_expand1_warn && to_expand2_warn);
  expand_inplace(LIBRARY_STATE r2, to_expand2, tensor, to_expand2_name, tensor_name, fallback,
                 tensor_size, to_expand2_nElem, tensor_nElem, to_expand1_warn && to_expand2_warn);

  check_backincompat_expand_warn<TensorType, TensorType>(to_expand1, tensor, to_expand1_name, tensor_name, fallback,
                                                         to_expand1_nElem, tensor_nElem);
  check_backincompat_expand_warn<TensorType, TensorType>(to_expand2, tensor, to_expand2_name, tensor_name, fallback,
                                                         to_expand2_nElem, tensor_nElem);
}

template <typename TensorType1, typename TensorType2>
void expand_outplace2(LIBRARY_STATE_TYPE TensorType1 *r1, TensorType2 *r2,
                      TensorType1 *to_expand1, TensorType2 *to_expand2,
                      char *to_expand1_name, char *to_expand2_name, bool fallback) {
  ptrdiff_t to_expand1_nElem = THSize_nElement(to_expand1->nDimension, to_expand1->size);
  ptrdiff_t to_expand2_nElem = THSize_nElement(to_expand2->nDimension, to_expand2->size);
  bool expand_warn = fallback && (to_expand1_nElem == to_expand2_nElem) && to_expand1_nElem != 0;
  try {
    expand2<TensorType1, TensorType2>(LIBRARY_STATE r1, r2, to_expand1, to_expand2, to_expand1_name, to_expand2_name);
  } catch (std::exception &e) {
    if (expand_warn) {
      std::ostringstream warn;
      warn << to_expand1_name << " and " << to_expand2_name << " not broadcastable, but have the same number of "
           << "elements.  Falling back to deprecated pointwise behavior.";
      PyErr_WarnEx(PyExc_UserWarning, warn.str().c_str(), 1);
    }
    throw;
  }

  check_backincompat_expand_warn<TensorType1, TensorType2>(to_expand1, to_expand2, to_expand1_name, to_expand2_name,
                                                           fallback, to_expand1_nElem, to_expand2_nElem);
}

template <typename TensorType1, typename TensorType2, typename TensorType3>
void expand_outplace3(LIBRARY_STATE_TYPE TensorType1 *r1, TensorType2 *r2, TensorType3 *r3,
                      TensorType1 *to_expand1, TensorType2 *to_expand2, TensorType3 *to_expand3,
                      char *to_expand1_name, char *to_expand2_name, char *to_expand3_name, bool fallback) {
  ptrdiff_t to_expand1_nElem = THSize_nElement(to_expand1->nDimension, to_expand1->size);
  ptrdiff_t to_expand2_nElem = THSize_nElement(to_expand2->nDimension, to_expand2->size);
  ptrdiff_t to_expand3_nElem = THSize_nElement(to_expand3->nDimension, to_expand3->size);
  bool to_expand2_warn = fallback && (to_expand1_nElem == to_expand2_nElem) && to_expand1_nElem != 0;
  bool to_expand3_warn = fallback && (to_expand1_nElem == to_expand3_nElem) && to_expand1_nElem != 0;

  try {
    expand3<TensorType1, TensorType2, TensorType3>(LIBRARY_STATE r1, r2, r3,
                                                   to_expand1, to_expand2, to_expand3,
                                                   to_expand1_name, to_expand2_name, to_expand3_name);
  } catch (std::exception &e) {
    if(to_expand2_warn && to_expand3_warn) {
      std::ostringstream warn;
      warn << to_expand1_name << ", " << to_expand2_name << ", and " << to_expand3_name << " not broadcastable,"
           << " but have the same number of elements.  Falling back to deprecated pointwise behavior.";
      PyErr_WarnEx(PyExc_UserWarning, warn.str().c_str(), 1);
    }
    throw;
  }

  check_backincompat_expand_warn<TensorType1, TensorType2>(to_expand1, to_expand2, to_expand1_name, to_expand2_name,
                                                         fallback, to_expand1_nElem, to_expand2_nElem);
  check_backincompat_expand_warn<TensorType1, TensorType3>(to_expand1, to_expand3, to_expand1_name, to_expand3_name,
                                                         fallback, to_expand1_nElem, to_expand3_nElem);
}

#endif
