template <>
THFloatTensor *newForExpand(LIBRARY_STATE_TYPE_NOARGS) {
  return THFloatTensor_new();
}

template <>
THDoubleTensor *newForExpand(LIBRARY_STATE_TYPE_NOARGS) {
  return THDoubleTensor_new();
}

template <>
THHalfTensor *newForExpand(LIBRARY_STATE_TYPE_NOARGS) {
  return THHalfTensor_new();
}

template <>
THByteTensor *newForExpand(LIBRARY_STATE_TYPE_NOARGS) {
  return THByteTensor_new();
}

template <>
THCharTensor *newForExpand(LIBRARY_STATE_TYPE_NOARGS) {
  return THCharTensor_new();
}

template <>
THShortTensor *newForExpand(LIBRARY_STATE_TYPE_NOARGS) {
  return THShortTensor_new();
}

template <>
THIntTensor *newForExpand(LIBRARY_STATE_TYPE_NOARGS) {
  return THIntTensor_new();
}

template <>
THLongTensor *newForExpand(LIBRARY_STATE_TYPE_NOARGS) {
  return THLongTensor_new();
}

template<>
int expand(LIBRARY_STATE_TYPE THFloatTensor *r, THFloatTensor *tensor, THLongStorage *sizes, int raiseErrors) {
  return THFloatTensor_expand(r, tensor, sizes, raiseErrors);
}

template<>
int expand(LIBRARY_STATE_TYPE THDoubleTensor *r, THDoubleTensor *tensor, THLongStorage *sizes, int raiseErrors) {
  return THDoubleTensor_expand(r, tensor, sizes, raiseErrors);
}

template<>
int expand(LIBRARY_STATE_TYPE THHalfTensor *r, THHalfTensor *tensor, THLongStorage *sizes, int raiseErrors) {
  return THHalfTensor_expand(r, tensor, sizes, raiseErrors);
}

template<>
int expand(LIBRARY_STATE_TYPE THByteTensor *r, THByteTensor *tensor, THLongStorage *sizes, int raiseErrors) {
  return THByteTensor_expand(r, tensor, sizes, raiseErrors);
}

template<>
int expand(LIBRARY_STATE_TYPE THCharTensor *r, THCharTensor *tensor, THLongStorage *sizes, int raiseErrors) {
  return THCharTensor_expand(r, tensor, sizes, raiseErrors);
}

template<>
int expand(LIBRARY_STATE_TYPE THShortTensor *r, THShortTensor *tensor, THLongStorage *sizes, int raiseErrors) {
  return THShortTensor_expand(r, tensor, sizes, raiseErrors);
}

template<>
int expand(LIBRARY_STATE_TYPE THIntTensor *r, THIntTensor *tensor, THLongStorage *sizes, int raiseErrors) {
  return THIntTensor_expand(r, tensor, sizes, raiseErrors);
}

template<>
int expand(LIBRARY_STATE_TYPE THLongTensor *r, THLongTensor *tensor, THLongStorage *sizes, int raiseErrors) {
  return THLongTensor_expand(r, tensor, sizes, raiseErrors);
}

template <>
int expand2(LIBRARY_STATE_TYPE THFloatTensor *r1, THFloatTensor *r2,
            THFloatTensor *e1, THFloatTensor *e2, int raiseErrors) {
  return THFloatTensor_expand2(r1, r2, e1, e2, raiseErrors);
}

template <>
int expand2(LIBRARY_STATE_TYPE THDoubleTensor *r1, THDoubleTensor *r2,
            THDoubleTensor *e1, THDoubleTensor *e2, int raiseErrors) {
  return THDoubleTensor_expand2(r1, r2, e1, e2, raiseErrors);
}

template <>
int expand2(LIBRARY_STATE_TYPE THHalfTensor *r1, THHalfTensor *r2,
            THHalfTensor *e1, THHalfTensor *e2, int raiseErrors) {
  return THHalfTensor_expand2(r1, r2, e1, e2, raiseErrors);
}

template <>
int expand2(LIBRARY_STATE_TYPE THByteTensor *r1, THByteTensor *r2,
            THByteTensor *e1, THByteTensor *e2, int raiseErrors) {
  return THByteTensor_expand2(r1, r2, e1, e2, raiseErrors);
}

template <>
int expand2(LIBRARY_STATE_TYPE THCharTensor *r1, THCharTensor *r2,
            THCharTensor *e1, THCharTensor *e2, int raiseErrors) {
  return THCharTensor_expand2(r1, r2, e1, e2, raiseErrors);
}

template <>
int expand2(LIBRARY_STATE_TYPE THShortTensor *r1, THShortTensor *r2,
            THShortTensor *e1, THShortTensor *e2, int raiseErrors) {
  return THShortTensor_expand2(r1, r2, e1, e2, raiseErrors);
}

template <>
int expand2(LIBRARY_STATE_TYPE THIntTensor *r1, THIntTensor *r2,
            THIntTensor *e1, THIntTensor *e2, int raiseErrors) {
  return THIntTensor_expand2(r1, r2, e1, e2, raiseErrors);
}

template <>
int expand2(LIBRARY_STATE_TYPE THLongTensor *r1, THLongTensor *r2,
            THLongTensor *e1, THLongTensor *e2, int raiseErrors) {
  return THLongTensor_expand2(r1, r2, e1, e2, raiseErrors);
}

template <>
int expand3(LIBRARY_STATE_TYPE THFloatTensor *r1, THFloatTensor *r2, THFloatTensor *r3,
            THFloatTensor *e1, THFloatTensor *e2, THFloatTensor *e3, int raiseErrors) {
  return THFloatTensor_expand3(r1, r2, r3, e1, e2, e3, raiseErrors);
}

template <>
int expand3(LIBRARY_STATE_TYPE THDoubleTensor *r1, THDoubleTensor *r2, THDoubleTensor *r3,
            THDoubleTensor *e1, THDoubleTensor *e2, THDoubleTensor *e3, int raiseErrors) {
  return THDoubleTensor_expand3(r1, r2, r3, e1, e2, e3, raiseErrors);
}

template <>
int expand3(LIBRARY_STATE_TYPE THHalfTensor *r1, THHalfTensor *r2, THHalfTensor *r3,
            THHalfTensor *e1, THHalfTensor *e2, THHalfTensor *e3, int raiseErrors) {
  return THHalfTensor_expand3(r1, r2, r3, e1, e2, e3, raiseErrors);
}

template <>
int expand3(LIBRARY_STATE_TYPE THByteTensor *r1, THByteTensor *r2, THByteTensor *r3,
            THByteTensor *e1, THByteTensor *e2, THByteTensor *e3, int raiseErrors) {
  return THByteTensor_expand3(r1, r2, r3, e1, e2, e3, raiseErrors);
}

template <>
int expand3(LIBRARY_STATE_TYPE THCharTensor *r1, THCharTensor *r2, THCharTensor *r3,
            THCharTensor *e1, THCharTensor *e2, THCharTensor *e3, int raiseErrors) {
  return THCharTensor_expand3(r1, r2, r3, e1, e2, e3, raiseErrors);
}

template <>
int expand3(LIBRARY_STATE_TYPE THShortTensor *r1, THShortTensor *r2, THShortTensor *r3,
            THShortTensor *e1, THShortTensor *e2, THShortTensor *e3, int raiseErrors) {
  return THShortTensor_expand3(r1, r2, r3, e1, e2, e3, raiseErrors);
}

template <>
int expand3(LIBRARY_STATE_TYPE THIntTensor *r1, THIntTensor *r2, THIntTensor *r3,
            THIntTensor *e1, THIntTensor *e2, THIntTensor *e3, int raiseErrors) {
  return THIntTensor_expand3(r1, r2, r3, e1, e2, e3, raiseErrors);
}

template <>
int expand3(LIBRARY_STATE_TYPE THLongTensor *r1, THLongTensor *r2, THLongTensor *r3,
            THLongTensor *e1, THLongTensor *e2, THLongTensor *e3, int raiseErrors) {
  return THLongTensor_expand3(r1, r2, r3, e1, e2, e3, raiseErrors);
}
