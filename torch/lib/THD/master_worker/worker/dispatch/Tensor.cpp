
static std::unique_ptr<Tensor> createTensor(Type type) {
  if (type == Type::UCHAR)
    return std::unique_ptr<Tensor>(new THTensor<unsigned char>());
  else if (type == Type::CHAR)
    return std::unique_ptr<Tensor>(new THTensor<char>());
  else if (type == Type::SHORT)
    return std::unique_ptr<Tensor>(new THTensor<short>());
  else if (type == Type::INT)
    return std::unique_ptr<Tensor>(new THTensor<int>());
  else if (type == Type::LONG)
    return std::unique_ptr<Tensor>(new THTensor<long>());
  else if (type == Type::FLOAT)
    return std::unique_ptr<Tensor>(new THTensor<float>());
  else if (type == Type::DOUBLE)
    return std::unique_ptr<Tensor>(new THTensor<double>());
  throw std::invalid_argument("passed character doesn't represent a tensor type");
}

static void tensorConstruct(rpc::RPCMessage& raw_message) {
  Type type = unpackType(raw_message);
  thd::object_id_type id = unpackTensor(raw_message);
  finalize(raw_message);
  workerTensors.emplace(
    id,
    createTensor(type)
  );
}

static void tensorConstructWithSize(rpc::RPCMessage& raw_message) {
  Type type = unpackType(raw_message);
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  THLongStorage *size = unpackTHLongStorage(raw_message);
  THLongStorage *stride = unpackTHLongStorage(raw_message);
  finalize(raw_message);
  // TODO
  THLongStorage_free(size);
  THLongStorage_free(stride);
  throw std::runtime_error("construct with size is not yet implemented");
}

static void tensorResize(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  THLongStorage *size = unpackTHLongStorage(raw_message);
  THLongStorage *stride = unpackTHLongStorage(raw_message);
  finalize(raw_message);
  tensor->resize(size, stride);
  THLongStorage_free(size);
  THLongStorage_free(stride);
}

static void tensorResizeAs(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  Tensor *src = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  tensor->resizeAs(*src);
}

static void tensorResizeNd(rpc::RPCMessage& raw_message, std::size_t N) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  std::vector<long> size(N);
  for (std::size_t i = 0; i < N; ++i) {
    size[i] = unpackInteger(raw_message);
  }
  finalize(raw_message);
  tensor->resize(size);
}

static void tensorResize1d(rpc::RPCMessage& raw_message) {
  tensorResizeNd(raw_message, 1);
}

static void tensorResize2d(rpc::RPCMessage& raw_message) {
  tensorResizeNd(raw_message, 2);
}

static void tensorResize3d(rpc::RPCMessage& raw_message) {
  tensorResizeNd(raw_message, 3);
}

static void tensorResize4d(rpc::RPCMessage& raw_message) {
  tensorResizeNd(raw_message, 4);
}

static void tensorResize5d(rpc::RPCMessage& raw_message) {
  tensorResizeNd(raw_message, 5);
}

static void tensorSet(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  Tensor *src = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  tensor->set(*src);
}

static void tensorSetStorage(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  Storage *storage = unpackRetrieveStorage(raw_message);
  ptrdiff_t storageOffset = unpackInteger(raw_message);
  THLongStorage *size = unpackTHLongStorage(raw_message);
  THLongStorage *stride = unpackTHLongStorage(raw_message);
  finalize(raw_message);
  tensor->setStorage(
      *storage,
      storageOffset,
      size,
      stride
  );
  THLongStorage_free(size);
  THLongStorage_free(stride);
}

static void tensorSetStorage1d(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  Storage *storage = unpackRetrieveStorage(raw_message);
  ptrdiff_t storageOffset = unpackInteger(raw_message);
  long size0 = unpackInteger(raw_message);
  long stride0 = unpackInteger(raw_message);
  finalize(raw_message);
  THLongStorage *sizes = THLongStorage_newWithSize1(size0);
  THLongStorage *strides = THLongStorage_newWithSize1(stride0);
  tensor->setStorage(
    *storage,
    storageOffset,
    sizes,
    strides
  );
  THLongStorage_free(sizes);
  THLongStorage_free(strides);
}

static void tensorSetStorage2d(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  Storage *storage = unpackRetrieveStorage(raw_message);
  ptrdiff_t storageOffset = unpackInteger(raw_message);
  long size0 = unpackInteger(raw_message);
  long stride0 = unpackInteger(raw_message);
  long size1 = unpackInteger(raw_message);
  long stride1 = unpackInteger(raw_message);
  finalize(raw_message);
  THLongStorage *sizes = THLongStorage_newWithSize2(size0, size1);
  THLongStorage *strides = THLongStorage_newWithSize2(stride0, stride1);
  tensor->setStorage(
    *storage,
    storageOffset,
    sizes,
    strides
  );
  THLongStorage_free(sizes);
  THLongStorage_free(strides);
}

static void tensorSetStorage3d(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  Storage *storage = unpackRetrieveStorage(raw_message);
  ptrdiff_t storageOffset = unpackInteger(raw_message);
  long size0 = unpackInteger(raw_message);
  long stride0 = unpackInteger(raw_message);
  long size1 = unpackInteger(raw_message);
  long stride1 = unpackInteger(raw_message);
  long size2 = unpackInteger(raw_message);
  long stride2 = unpackInteger(raw_message);
  finalize(raw_message);
  THLongStorage *sizes = THLongStorage_newWithSize3(size0, size1, size2);
  THLongStorage *strides = THLongStorage_newWithSize3(stride0, stride1, stride2);
  tensor->setStorage(
    *storage,
    storageOffset,
    sizes,
    strides
  );
  THLongStorage_free(sizes);
  THLongStorage_free(strides);
}

static void tensorSetStorage4d(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  Storage *storage = unpackRetrieveStorage(raw_message);
  ptrdiff_t storageOffset = unpackInteger(raw_message);
  long size0 = unpackInteger(raw_message);
  long stride0 = unpackInteger(raw_message);
  long size1 = unpackInteger(raw_message);
  long stride1 = unpackInteger(raw_message);
  long size2 = unpackInteger(raw_message);
  long stride2 = unpackInteger(raw_message);
  long size3 = unpackInteger(raw_message);
  long stride3 = unpackInteger(raw_message);
  finalize(raw_message);
  THLongStorage *sizes = THLongStorage_newWithSize4(size0, size1, size2, size3);
  THLongStorage *strides = THLongStorage_newWithSize4(stride0, stride1,
                                                      stride2, stride3);
  tensor->setStorage(
    *storage,
    storageOffset,
    sizes,
    strides
  );
  THLongStorage_free(sizes);
  THLongStorage_free(strides);
}

static void tensorNarrow(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  Tensor *src = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  long firstIndex = unpackInteger(raw_message);
  long size = unpackInteger(raw_message);
  finalize(raw_message);
  tensor->narrow(*src, dimension, firstIndex, size);
}

static void tensorSelect(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  Tensor *src = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  long sliceIndex = unpackInteger(raw_message);
  finalize(raw_message);
  tensor->select(*src, dimension, sliceIndex);
}

static void tensorTranspose(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  Tensor *src = unpackRetrieveTensor(raw_message);
  int dimension1 = unpackInteger(raw_message);
  int dimension2 = unpackInteger(raw_message);
  finalize(raw_message);
  // This does THTensor_(transpose)(self, NULL, d1, d2)
  tensor->transpose(*src, dimension1, dimension2);
}

static void tensorUnfold(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  Tensor *src = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  long size = unpackInteger(raw_message);
  long step = unpackInteger(raw_message);
  finalize(raw_message);
  tensor->unfold(*src, dimension, size, step);
}

static void tensorFree(rpc::RPCMessage& raw_message) {
  object_id_type tensor_id = unpackInteger(raw_message);
  (void)workerTensors.erase(tensor_id);
}

static void tensorGather(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  Tensor *src = unpackRetrieveTensor(raw_message);
  int dim = unpackInteger(raw_message);
  Tensor *index = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  tensor->gather(*src, dim, *index);
}

static void tensorScatter(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  int dim = unpackInteger(raw_message);
  Tensor *index = unpackRetrieveTensor(raw_message);
  Tensor *src = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  tensor->scatter(dim, *index, *src);
}

static void tensorScatterFill(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  int dim = unpackInteger(raw_message);
  Tensor *index = unpackRetrieveTensor(raw_message);

  if (isInteger(tensor->type())) {
    long long value = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<IntTensor*>(tensor)->scatterFill(dim, *index, value);
  } else if (isFloat(tensor->type())) {
    double value = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<FloatTensor*>(tensor)->scatterFill(dim, *index, value);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorDot(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  Tensor *src = unpackRetrieveTensor(raw_message);
  finalize(raw_message);

  if (isInteger(tensor->type())) {
    IntTensor* int_tensor = dynamic_cast<IntTensor*>(tensor);
    long long value = int_tensor->dot(*src);
    sendValueToMaster(int_tensor, value);
  } else if (isFloat(tensor->type())) {
    FloatTensor* float_tensor = dynamic_cast<FloatTensor*>(tensor);
    double value = float_tensor->dot(*src);
    sendValueToMaster(float_tensor, value);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorMinall(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  finalize(raw_message);

  if (isInteger(tensor->type())) {
    IntTensor* int_tensor = dynamic_cast<IntTensor*>(tensor);
    long long value = int_tensor->minall();
    sendValueToMaster(int_tensor, value);
  } else if (isFloat(tensor->type())) {
    FloatTensor* float_tensor = dynamic_cast<FloatTensor*>(tensor);
    double value = float_tensor->minall();
    sendValueToMaster(float_tensor, value);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorMaxall(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  finalize(raw_message);

  if (isInteger(tensor->type())) {
    IntTensor* int_tensor = dynamic_cast<IntTensor*>(tensor);
    long long value = int_tensor->maxall();
    sendValueToMaster(int_tensor, value);
  } else if (isFloat(tensor->type())) {
    FloatTensor* float_tensor = dynamic_cast<FloatTensor*>(tensor);
    double value = float_tensor->maxall();
    sendValueToMaster(float_tensor, value);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorSumall(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  finalize(raw_message);

  if (isInteger(tensor->type())) {
    IntTensor* int_tensor = dynamic_cast<IntTensor*>(tensor);
    long long value = int_tensor->sumall();
    sendValueToMaster(int_tensor, value);
  } else if (isFloat(tensor->type())) {
    FloatTensor* float_tensor = dynamic_cast<FloatTensor*>(tensor);
    double value = float_tensor->sumall();
    sendValueToMaster(float_tensor, value);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorProdall(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  finalize(raw_message);

  if (isInteger(tensor->type())) {
    IntTensor* int_tensor = dynamic_cast<IntTensor*>(tensor);
    long long value = int_tensor->prodall();
    sendValueToMaster(int_tensor, value);
  } else if (isFloat(tensor->type())) {
    FloatTensor* float_tensor = dynamic_cast<FloatTensor*>(tensor);
    double value = float_tensor->prodall();
    sendValueToMaster(float_tensor, value);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorNeg(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  Tensor *src = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  tensor->neg(*src);
}

static void tensorCinv(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  Tensor *src = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  tensor->cinv(*src);
}

static void tensorAdd(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  Tensor *src = unpackRetrieveTensor(raw_message);

  if (isInteger(tensor->type())) {
    long long value = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<IntTensor*>(tensor)->add(*src, value);
  } else if (isFloat(tensor->type())) {
    double value = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<FloatTensor*>(tensor)->add(*src, value);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorSub(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  Tensor *src = unpackRetrieveTensor(raw_message);

  if (isInteger(tensor->type())) {
    long long value = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<IntTensor*>(tensor)->sub(*src, value);
  } else if (isFloat(tensor->type())) {
    double value = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<FloatTensor*>(tensor)->sub(*src, value);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorMul(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  Tensor *src = unpackRetrieveTensor(raw_message);

  if (isInteger(tensor->type())) {
    long long value = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<IntTensor*>(tensor)->mul(*src, value);
  } else if (isFloat(tensor->type())) {
    double value = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<FloatTensor*>(tensor)->mul(*src, value);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorDiv(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  Tensor *src = unpackRetrieveTensor(raw_message);

  if (isInteger(tensor->type())) {
    long long value = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<IntTensor*>(tensor)->div(*src, value);
  } else if (isFloat(tensor->type())) {
    double value = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<FloatTensor*>(tensor)->div(*src, value);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorFmod(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  Tensor *src = unpackRetrieveTensor(raw_message);

  if (isInteger(tensor->type())) {
    long long value = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<IntTensor*>(tensor)->fmod(*src, value);
  } else {
    double value = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<FloatTensor*>(tensor)->fmod(*src, value);
  }
}

static void tensorRemainder(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  Tensor *src = unpackRetrieveTensor(raw_message);

  if (isInteger(tensor->type())) {
    long long value = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<IntTensor*>(tensor)->remainder(*src, value);
  } else if (isFloat(tensor->type())) {
    double value = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<FloatTensor*>(tensor)->remainder(*src, value);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorClamp(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  Tensor *src = unpackRetrieveTensor(raw_message);

  if (isInteger(tensor->type())) {
    long long min_value = unpackInteger(raw_message);
    long long max_value = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<IntTensor*>(tensor)->clamp(*src, min_value, max_value);
  } else if (isFloat(tensor->type())) {
    double min_value = unpackFloat(raw_message);
    double max_value = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<FloatTensor*>(tensor)->clamp(*src, min_value, max_value);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorCadd(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  Tensor *src1 = unpackRetrieveTensor(raw_message);
  Tensor *src2 = unpackRetrieveTensor(raw_message);

  if (isInteger(tensor->type())) {
    long long value = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<IntTensor*>(tensor)->cadd(*src1, value, *src2);
  } else if (isFloat(tensor->type())) {
    double value = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<FloatTensor*>(tensor)->cadd(*src1, value, *src2);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorCsub(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  Tensor *src1 = unpackRetrieveTensor(raw_message);
  Tensor *src2 = unpackRetrieveTensor(raw_message);

  if (isInteger(tensor->type())) {
    long long value = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<IntTensor*>(tensor)->csub(*src1, value, *src2);
  } else if (isFloat(tensor->type())) {
    double value = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<FloatTensor*>(tensor)->csub(*src1, value, *src2);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorCmul(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  Tensor *src1 = unpackRetrieveTensor(raw_message);
  Tensor *src2 = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  tensor->cmul(*src1, *src2);
}

static void tensorCpow(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  Tensor *src1 = unpackRetrieveTensor(raw_message);
  Tensor *src2 = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  tensor->cpow(*src1, *src2);
}

static void tensorCdiv(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  Tensor *src1 = unpackRetrieveTensor(raw_message);
  Tensor *src2 = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  tensor->cdiv(*src1, *src2);
}

static void tensorCfmod(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  Tensor *src1 = unpackRetrieveTensor(raw_message);
  Tensor *src2 = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  tensor->cfmod(*src1, *src2);
}

static void tensorCremainder(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  Tensor *src1 = unpackRetrieveTensor(raw_message);
  Tensor *src2 = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  tensor->cremainder(*src1, *src2);
}

static void tensorAddcmul(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  Tensor *src1 = unpackRetrieveTensor(raw_message);
  Tensor *src2 = unpackRetrieveTensor(raw_message);
  Tensor *src3 = unpackRetrieveTensor(raw_message);

  if (isInteger(tensor->type())) {
    long long value = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<IntTensor*>(tensor)->addcmul(*src1, value, *src2, *src3);
  } else if (isFloat(tensor->type())) {
    double value = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<FloatTensor*>(tensor)->addcmul(*src1, value, *src2, *src3);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorAddcdiv(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  Tensor *src1 = unpackRetrieveTensor(raw_message);
  Tensor *src2 = unpackRetrieveTensor(raw_message);
  Tensor *src3 = unpackRetrieveTensor(raw_message);

  if (isInteger(tensor->type())) {
    long long value = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<IntTensor*>(tensor)->addcdiv(*src1, value, *src2, *src3);
  } else if (isFloat(tensor->type())) {
    double value = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<FloatTensor*>(tensor)->addcdiv(*src1, value, *src2, *src3);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorAddmv(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  Tensor *src = unpackRetrieveTensor(raw_message);
  Tensor *mat = unpackRetrieveTensor(raw_message);
  Tensor *vec = unpackRetrieveTensor(raw_message);

  if (isInteger(tensor->type())) {
    long long beta = unpackInteger(raw_message);
    long long alpha = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<IntTensor*>(tensor)->addmv(beta, *src, alpha, *mat, *vec);
  } else if (isFloat(tensor->type())) {
    double beta = unpackFloat(raw_message);
    double alpha = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<FloatTensor*>(tensor)->addmv(beta, *src, alpha, *mat, *vec);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorAddmm(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  Tensor *src = unpackRetrieveTensor(raw_message);
  Tensor *mat1 = unpackRetrieveTensor(raw_message);
  Tensor *mat2 = unpackRetrieveTensor(raw_message);

  if (isInteger(tensor->type())) {
    long long beta = unpackInteger(raw_message);
    long long alpha = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<IntTensor*>(tensor)->addmm(beta, *src, alpha, *mat1, *mat2);
  } else if (isFloat(tensor->type())) {
    double beta = unpackFloat(raw_message);
    double alpha = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<FloatTensor*>(tensor)->addmm(beta, *src, alpha, *mat1, *mat2);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorAddr(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  Tensor *src = unpackRetrieveTensor(raw_message);
  Tensor *vec1 = unpackRetrieveTensor(raw_message);
  Tensor *vec2 = unpackRetrieveTensor(raw_message);

  if (isInteger(tensor->type())) {
    long long beta = unpackInteger(raw_message);
    long long alpha = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<IntTensor*>(tensor)->addr(beta, *src, alpha, *vec1, *vec2);
  } else if (isFloat(tensor->type())) {
    double beta = unpackFloat(raw_message);
    double alpha = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<FloatTensor*>(tensor)->addr(beta, *src, alpha, *vec1, *vec2);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorAddbmm(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  Tensor *src = unpackRetrieveTensor(raw_message);
  Tensor *batch1 = unpackRetrieveTensor(raw_message);
  Tensor *batch2 = unpackRetrieveTensor(raw_message);

  if (isInteger(tensor->type())) {
    long long beta = unpackInteger(raw_message);
    long long alpha = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<IntTensor*>(tensor)->addbmm(beta, *src, alpha, *batch1, *batch2);
  } else if (isFloat(tensor->type())) {
    double beta = unpackFloat(raw_message);
    double alpha = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<FloatTensor*>(tensor)->addbmm(beta, *src, alpha, *batch1, *batch2);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorBaddbmm(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  Tensor *src = unpackRetrieveTensor(raw_message);
  Tensor *batch1 = unpackRetrieveTensor(raw_message);
  Tensor *batch2 = unpackRetrieveTensor(raw_message);

  if (isInteger(tensor->type())) {
    long long beta = unpackInteger(raw_message);
    long long alpha = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<IntTensor*>(tensor)->baddbmm(beta, *src, alpha, *batch1, *batch2);
  } else if (isFloat(tensor->type())) {
    double beta = unpackFloat(raw_message);
    double alpha = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<FloatTensor*>(tensor)->baddbmm(beta, *src, alpha, *batch1, *batch2);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorMatch(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  Tensor *m1 = unpackRetrieveTensor(raw_message);
  Tensor *m2 = unpackRetrieveTensor(raw_message);

  if (isInteger(tensor->type())) {
    long long gain = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<IntTensor*>(tensor)->match(*m1, *m2, gain);
  } else if (isFloat(tensor->type())) {
    double gain = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<FloatTensor*>(tensor)->match(*m1, *m2, gain);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorMax(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  Tensor *indices_ = unpackRetrieveTensor(raw_message);
  Tensor *src = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  finalize(raw_message);
  tensor->max(*indices_, *src, dimension);
}

static void tensorMin(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  Tensor *indices_ = unpackRetrieveTensor(raw_message);
  Tensor *src = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  finalize(raw_message);
  tensor->min(*indices_, *src, dimension);
}

static void tensorKthvalue(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  Tensor *indices_ = unpackRetrieveTensor(raw_message);
  Tensor *src = unpackRetrieveTensor(raw_message);
  int k = unpackInteger(raw_message);
  int dimension = unpackInteger(raw_message);
  finalize(raw_message);
  tensor->kthvalue(*indices_, *src, k, dimension);
}

static void tensorMode(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  Tensor *indices_ = unpackRetrieveTensor(raw_message);
  Tensor *src = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  finalize(raw_message);
  tensor->mode(*indices_, *src, dimension);
}

static void tensorMedian(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  Tensor *indices_ = unpackRetrieveTensor(raw_message);
  Tensor *src = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  finalize(raw_message);
  tensor->median(*indices_, *src, dimension);
}

static void tensorSum(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  Tensor *src = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  finalize(raw_message);
  tensor->sum(*src, dimension);
}

static void tensorProd(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  Tensor *src = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  finalize(raw_message);
  tensor->prod(*src, dimension);
}

static void tensorCumsum(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  Tensor *src = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  finalize(raw_message);
  tensor->cumsum(*src, dimension);
}

static void tensorCumprod(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  Tensor *src = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  finalize(raw_message);
  tensor->cumprod(*src, dimension);
}

static void tensorSign(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  Tensor *src = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  tensor->sign(*src);
}

static void tensorTrace(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  finalize(raw_message);

  if (isInteger(tensor->type())) {
    IntTensor* int_tensor = dynamic_cast<IntTensor*>(tensor);
    long long value = int_tensor->trace();
    sendValueToMaster(int_tensor, value);
  } else if (isFloat(tensor->type())) {
    FloatTensor* float_tensor = dynamic_cast<FloatTensor*>(tensor);
    double value = float_tensor->trace();
    sendValueToMaster(float_tensor, value);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorCross(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  Tensor *src1 = unpackRetrieveTensor(raw_message);
  Tensor *src2 = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  finalize(raw_message);
  tensor->cross(*src1, *src2, dimension);
}

static void tensorCmax(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  Tensor *src1 = unpackRetrieveTensor(raw_message);
  Tensor *src2 = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  tensor->cmax(*src1, *src2);
}

static void tensorCmin(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  Tensor *src1 = unpackRetrieveTensor(raw_message);
  Tensor *src2 = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  tensor->cmin(*src1, *src2);
}

static void tensorCmaxValue(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  Tensor *src = unpackRetrieveTensor(raw_message);

  if (isInteger(tensor->type())) {
    long long value = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<IntTensor*>(tensor)->cmaxValue(*src, value);
  } else if (isFloat(tensor->type())) {
    double value = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<FloatTensor*>(tensor)->cmaxValue(*src, value);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorCminValue(rpc::RPCMessage& raw_message) {
  Tensor *tensor = unpackRetrieveTensor(raw_message);
  Tensor *src = unpackRetrieveTensor(raw_message);

  if (isInteger(tensor->type())) {
    long long value = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<IntTensor*>(tensor)->cminValue(*src, value);
  } else if (isFloat(tensor->type())) {
    double value = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<FloatTensor*>(tensor)->cminValue(*src, value);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}
