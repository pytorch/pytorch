
template<typename... Ts>
static std::unique_ptr<thpp::Tensor> createTensor(thpp::Type type, Ts &... args) {
  if (type == thpp::Type::UCHAR)
    return std::unique_ptr<thpp::Tensor>(
        new thpp::THTensor<unsigned char>(std::forward<Ts>(args)...));
  else if (type == thpp::Type::CHAR)
    return std::unique_ptr<thpp::Tensor>(
        new thpp::THTensor<char>(std::forward<Ts>(args)...));
  else if (type == thpp::Type::SHORT)
    return std::unique_ptr<thpp::Tensor>(
        new thpp::THTensor<short>(std::forward<Ts>(args)...));
  else if (type == thpp::Type::INT)
    return std::unique_ptr<thpp::Tensor>(
        new thpp::THTensor<int>(std::forward<Ts>(args)...));
  else if (type == thpp::Type::LONG)
    return std::unique_ptr<thpp::Tensor>(
        new thpp::THTensor<long>(std::forward<Ts>(args)...));
  else if (type == thpp::Type::FLOAT)
    return std::unique_ptr<thpp::Tensor>(
        new thpp::THTensor<float>(std::forward<Ts>(args)...));
  else if (type == thpp::Type::DOUBLE)
    return std::unique_ptr<thpp::Tensor>(
        new thpp::THTensor<double>(std::forward<Ts>(args)...));
  throw std::invalid_argument("passed character doesn't represent a tensor type");
}

static void tensorNew(rpc::RPCMessage& raw_message) {
  thpp::Type type = unpackType(raw_message);
  thd::object_id_type id = unpackTensor(raw_message);
  finalize(raw_message);
  workerTensors.emplace(
    id,
    createTensor(type)
  );
}

static void tensorNewWithSize(rpc::RPCMessage& raw_message) {
  thpp::Type type = unpackType(raw_message);
  thd::object_id_type id = unpackTensor(raw_message);
  THLongStorage *size = unpackTHLongStorage(raw_message);
  THLongStorage *stride = unpackTHLongStorage(raw_message);
  finalize(raw_message);
  workerTensors.emplace(
    id,
    createTensor(type, size, stride)
  );
  THLongStorage_free(size);
  THLongStorage_free(stride);
}

static void tensorNewWithStorage(rpc::RPCMessage& raw_message) {
  thpp::Type type = unpackType(raw_message);
  thd::object_id_type id = unpackTensor(raw_message);
  thpp::Storage *storage = unpackRetrieveStorage(raw_message);
  ptrdiff_t storageOffset = unpackInteger(raw_message);
  THLongStorage *size = unpackTHLongStorage(raw_message);
  THLongStorage *stride = unpackTHLongStorage(raw_message);
  finalize(raw_message);
  workerTensors.emplace(
    id,
    createTensor(type, *storage, storageOffset, size, stride)
  );
  THLongStorage_free(size);
  THLongStorage_free(stride);
}

static void tensorNewWithTensor(rpc::RPCMessage& raw_message) {
  thpp::Type type = unpackType(raw_message);
  thd::object_id_type id = unpackTensor(raw_message);
  thpp::Tensor *self = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  workerTensors.emplace(
    id,
    createTensor(type, *self)
  );
}

static void tensorNewClone(rpc::RPCMessage& raw_message) {
  thd::object_id_type id = unpackTensor(raw_message);
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  workerTensors.emplace(
    id,
    std::unique_ptr<thpp::Tensor>(tensor->clone())
  );
}

static void tensorResize(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  THLongStorage *size = unpackTHLongStorage(raw_message);
  THLongStorage *stride = unpackTHLongStorage(raw_message);
  finalize(raw_message);
  tensor->resize(size, stride);
  THLongStorage_free(size);
  THLongStorage_free(stride);
}

static void tensorResizeAs(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  tensor->resizeAs(*src);
}

static void tensorResizeNd(rpc::RPCMessage& raw_message, std::size_t N) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
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
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  tensor->set(*src);
}

static void tensorSetStorage(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Storage *storage = unpackRetrieveStorage(raw_message);
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
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Storage *storage = unpackRetrieveStorage(raw_message);
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
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Storage *storage = unpackRetrieveStorage(raw_message);
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
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Storage *storage = unpackRetrieveStorage(raw_message);
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
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Storage *storage = unpackRetrieveStorage(raw_message);
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
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  long firstIndex = unpackInteger(raw_message);
  long size = unpackInteger(raw_message);
  finalize(raw_message);
  tensor->narrow(*src, dimension, firstIndex, size);
}

static void tensorSelect(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  long sliceIndex = unpackInteger(raw_message);
  finalize(raw_message);
  tensor->select(*src, dimension, sliceIndex);
}

static void tensorTranspose(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src = unpackRetrieveTensor(raw_message);
  int dimension1 = unpackInteger(raw_message);
  int dimension2 = unpackInteger(raw_message);
  finalize(raw_message);
  tensor->transpose(*src, dimension1, dimension2);
}

static void tensorUnfold(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  long size = unpackInteger(raw_message);
  long step = unpackInteger(raw_message);
  finalize(raw_message);
  tensor->unfold(*src, dimension, size, step);
}

static void tensorSqueeze(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  tensor->squeeze(*src);
}

static void tensorSqueeze1d(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  finalize(raw_message);
  tensor->squeeze(*src, dimension);
}

static void tensorFree(rpc::RPCMessage& raw_message) {
  object_id_type tensor_id = unpackTensor(raw_message);
  (void)workerTensors.erase(tensor_id);
}

static void tensorGather(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src = unpackRetrieveTensor(raw_message);
  int dim = unpackInteger(raw_message);
  thpp::Tensor *index = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  tensor->gather(*src, dim, *index);
}

static void tensorScatter(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  int dim = unpackInteger(raw_message);
  thpp::Tensor *index = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  tensor->scatter(dim, *index, *src);
}

static void tensorScatterFill(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  int dim = unpackInteger(raw_message);
  thpp::Tensor *index = unpackRetrieveTensor(raw_message);

  if (thpp::isInteger(tensor->type())) {
    long long value = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::IntTensor*>(tensor)->scatterFill(dim, *index, value);
  } else if (thpp::isFloat(tensor->type())) {
    double value = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::FloatTensor*>(tensor)->scatterFill(dim, *index, value);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorDot(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src = unpackRetrieveTensor(raw_message);
  finalize(raw_message);

  if (thpp::isInteger(tensor->type())) {
    long long value = dynamic_cast<thpp::IntTensor*>(tensor)->dot(*src);
    sendValueToMaster(value);
  } else if (thpp::isFloat(tensor->type())) {
    double value = dynamic_cast<thpp::FloatTensor*>(tensor)->dot(*src);
    sendValueToMaster(value);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorMinall(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  finalize(raw_message);

  if (thpp::isInteger(tensor->type())) {
    long long value = dynamic_cast<thpp::IntTensor*>(tensor)->minall();
    sendValueToMaster(value);
  } else if (thpp::isFloat(tensor->type())) {
    double value = dynamic_cast<thpp::FloatTensor*>(tensor)->minall();
    sendValueToMaster(value);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorMaxall(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  finalize(raw_message);

  if (thpp::isInteger(tensor->type())) {
    long long value = dynamic_cast<thpp::IntTensor*>(tensor)->maxall();
    sendValueToMaster(value);
  } else if (thpp::isFloat(tensor->type())) {
    double value = dynamic_cast<thpp::FloatTensor*>(tensor)->maxall();
    sendValueToMaster(value);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorSumall(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  finalize(raw_message);

  if (thpp::isInteger(tensor->type())) {
    long long value = dynamic_cast<thpp::IntTensor*>(tensor)->sumall();
    sendValueToMaster(value);
  } else if (thpp::isFloat(tensor->type())) {
    double value = dynamic_cast<thpp::FloatTensor*>(tensor)->sumall();
    sendValueToMaster(value);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorProdall(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  finalize(raw_message);

  if (thpp::isInteger(tensor->type())) {
    long long value = dynamic_cast<thpp::IntTensor*>(tensor)->prodall();
    sendValueToMaster(value);
  } else if (thpp::isFloat(tensor->type())) {
    double value = dynamic_cast<thpp::FloatTensor*>(tensor)->prodall();
    sendValueToMaster(value);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorAdd(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src = unpackRetrieveTensor(raw_message);

  if (thpp::isInteger(tensor->type())) {
    long long value = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::IntTensor*>(tensor)->add(*src, value);
  } else if (thpp::isFloat(tensor->type())) {
    double value = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::FloatTensor*>(tensor)->add(*src, value);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorSub(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src = unpackRetrieveTensor(raw_message);

  if (thpp::isInteger(tensor->type())) {
    long long value = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::IntTensor*>(tensor)->sub(*src, value);
  } else if (thpp::isFloat(tensor->type())) {
    double value = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::FloatTensor*>(tensor)->sub(*src, value);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorMul(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src = unpackRetrieveTensor(raw_message);

  if (thpp::isInteger(tensor->type())) {
    long long value = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::IntTensor*>(tensor)->mul(*src, value);
  } else if (thpp::isFloat(tensor->type())) {
    double value = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::FloatTensor*>(tensor)->mul(*src, value);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorDiv(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src = unpackRetrieveTensor(raw_message);

  if (thpp::isInteger(tensor->type())) {
    long long value = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::IntTensor*>(tensor)->div(*src, value);
  } else if (thpp::isFloat(tensor->type())) {
    double value = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::FloatTensor*>(tensor)->div(*src, value);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorFmod(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src = unpackRetrieveTensor(raw_message);

  if (thpp::isInteger(tensor->type())) {
    long long value = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::IntTensor*>(tensor)->fmod(*src, value);
  } else {
    double value = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::FloatTensor*>(tensor)->fmod(*src, value);
  }
}

static void tensorRemainder(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src = unpackRetrieveTensor(raw_message);

  if (thpp::isInteger(tensor->type())) {
    long long value = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::IntTensor*>(tensor)->remainder(*src, value);
  } else if (thpp::isFloat(tensor->type())) {
    double value = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::FloatTensor*>(tensor)->remainder(*src, value);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorClamp(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src = unpackRetrieveTensor(raw_message);

  if (thpp::isInteger(tensor->type())) {
    long long min_value = unpackInteger(raw_message);
    long long max_value = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::IntTensor*>(tensor)->clamp(*src, min_value, max_value);
  } else if (thpp::isFloat(tensor->type())) {
    double min_value = unpackFloat(raw_message);
    double max_value = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::FloatTensor*>(tensor)->clamp(*src, min_value, max_value);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorCadd(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src1 = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src2 = unpackRetrieveTensor(raw_message);

  if (thpp::isInteger(tensor->type())) {
    long long value = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::IntTensor*>(tensor)->cadd(*src1, value, *src2);
  } else if (thpp::isFloat(tensor->type())) {
    double value = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::FloatTensor*>(tensor)->cadd(*src1, value, *src2);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorCsub(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src1 = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src2 = unpackRetrieveTensor(raw_message);

  if (thpp::isInteger(tensor->type())) {
    long long value = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::IntTensor*>(tensor)->csub(*src1, value, *src2);
  } else if (thpp::isFloat(tensor->type())) {
    double value = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::FloatTensor*>(tensor)->csub(*src1, value, *src2);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorCmul(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src1 = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src2 = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  tensor->cmul(*src1, *src2);
}

static void tensorCpow(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src1 = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src2 = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  tensor->cpow(*src1, *src2);
}

static void tensorCdiv(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src1 = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src2 = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  tensor->cdiv(*src1, *src2);
}

static void tensorCfmod(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src1 = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src2 = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  tensor->cfmod(*src1, *src2);
}

static void tensorCremainder(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src1 = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src2 = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  tensor->cremainder(*src1, *src2);
}

static void tensorAddcmul(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src1 = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src2 = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src3 = unpackRetrieveTensor(raw_message);

  if (thpp::isInteger(tensor->type())) {
    long long value = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::IntTensor*>(tensor)->addcmul(*src1, value, *src2, *src3);
  } else if (thpp::isFloat(tensor->type())) {
    double value = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::FloatTensor*>(tensor)->addcmul(*src1, value, *src2, *src3);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorAddcdiv(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src1 = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src2 = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src3 = unpackRetrieveTensor(raw_message);

  if (thpp::isInteger(tensor->type())) {
    long long value = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::IntTensor*>(tensor)->addcdiv(*src1, value, *src2, *src3);
  } else if (thpp::isFloat(tensor->type())) {
    double value = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::FloatTensor*>(tensor)->addcdiv(*src1, value, *src2, *src3);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorAddmv(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src = unpackRetrieveTensor(raw_message);
  thpp::Tensor *mat = unpackRetrieveTensor(raw_message);
  thpp::Tensor *vec = unpackRetrieveTensor(raw_message);

  if (thpp::isInteger(tensor->type())) {
    long long beta = unpackInteger(raw_message);
    long long alpha = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::IntTensor*>(tensor)->addmv(beta, *src, alpha, *mat, *vec);
  } else if (thpp::isFloat(tensor->type())) {
    double beta = unpackFloat(raw_message);
    double alpha = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::FloatTensor*>(tensor)->addmv(beta, *src, alpha, *mat, *vec);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorAddmm(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src = unpackRetrieveTensor(raw_message);
  thpp::Tensor *mat1 = unpackRetrieveTensor(raw_message);
  thpp::Tensor *mat2 = unpackRetrieveTensor(raw_message);

  if (thpp::isInteger(tensor->type())) {
    long long beta = unpackInteger(raw_message);
    long long alpha = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::IntTensor*>(tensor)->addmm(beta, *src, alpha, *mat1, *mat2);
  } else if (thpp::isFloat(tensor->type())) {
    double beta = unpackFloat(raw_message);
    double alpha = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::FloatTensor*>(tensor)->addmm(beta, *src, alpha, *mat1, *mat2);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorAddr(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src = unpackRetrieveTensor(raw_message);
  thpp::Tensor *vec1 = unpackRetrieveTensor(raw_message);
  thpp::Tensor *vec2 = unpackRetrieveTensor(raw_message);

  if (thpp::isInteger(tensor->type())) {
    long long beta = unpackInteger(raw_message);
    long long alpha = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::IntTensor*>(tensor)->addr(beta, *src, alpha, *vec1, *vec2);
  } else if (thpp::isFloat(tensor->type())) {
    double beta = unpackFloat(raw_message);
    double alpha = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::FloatTensor*>(tensor)->addr(beta, *src, alpha, *vec1, *vec2);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorAddbmm(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src = unpackRetrieveTensor(raw_message);
  thpp::Tensor *batch1 = unpackRetrieveTensor(raw_message);
  thpp::Tensor *batch2 = unpackRetrieveTensor(raw_message);

  if (thpp::isInteger(tensor->type())) {
    long long beta = unpackInteger(raw_message);
    long long alpha = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::IntTensor*>(tensor)->addbmm(beta, *src, alpha, *batch1, *batch2);
  } else if (thpp::isFloat(tensor->type())) {
    double beta = unpackFloat(raw_message);
    double alpha = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::FloatTensor*>(tensor)->addbmm(beta, *src, alpha, *batch1, *batch2);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorBaddbmm(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src = unpackRetrieveTensor(raw_message);
  thpp::Tensor *batch1 = unpackRetrieveTensor(raw_message);
  thpp::Tensor *batch2 = unpackRetrieveTensor(raw_message);

  if (thpp::isInteger(tensor->type())) {
    long long beta = unpackInteger(raw_message);
    long long alpha = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::IntTensor*>(tensor)->baddbmm(beta, *src, alpha, *batch1, *batch2);
  } else if (thpp::isFloat(tensor->type())) {
    double beta = unpackFloat(raw_message);
    double alpha = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::FloatTensor*>(tensor)->baddbmm(beta, *src, alpha, *batch1, *batch2);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorMatch(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Tensor *m1 = unpackRetrieveTensor(raw_message);
  thpp::Tensor *m2 = unpackRetrieveTensor(raw_message);

  if (thpp::isInteger(tensor->type())) {
    long long gain = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::IntTensor*>(tensor)->match(*m1, *m2, gain);
  } else if (thpp::isFloat(tensor->type())) {
    double gain = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::FloatTensor*>(tensor)->match(*m1, *m2, gain);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorMax(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Tensor *indices_ = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  finalize(raw_message);
  tensor->max(*indices_, *src, dimension);
}

static void tensorMin(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Tensor *indices_ = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  finalize(raw_message);
  tensor->min(*indices_, *src, dimension);
}

static void tensorKthvalue(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Tensor *indices_ = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src = unpackRetrieveTensor(raw_message);
  int k = unpackInteger(raw_message);
  int dimension = unpackInteger(raw_message);
  finalize(raw_message);
  tensor->kthvalue(*indices_, *src, k, dimension);
}

static void tensorMode(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Tensor *indices_ = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  finalize(raw_message);
  tensor->mode(*indices_, *src, dimension);
}

static void tensorMedian(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Tensor *indices_ = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  finalize(raw_message);
  tensor->median(*indices_, *src, dimension);
}

static void tensorSum(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  finalize(raw_message);
  tensor->sum(*src, dimension);
}

static void tensorProd(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  finalize(raw_message);
  tensor->prod(*src, dimension);
}

static void tensorCumsum(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  finalize(raw_message);
  tensor->cumsum(*src, dimension);
}

static void tensorCumprod(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  finalize(raw_message);
  tensor->cumprod(*src, dimension);
}

static void tensorSign(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  tensor->sign(*src);
}

static void tensorTrace(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  finalize(raw_message);

  if (thpp::isInteger(tensor->type())) {
    long long value = dynamic_cast<thpp::IntTensor*>(tensor)->trace();
    sendValueToMaster(value);
  } else if (thpp::isFloat(tensor->type())) {
    double value = dynamic_cast<thpp::FloatTensor*>(tensor)->trace();
    sendValueToMaster(value);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorCross(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src1 = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src2 = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  finalize(raw_message);
  tensor->cross(*src1, *src2, dimension);
}

static void tensorCmax(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src1 = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src2 = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  tensor->cmax(*src1, *src2);
}

static void tensorCmin(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src1 = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src2 = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  tensor->cmin(*src1, *src2);
}

static void tensorCmaxValue(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src = unpackRetrieveTensor(raw_message);

  if (thpp::isInteger(tensor->type())) {
    long long value = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::IntTensor*>(tensor)->cmaxValue(*src, value);
  } else if (thpp::isFloat(tensor->type())) {
    double value = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::FloatTensor*>(tensor)->cmaxValue(*src, value);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorCminValue(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src = unpackRetrieveTensor(raw_message);

  if (thpp::isInteger(tensor->type())) {
    long long value = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::IntTensor*>(tensor)->cminValue(*src, value);
  } else if (thpp::isFloat(tensor->type())) {
    double value = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::FloatTensor*>(tensor)->cminValue(*src, value);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}
