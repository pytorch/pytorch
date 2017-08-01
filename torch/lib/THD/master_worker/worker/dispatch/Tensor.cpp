
template<typename... Ts>
static std::unique_ptr<thpp::Tensor> createTensor(thpp::Type type, Ts &... args) {
  if (type == thpp::Type::UCHAR)
    return std::unique_ptr<thpp::Tensor>(
        new thpp::THTensor<unsigned char>(std::forward<Ts>(args)...));
  else if (type == thpp::Type::CHAR)
    return std::unique_ptr<thpp::Tensor>(
        new thpp::THTensor<int8_t>(std::forward<Ts>(args)...));
  else if (type == thpp::Type::SHORT)
    return std::unique_ptr<thpp::Tensor>(
        new thpp::THTensor<short>(std::forward<Ts>(args)...));
  else if (type == thpp::Type::INT)
    return std::unique_ptr<thpp::Tensor>(
        new thpp::THTensor<int>(std::forward<Ts>(args)...));
  else if (type == thpp::Type::LONG)
    return std::unique_ptr<thpp::Tensor>(
        new thpp::THTensor<int64_t>(std::forward<Ts>(args)...));
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
  at::Storage *storage = unpackRetrieveStorage(raw_message);
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
  at::Tensor self = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  workerTensors.emplace(
    id,
    createTensor(type, self)
  );
}

static void tensorNewClone(rpc::RPCMessage& raw_message) {
  thd::object_id_type id = unpackTensor(raw_message);
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  workerTensors.emplace(
    id,
    tensor.clone()
  );
}

static void tensorResize(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  THLongStorage *size = unpackTHLongStorage(raw_message);
  THLongStorage *stride = unpackTHLongStorage(raw_message);
  finalize(raw_message);
  at::ArrayRef<int64_t> sizeRef(size->data, size->size);
  at::ArrayRef<int64_t> strideRef(stride->data, stride->size);
  tensor.reshape_(sizeRef, strideRef);
  THLongStorage_free(size);
  THLongStorage_free(stride);
}

static void tensorResizeAs(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  tensor.resize_as_(src);
}

static void tensorResizeNd(rpc::RPCMessage& raw_message, std::size_t N) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  std::vector<int64_t> size(N);
  for (std::size_t i = 0; i < N; ++i) {
    size[i] = unpackInteger(raw_message);
  }
  finalize(raw_message);
  tensor.resize_(size);
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
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  tensor.set_(src);
}

static void tensorSetStorage(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Storage *storage = unpackRetrieveStorage(raw_message);
  ptrdiff_t storageOffset = unpackInteger(raw_message);
  THLongStorage *size = unpackTHLongStorage(raw_message);
  THLongStorage *stride = unpackTHLongStorage(raw_message);
  finalize(raw_message);
  at::ArrayRef<int64_t> sizeRef(size->data, size->size);
  at::ArrayRef<int64_t> strideRef(stride->data, stride->size);
  tensor.set_(
      *storage,
      storageOffset,
      sizeRef,
      strideRef
  );
  THLongStorage_free(size);
  THLongStorage_free(stride);
}

static void tensorSetStorage1d(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Storage *storage = unpackRetrieveStorage(raw_message);
  ptrdiff_t storageOffset = unpackInteger(raw_message);
  int64_t size0 = unpackInteger(raw_message);
  int64_t stride0 = unpackInteger(raw_message);
  finalize(raw_message);
  at::ArrayRef<int64_t> sizes(size0);
  at::ArrayRef<int64_t> strides(stride0);
  tensor.set_(
    *storage,
    storageOffset,
    sizes,
    strides
  );
}

static void tensorSetStorage2d(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Storage *storage = unpackRetrieveStorage(raw_message);
  ptrdiff_t storageOffset = unpackInteger(raw_message);
  int64_t size0 = unpackInteger(raw_message);
  int64_t stride0 = unpackInteger(raw_message);
  int64_t size1 = unpackInteger(raw_message);
  int64_t stride1 = unpackInteger(raw_message);
  finalize(raw_message);
  THLongStorage *sizes = THLongStorage_newWithSize2(size0, size1);
  THLongStorage *strides = THLongStorage_newWithSize2(stride0, stride1);
  at::ArrayRef<int64_t> sizeRef(sizes->data, sizes->size);
  at::ArrayRef<int64_t> strideRef(strides->data, strides->size);
  tensor.set_(
    *storage,
    storageOffset,
    sizeRef,
    strideRef
  );
  THLongStorage_free(sizes);
  THLongStorage_free(strides);
}

static void tensorSetStorage3d(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Storage *storage = unpackRetrieveStorage(raw_message);
  ptrdiff_t storageOffset = unpackInteger(raw_message);
  int64_t size0 = unpackInteger(raw_message);
  int64_t stride0 = unpackInteger(raw_message);
  int64_t size1 = unpackInteger(raw_message);
  int64_t stride1 = unpackInteger(raw_message);
  int64_t size2 = unpackInteger(raw_message);
  int64_t stride2 = unpackInteger(raw_message);
  finalize(raw_message);
  THLongStorage *sizes = THLongStorage_newWithSize3(size0, size1, size2);
  THLongStorage *strides = THLongStorage_newWithSize3(stride0, stride1, stride2);
  at::ArrayRef<int64_t> sizeRef(sizes->data, sizes->size);
  at::ArrayRef<int64_t> strideRef(strides->data, strides->size);
  tensor.set_(
    *storage,
    storageOffset,
    sizeRef,
    strideRef
  );
  THLongStorage_free(sizes);
  THLongStorage_free(strides);
}

static void tensorSetStorage4d(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Storage *storage = unpackRetrieveStorage(raw_message);
  ptrdiff_t storageOffset = unpackInteger(raw_message);
  int64_t size0 = unpackInteger(raw_message);
  int64_t stride0 = unpackInteger(raw_message);
  int64_t size1 = unpackInteger(raw_message);
  int64_t stride1 = unpackInteger(raw_message);
  int64_t size2 = unpackInteger(raw_message);
  int64_t stride2 = unpackInteger(raw_message);
  int64_t size3 = unpackInteger(raw_message);
  int64_t stride3 = unpackInteger(raw_message);
  finalize(raw_message);
  THLongStorage *sizes = THLongStorage_newWithSize4(size0, size1, size2, size3);
  THLongStorage *strides = THLongStorage_newWithSize4(stride0, stride1,
                                                      stride2, stride3);
  at::ArrayRef<int64_t> sizeRef(sizes->data, sizes->size);
  at::ArrayRef<int64_t> strideRef(strides->data, strides->size);
  tensor.set_(
    *storage,
    storageOffset,
    sizeRef,
    strideRef
  );
  THLongStorage_free(sizes);
  THLongStorage_free(strides);
}

static void tensorNarrow(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  int64_t firstIndex = unpackInteger(raw_message);
  int64_t size = unpackInteger(raw_message);
  finalize(raw_message);
  tensor = src.narrow(dimension, firstIndex, size);
}

static void tensorSelect(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  int64_t sliceIndex = unpackInteger(raw_message);
  finalize(raw_message);
  at::select_out(src, dimension, sliceIndex, tensor);
}

static void tensorTranspose(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src = unpackRetrieveTensor(raw_message);
  int dimension1 = unpackInteger(raw_message);
  int dimension2 = unpackInteger(raw_message);
  finalize(raw_message);
  tensor = src.transpose(dimension1, dimension2);
}

static void tensorUnfold(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  int64_t size = unpackInteger(raw_message);
  int64_t step = unpackInteger(raw_message);
  finalize(raw_message);
  tensor = src.unfold(dimension, size, step);
}

static void tensorSqueeze(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  at::squeeze_out(src, tensor);
}

static void tensorSqueeze1d(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  finalize(raw_message);
  at::squeeze_out(src, dimension, tensor);
}

static void tensorFree(rpc::RPCMessage& raw_message) {
  object_id_type tensor_id = unpackTensor(raw_message);
  (void)workerTensors.erase(tensor_id);
}

static void tensorGather(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src = unpackRetrieveTensor(raw_message);
  int dim = unpackInteger(raw_message);
  at::Tensor index = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  at::gather_out(src, dim, index, tensor);
}

static void tensorScatter(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  int dim = unpackInteger(raw_message);
  at::Tensor index = unpackRetrieveTensor(raw_message);
  at::Tensor src = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  tensor.scatter_(dim, index, src);
}

static void tensorScatterFill(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  int dim = unpackInteger(raw_message);
  at::Tensor index = unpackRetrieveTensor(raw_message);

  if (at::isIntegralType(tensor.type().scalarType())) {
    int64_t value = unpackInteger(raw_message);
    finalize(raw_message);
    tensor.scatter_(dim, index, value);
  } else if (at::isFloatingType(tensor.type().scalarType())) {
    double value = unpackFloat(raw_message);
    finalize(raw_message);
    tensor.scatter_(dim, index, value);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorDot(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src = unpackRetrieveTensor(raw_message);
  finalize(raw_message);

  if (at::isIntegralType(tensor.type().scalarType())) {
    int64_t value = tensor.dot(src).toLong();
    sendValueToMaster(value);
  } else if (at::isFloatingType(tensor.type().scalarType())) {
    double value = tensor.dot(src).toDouble();
    sendValueToMaster(value);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorMinall(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  finalize(raw_message);

  if (at::isIntegralType(tensor.type().scalarType())) {
    int64_t value = tensor.min().toLong();
    sendValueToMaster(value);
  } else if (at::isFloatingType(tensor.type().scalarType())) {
    double value = tensor.min().toDouble();
    sendValueToMaster(value);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorMaxall(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  finalize(raw_message);

  if (at::isIntegralType(tensor.type().scalarType())) {
    int64_t value = tensor.max().toLong();
    sendValueToMaster(value);
  } else if (at::isFloatingType(tensor.type().scalarType())) {
    double value = tensor.max().toDouble();
    sendValueToMaster(value);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorMedianall(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  finalize(raw_message);

  if (at::isIntegralType(tensor.type().scalarType())) {
    int64_t value = tensor.median().toLong();
    sendValueToMaster(value);
  } else if (at::isFloatingType(tensor.type().scalarType())) {
    double value = tensor.median().toDouble();
    sendValueToMaster(value);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorSumall(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  finalize(raw_message);

  if (at::isIntegralType(tensor.type().scalarType())) {
    int64_t value = tensor.sum().toLong();
    sendValueToMaster(value);
  } else if (at::isFloatingType(tensor.type().scalarType())) {
    double value = tensor.sum().toDouble();
    sendValueToMaster(value);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorProdall(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  finalize(raw_message);

  if (at::isIntegralType(tensor.type().scalarType())) {
    int64_t value = tensor.prod().toLong();
    sendValueToMaster(value);
  } else if (at::isFloatingType(tensor.type().scalarType())) {
    double value = tensor.prod().toDouble();
    sendValueToMaster(value);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorAdd(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src = unpackRetrieveTensor(raw_message);

  if (at::isIntegralType(tensor.type().scalarType())) {
    int64_t value = unpackInteger(raw_message);
    finalize(raw_message);
    at::add_out(src, at::Scalar((int64_t)value), tensor);
  } else if (at::isFloatingType(tensor.type().scalarType())) {
    double value = unpackFloat(raw_message);
    finalize(raw_message);
    at::add_out(src, value, tensor);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorSub(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src = unpackRetrieveTensor(raw_message);

  if (at::isIntegralType(tensor.type().scalarType())) {
    int64_t value = unpackInteger(raw_message);
    finalize(raw_message);
    at::sub_out(src, at::Scalar((int64_t)value), tensor);
  } else if (at::isFloatingType(tensor.type().scalarType())) {
    double value = unpackFloat(raw_message);
    finalize(raw_message);
    at::sub_out(src, value, tensor);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorMul(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src = unpackRetrieveTensor(raw_message);

  if (at::isIntegralType(tensor.type().scalarType())) {
    int64_t value = unpackInteger(raw_message);
    finalize(raw_message);
    at::mul_out(src, at::Scalar((int64_t)value), tensor);
  } else if (at::isFloatingType(tensor.type().scalarType())) {
    double value = unpackFloat(raw_message);
    finalize(raw_message);
    at::mul_out(src, value, tensor);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorDiv(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src = unpackRetrieveTensor(raw_message);

  if (at::isIntegralType(tensor.type().scalarType())) {
    int64_t value = unpackInteger(raw_message);
    finalize(raw_message);
    at::div_out(src, at::Scalar((int64_t)value), tensor);
  } else if (at::isFloatingType(tensor.type().scalarType())) {
    double value = unpackFloat(raw_message);
    finalize(raw_message);
    at::div_out(src, value, tensor);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorFmod(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src = unpackRetrieveTensor(raw_message);

  if (at::isIntegralType(tensor.type().scalarType())) {
    int64_t value = unpackInteger(raw_message);
    finalize(raw_message);
    at::fmod_out(src, at::Scalar((int64_t)value), tensor);
  } else {
    double value = unpackFloat(raw_message);
    finalize(raw_message);
    at::fmod_out(src, value, tensor);
  }
}

static void tensorRemainder(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src = unpackRetrieveTensor(raw_message);

  if (at::isIntegralType(tensor.type().scalarType())) {
    int64_t value = unpackInteger(raw_message);
    finalize(raw_message);
    at::remainder_out(src, at::Scalar((int64_t)value), tensor);
  } else if (at::isFloatingType(tensor.type().scalarType())) {
    double value = unpackFloat(raw_message);
    finalize(raw_message);
    at::remainder_out(src, value, tensor);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorClamp(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src = unpackRetrieveTensor(raw_message);

  if (at::isIntegralType(tensor.type().scalarType())) {
    int64_t min_value = unpackInteger(raw_message);
    int64_t max_value = unpackInteger(raw_message);
    finalize(raw_message);
    at::clamp_out(src, at::Scalar((int64_t)min_value), at::Scalar((int64_t)max_value), tensor);
  } else if (at::isFloatingType(tensor.type().scalarType())) {
    double min_value = unpackFloat(raw_message);
    double max_value = unpackFloat(raw_message);
    finalize(raw_message);
    at::clamp_out(src, min_value, max_value, tensor);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorCadd(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src1 = unpackRetrieveTensor(raw_message);
  at::Tensor src2 = unpackRetrieveTensor(raw_message);

  if (at::isIntegralType(tensor.type().scalarType())) {
    int64_t value = unpackInteger(raw_message);
    finalize(raw_message);
    at::add_out(src1, at::Scalar((int64_t)value), src2, tensor);
  } else if (at::isFloatingType(tensor.type().scalarType())) {
    double value = unpackFloat(raw_message);
    finalize(raw_message);
    at::add_out(src1, value, src2, tensor);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorCsub(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src1 = unpackRetrieveTensor(raw_message);
  at::Tensor src2 = unpackRetrieveTensor(raw_message);

  if (at::isIntegralType(tensor.type().scalarType())) {
    int64_t value = unpackInteger(raw_message);
    finalize(raw_message);
    at::sub_out(src1, at::Scalar((int64_t)value), src2, tensor);
  } else if (at::isFloatingType(tensor.type().scalarType())) {
    double value = unpackFloat(raw_message);
    finalize(raw_message);
    at::sub_out(src1, value, src2, tensor);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorCmul(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src1 = unpackRetrieveTensor(raw_message);
  at::Tensor src2 = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  at::mul_out(src1, src2, tensor);
}

static void tensorCpow(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src1 = unpackRetrieveTensor(raw_message);
  at::Tensor src2 = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  at::pow_out(src1, src2, tensor);
}

static void tensorCdiv(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src1 = unpackRetrieveTensor(raw_message);
  at::Tensor src2 = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  at::div_out(src1, src2, tensor);
}

static void tensorCfmod(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src1 = unpackRetrieveTensor(raw_message);
  at::Tensor src2 = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  at::fmod_out(src1, src2, tensor);
}

static void tensorCremainder(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src1 = unpackRetrieveTensor(raw_message);
  at::Tensor src2 = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  at::remainder_out(src1, src2, tensor);
}

static void tensorAddcmul(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src1 = unpackRetrieveTensor(raw_message);
  at::Tensor src2 = unpackRetrieveTensor(raw_message);
  at::Tensor src3 = unpackRetrieveTensor(raw_message);

  if (at::isIntegralType(tensor.type().scalarType())) {
    int64_t value = unpackInteger(raw_message);
    finalize(raw_message);
    at::addcmul_out(src1, at::Scalar((int64_t)value), src2, src3, tensor);
  } else if (at::isFloatingType(tensor.type().scalarType())) {
    double value = unpackFloat(raw_message);
    finalize(raw_message);
    at::addcmul_out(src1, value, src2, src3, tensor);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorAddcdiv(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src1 = unpackRetrieveTensor(raw_message);
  at::Tensor src2 = unpackRetrieveTensor(raw_message);
  at::Tensor src3 = unpackRetrieveTensor(raw_message);

  if (at::isIntegralType(tensor.type().scalarType())) {
    int64_t value = unpackInteger(raw_message);
    finalize(raw_message);
    at::addcdiv_out(src1, at::Scalar((int64_t)value), src2, src3, tensor);
  } else if (at::isFloatingType(tensor.type().scalarType())) {
    double value = unpackFloat(raw_message);
    finalize(raw_message);
    at::addcdiv_out(src1, value, src2, src3, tensor);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorAddmv(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src = unpackRetrieveTensor(raw_message);
  at::Tensor mat = unpackRetrieveTensor(raw_message);
  at::Tensor vec = unpackRetrieveTensor(raw_message);

  if (at::isIntegralType(tensor.type().scalarType())) {
    int64_t beta = unpackInteger(raw_message);
    int64_t alpha = unpackInteger(raw_message);
    finalize(raw_message);
    at::addmv_out(at::Scalar((int64_t)beta), src, at::Scalar((int64_t)alpha), mat, vec, tensor);
  } else if (at::isFloatingType(tensor.type().scalarType())) {
    double beta = unpackFloat(raw_message);
    double alpha = unpackFloat(raw_message);
    finalize(raw_message);
    at::addmv_out(beta, src, alpha, mat, vec, tensor);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorAddmm(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src = unpackRetrieveTensor(raw_message);
  at::Tensor mat1 = unpackRetrieveTensor(raw_message);
  at::Tensor mat2 = unpackRetrieveTensor(raw_message);

  if (at::isIntegralType(tensor.type().scalarType())) {
    int64_t beta = unpackInteger(raw_message);
    int64_t alpha = unpackInteger(raw_message);
    finalize(raw_message);
    at::addmm_out(at::Scalar((int64_t)beta), src, at::Scalar((int64_t)alpha), mat1, mat2, tensor);
  } else if (at::isFloatingType(tensor.type().scalarType())) {
    double beta = unpackFloat(raw_message);
    double alpha = unpackFloat(raw_message);
    finalize(raw_message);
    at::addmm_out(beta, src, alpha, mat1, mat2, tensor);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorAddr(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src = unpackRetrieveTensor(raw_message);
  at::Tensor vec1 = unpackRetrieveTensor(raw_message);
  at::Tensor vec2 = unpackRetrieveTensor(raw_message);

  if (at::isIntegralType(tensor.type().scalarType())) {
    int64_t beta = unpackInteger(raw_message);
    int64_t alpha = unpackInteger(raw_message);
    finalize(raw_message);
    at::addr_out(at::Scalar((int64_t)beta), src, at::Scalar((int64_t)alpha), vec1, vec2, tensor);
  } else if (at::isFloatingType(tensor.type().scalarType())) {
    double beta = unpackFloat(raw_message);
    double alpha = unpackFloat(raw_message);
    finalize(raw_message);
    at::addr_out(beta, src, alpha, vec1, vec2, tensor);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorAddbmm(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src = unpackRetrieveTensor(raw_message);
  at::Tensor batch1 = unpackRetrieveTensor(raw_message);
  at::Tensor batch2 = unpackRetrieveTensor(raw_message);

  if (at::isIntegralType(tensor.type().scalarType())) {
    int64_t beta = unpackInteger(raw_message);
    int64_t alpha = unpackInteger(raw_message);
    finalize(raw_message);
    at::addbmm_out(at::Scalar((int64_t)beta), src, at::Scalar((int64_t)alpha), batch1, batch2, tensor);
  } else if (at::isFloatingType(tensor.type().scalarType())) {
    double beta = unpackFloat(raw_message);
    double alpha = unpackFloat(raw_message);
    finalize(raw_message);
    at::addbmm_out(beta, src, alpha, batch1, batch2, tensor);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorBaddbmm(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src = unpackRetrieveTensor(raw_message);
  at::Tensor batch1 = unpackRetrieveTensor(raw_message);
  at::Tensor batch2 = unpackRetrieveTensor(raw_message);

  if (at::isIntegralType(tensor.type().scalarType())) {
    int64_t beta = unpackInteger(raw_message);
    int64_t alpha = unpackInteger(raw_message);
    finalize(raw_message);
    at::baddbmm_out(at::Scalar((int64_t)beta), src, at::Scalar((int64_t)alpha), batch1, batch2, tensor);
  } else if (at::isFloatingType(tensor.type().scalarType())) {
    double beta = unpackFloat(raw_message);
    double alpha = unpackFloat(raw_message);
    finalize(raw_message);
    at::baddbmm_out(beta, src, alpha, batch1, batch2, tensor);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorMatch(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Tensor *m1 = unpackRetrieveTensor(raw_message);
  thpp::Tensor *m2 = unpackRetrieveTensor(raw_message);

  if (thpp::isInteger(tensor->type())) {
    int64_t gain = unpackInteger(raw_message);
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
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor indices_ = unpackRetrieveTensor(raw_message);
  at::Tensor src = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  int keepdim = unpackInteger(raw_message);
  finalize(raw_message);
  at::max_out(src, dimension, keepdim, tensor, indices_);
}

static void tensorMin(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor indices_ = unpackRetrieveTensor(raw_message);
  at::Tensor src = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  int keepdim = unpackInteger(raw_message);
  finalize(raw_message);
  at::min_out(src, dimension, keepdim, tensor, indices_);
}

static void tensorKthvalue(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor indices_ = unpackRetrieveTensor(raw_message);
  at::Tensor src = unpackRetrieveTensor(raw_message);
  int k = unpackInteger(raw_message);
  int dimension = unpackInteger(raw_message);
  int keepdim = unpackInteger(raw_message);
  finalize(raw_message);
  at::kthvalue_out(src, k, dimension, keepdim, tensor, indices_);
}

static void tensorMode(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor indices_ = unpackRetrieveTensor(raw_message);
  at::Tensor src = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  int keepdim = unpackInteger(raw_message);
  finalize(raw_message);
  at::mode_out(src, dimension, keepdim, tensor, indices_);
}

static void tensorMedian(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor indices_ = unpackRetrieveTensor(raw_message);
  at::Tensor src = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  int keepdim = unpackInteger(raw_message);
  finalize(raw_message);
  at::median_out(src, dimension, keepdim, tensor, indices_);
}

static void tensorSum(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  int keepdim = unpackInteger(raw_message);
  finalize(raw_message);
  at::sum_out(src, dimension, keepdim, tensor);
}

static void tensorProd(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  int keepdim = unpackInteger(raw_message);
  finalize(raw_message);
  at::prod_out(src, dimension, keepdim, tensor);
}

static void tensorCumsum(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  finalize(raw_message);
  at::cumsum_out(src, dimension, tensor);
}

static void tensorCumprod(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  finalize(raw_message);
  at::cumprod_out(src, dimension, tensor);
}

static void tensorSign(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  at::sign_out(src, tensor);
}

static void tensorTrace(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  finalize(raw_message);

  if (at::isIntegralType(tensor.type().scalarType())) {
    int64_t value = tensor.trace().toLong();
    sendValueToMaster(value);
  } else if (at::isFloatingType(tensor.type().scalarType())) {
    double value = tensor.trace().toDouble();
    sendValueToMaster(value);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorCross(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src1 = unpackRetrieveTensor(raw_message);
  at::Tensor src2 = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  finalize(raw_message);
  at::cross_out(src1, src2, dimension, tensor);
}

static void tensorCmax(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src1 = unpackRetrieveTensor(raw_message);
  at::Tensor src2 = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  at::max_out(src1, src2, tensor);
}

static void tensorCmin(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src1 = unpackRetrieveTensor(raw_message);
  at::Tensor src2 = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  at::min_out(src1, src2, tensor);
}

/* static void tensorCmaxValue(rpc::RPCMessage& raw_message) { */
/*   at::Tensor tensor = unpackRetrieveTensor(raw_message); */
/*   at::Tensor src = unpackRetrieveTensor(raw_message); */

/*   if (at::isIntegralType(tensor.type().scalarType())) { */
/*     int64_t value = unpackInteger(raw_message); */
/*     finalize(raw_message); */
/*     at::clamp_out(src, at::Scalar((int64_t)value), tensor); */
/*   } else if (at::isFloatingType(tensor.type().scalarType())) { */
/*     double value = unpackFloat(raw_message); */
/*     finalize(raw_message); */
/*     at::clamp_out(src, value, tensor); */
/*   } else { */
/*     throw std::invalid_argument("expected scalar type"); */
/*   } */
/* } */

/* static void tensorCminValue(rpc::RPCMessage& raw_message) { */
/*   at::Tensor tensor = unpackRetrieveTensor(raw_message); */
/*   at::Tensor src = unpackRetrieveTensor(raw_message); */

/*   if (at::isIntegralType(tensor.type().scalarType())) { */
/*     int64_t value = unpackInteger(raw_message); */
/*     finalize(raw_message); */
/*     dynamic_cast<thpp::IntTensor*>(tensor)->cminValue(*src, value); */
/*   } else if (at::isFloatingType(tensor.type().scalarType())) { */
/*     double value = unpackFloat(raw_message); */
/*     finalize(raw_message); */
/*     dynamic_cast<thpp::FloatTensor*>(tensor)->cminValue(*src, value); */
/*   } else { */
/*     throw std::invalid_argument("expected scalar type"); */
/*   } */
/* } */
