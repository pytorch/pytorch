template<typename... Ts>
static at::Tensor createTensor(RPCType type, Ts &... args) {
  if (type == RPCType::UCHAR)
    return at::CPU(at::kByte).tensor(std::forward<Ts>(args)...);
  else if (type == RPCType::CHAR)
    return at::CPU(at::kChar).tensor(std::forward<Ts>(args)...);
  else if (type == RPCType::SHORT)
    return at::CPU(at::kShort).tensor(std::forward<Ts>(args)...);
  else if (type == RPCType::INT)
    return at::CPU(at::kInt).tensor(std::forward<Ts>(args)...);
  else if (type == RPCType::LONG)
    return at::CPU(at::kLong).tensor(std::forward<Ts>(args)...);
  else if (type == RPCType::FLOAT)
    return at::CPU(at::kFloat).tensor(std::forward<Ts>(args)...);
  else if (type == RPCType::DOUBLE)
    return at::CPU(at::kDouble).tensor(std::forward<Ts>(args)...);
  throw std::invalid_argument("passed character doesn't represent a tensor type");
}

static at::Tensor createTensorWithStorage(RPCType type, at::Storage* storage, ptrdiff_t storageOffset, at::IntList size, at::IntList stride) {
  if (type == RPCType::UCHAR)
    return at::CPU(at::kByte).tensor(*storage, storageOffset, size, stride);
  else if (type == RPCType::CHAR)
    return at::CPU(at::kChar).tensor(*storage, storageOffset, size, stride);
  else if (type == RPCType::SHORT)
    return at::CPU(at::kShort).tensor(*storage, storageOffset, size, stride);
  else if (type == RPCType::INT)
    return at::CPU(at::kInt).tensor(*storage, storageOffset, size, stride);
  else if (type == RPCType::LONG)
    return at::CPU(at::kLong).tensor(*storage, storageOffset, size, stride);
  else if (type == RPCType::FLOAT)
    return at::CPU(at::kFloat).tensor(*storage, storageOffset, size, stride);
  else if (type == RPCType::DOUBLE)
    return at::CPU(at::kDouble).tensor(*storage, storageOffset, size, stride);
  throw std::invalid_argument("passed character doesn't represent a tensor type");
}

static at::Tensor createTensorWithTensor(RPCType type, at::Tensor& tensor) {
  if (type == RPCType::UCHAR)
    return at::CPU(at::kByte).alias(tensor);
  else if (type == RPCType::CHAR)
    return at::CPU(at::kChar).alias(tensor);
  else if (type == RPCType::SHORT)
    return at::CPU(at::kShort).alias(tensor);
  else if (type == RPCType::INT)
    return at::CPU(at::kInt).alias(tensor);
  else if (type == RPCType::LONG)
    return at::CPU(at::kLong).alias(tensor);
  else if (type == RPCType::FLOAT)
    return at::CPU(at::kFloat).alias(tensor);
  else if (type == RPCType::DOUBLE)
    return at::CPU(at::kDouble).alias(tensor);
  throw std::invalid_argument("passed character doesn't represent a tensor type");
}

static void tensorNew(rpc::RPCMessage& raw_message) {
  RPCType type = unpackType(raw_message);
  thd::object_id_type id = unpackTensor(raw_message);
  finalize(raw_message);
  workerTensors.emplace(
    id,
    createTensor(type)
  );
}

static void tensorNewWithSize(rpc::RPCMessage& raw_message) {
  RPCType type = unpackType(raw_message);
  thd::object_id_type id = unpackTensor(raw_message);
  THLongStorage *size = unpackTHLongStorage(raw_message);
  THLongStorage *stride = unpackTHLongStorage(raw_message);
  finalize(raw_message);

  at::IntList sz(THLongStorage_data(size), THLongStorage_size(size));
  at::IntList str(THLongStorage_data(stride), THLongStorage_size(stride));
  workerTensors.emplace(
    id,
    createTensor(type, sz, str)
  );
  THLongStorage_free(size);
  THLongStorage_free(stride);
}

static void tensorNewWithStorage(rpc::RPCMessage& raw_message) {
  RPCType type = unpackType(raw_message);
  thd::object_id_type id = unpackTensor(raw_message);
  at::Storage *storage = unpackRetrieveStorage(raw_message);
  ptrdiff_t storageOffset = unpackInteger(raw_message);
  THLongStorage *size = unpackTHLongStorage(raw_message);
  THLongStorage *stride = unpackTHLongStorage(raw_message);
  finalize(raw_message);

  at::IntList sz(THLongStorage_data(size), THLongStorage_size(size));
  at::IntList str(THLongStorage_data(stride), THLongStorage_size(stride));
  workerTensors.emplace(
    id,
    createTensorWithStorage(type, storage, storageOffset, sz, str)
  );
  THLongStorage_free(size);
  THLongStorage_free(stride);
}

static void tensorNewWithTensor(rpc::RPCMessage& raw_message) {
  RPCType type = unpackType(raw_message);
  thd::object_id_type id = unpackTensor(raw_message);
  at::Tensor self = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  workerTensors.emplace(
    id,
    createTensorWithTensor(type, self)
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
  finalize(raw_message);
  at::ArrayRef<int64_t> sizeRef(THLongStorage_data(size), THLongStorage_size(size));
  tensor.resize_(sizeRef);
  THLongStorage_free(size);
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
  at::ArrayRef<int64_t> sizeRef(THLongStorage_data(size), THLongStorage_size(size));
  at::ArrayRef<int64_t> strideRef(THLongStorage_data(stride), THLongStorage_size(stride));
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
  at::ArrayRef<int64_t> sizeRef(THLongStorage_data(sizes), THLongStorage_size(sizes));
  at::ArrayRef<int64_t> strideRef(THLongStorage_data(strides), THLongStorage_size(strides));
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
  at::ArrayRef<int64_t> sizeRef(THLongStorage_data(sizes), THLongStorage_size(sizes));
  at::ArrayRef<int64_t> strideRef(THLongStorage_data(strides), THLongStorage_size(strides));
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
  at::ArrayRef<int64_t> sizeRef(THLongStorage_data(sizes), THLongStorage_size(sizes));
  at::ArrayRef<int64_t> strideRef(THLongStorage_data(strides), THLongStorage_size(strides));
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
  tensor = src.select(dimension, sliceIndex);
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
  // FIXME: could be at::squeeze_out(tensor, src), but we don't generate
  // _out functions for native ATen functions (and may not want to).
   tensor.set_(src.squeeze());
}

static void tensorSqueeze1d(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  finalize(raw_message);
  // FIXME: could be at::squeeze_out(tensor, src, dimension), but we don't generate
  // _out functions for native ATen functions (and may not want to).
  tensor.set_(src.squeeze(dimension));
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
  at::gather_out(tensor, src, dim, index);
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
    int64_t value = tensor.dot(src).toCLong();
    sendValueToMaster(value);
  } else if (at::isFloatingType(tensor.type().scalarType())) {
    double value = tensor.dot(src).toCDouble();
    sendValueToMaster(value);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorMinall(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  finalize(raw_message);

  if (at::isIntegralType(tensor.type().scalarType())) {
    int64_t value = tensor.min().toCLong();
    sendValueToMaster(value);
  } else if (at::isFloatingType(tensor.type().scalarType())) {
    double value = tensor.min().toCDouble();
    sendValueToMaster(value);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorMaxall(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  finalize(raw_message);

  if (at::isIntegralType(tensor.type().scalarType())) {
    int64_t value = tensor.max().toCLong();
    sendValueToMaster(value);
  } else if (at::isFloatingType(tensor.type().scalarType())) {
    double value = tensor.max().toCDouble();
    sendValueToMaster(value);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorMedianall(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  finalize(raw_message);

  if (at::isIntegralType(tensor.type().scalarType())) {
    int64_t value = tensor.median().toCLong();
    sendValueToMaster(value);
  } else if (at::isFloatingType(tensor.type().scalarType())) {
    double value = tensor.median().toCDouble();
    sendValueToMaster(value);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorSumall(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  finalize(raw_message);

  if (at::isIntegralType(tensor.type().scalarType())) {
    int64_t value = tensor.sum().toCLong();
    sendValueToMaster(value);
  } else if (at::isFloatingType(tensor.type().scalarType())) {
    double value = tensor.sum().toCDouble();
    sendValueToMaster(value);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorProdall(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  finalize(raw_message);

  if (at::isIntegralType(tensor.type().scalarType())) {
    int64_t value = tensor.prod().toCLong();
    sendValueToMaster(value);
  } else if (at::isFloatingType(tensor.type().scalarType())) {
    double value = tensor.prod().toCDouble();
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
    at::add_out(tensor, src, at::Scalar(value));
  } else if (at::isFloatingType(tensor.type().scalarType())) {
    double value = unpackFloat(raw_message);
    finalize(raw_message);
    at::add_out(tensor, src, value);
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
    at::sub_out(tensor, src, at::Scalar((int64_t)value));
  } else if (at::isFloatingType(tensor.type().scalarType())) {
    double value = unpackFloat(raw_message);
    finalize(raw_message);
    at::sub_out(tensor, src, value);
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
    at::mul_out(tensor, src, at::Scalar((int64_t)value));
  } else if (at::isFloatingType(tensor.type().scalarType())) {
    double value = unpackFloat(raw_message);
    finalize(raw_message);
    at::mul_out(tensor, src, value);
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
    at::div_out(tensor, src, at::Scalar((int64_t)value));
  } else if (at::isFloatingType(tensor.type().scalarType())) {
    double value = unpackFloat(raw_message);
    finalize(raw_message);
    at::div_out(tensor, src, value);
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
    at::fmod_out(tensor, src, at::Scalar((int64_t)value));
  } else {
    double value = unpackFloat(raw_message);
    finalize(raw_message);
    at::fmod_out(tensor, src, value);
  }
}

static void tensorRemainder(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src = unpackRetrieveTensor(raw_message);

  if (at::isIntegralType(tensor.type().scalarType())) {
    int64_t value = unpackInteger(raw_message);
    finalize(raw_message);
    at::remainder_out(tensor, src, at::Scalar((int64_t)value));
  } else if (at::isFloatingType(tensor.type().scalarType())) {
    double value = unpackFloat(raw_message);
    finalize(raw_message);
    at::remainder_out(tensor, src, value);
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
    at::clamp_out(tensor, src, at::Scalar((int64_t)min_value), at::Scalar((int64_t)max_value));
  } else if (at::isFloatingType(tensor.type().scalarType())) {
    double min_value = unpackFloat(raw_message);
    double max_value = unpackFloat(raw_message);
    finalize(raw_message);
    at::clamp_out(tensor, src, min_value, max_value);
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
    at::add_out(tensor, src1, src2, at::Scalar((int64_t)value));
  } else if (at::isFloatingType(tensor.type().scalarType())) {
    double value = unpackFloat(raw_message);
    finalize(raw_message);
    at::add_out(tensor, src1, src2, value);
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
    at::sub_out(tensor, src1, src2, at::Scalar((int64_t)value));
  } else if (at::isFloatingType(tensor.type().scalarType())) {
    double value = unpackFloat(raw_message);
    finalize(raw_message);
    at::sub_out(tensor, src1, src2, value);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorCmul(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src1 = unpackRetrieveTensor(raw_message);
  at::Tensor src2 = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  at::mul_out(tensor, src1, src2);
}

static void tensorCpow(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src1 = unpackRetrieveTensor(raw_message);
  at::Tensor src2 = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  at::pow_out(tensor, src1, src2);
}

static void tensorCdiv(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src1 = unpackRetrieveTensor(raw_message);
  at::Tensor src2 = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  at::div_out(tensor, src1, src2);
}

static void tensorCfmod(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src1 = unpackRetrieveTensor(raw_message);
  at::Tensor src2 = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  at::fmod_out(tensor, src1, src2);
}

static void tensorCremainder(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src1 = unpackRetrieveTensor(raw_message);
  at::Tensor src2 = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  at::remainder_out(tensor, src1, src2);
}

static void tensorAddcmul(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src1 = unpackRetrieveTensor(raw_message);
  at::Tensor src2 = unpackRetrieveTensor(raw_message);
  at::Tensor src3 = unpackRetrieveTensor(raw_message);

  if (at::isIntegralType(tensor.type().scalarType())) {
    int64_t value = unpackInteger(raw_message);
    finalize(raw_message);
    at::addcmul_out(tensor, src1, src2, src3, at::Scalar((int64_t)value));
  } else if (at::isFloatingType(tensor.type().scalarType())) {
    double value = unpackFloat(raw_message);
    finalize(raw_message);
    at::addcmul_out(tensor, src1, src2, src3, value);
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
    at::addcdiv_out(tensor, src1, src2, src3, at::Scalar((int64_t)value));
  } else if (at::isFloatingType(tensor.type().scalarType())) {
    double value = unpackFloat(raw_message);
    finalize(raw_message);
    at::addcdiv_out(tensor, src1, src2, src3, value);
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
    at::addmv_out(tensor, src, mat, vec, at::Scalar((int64_t)beta), at::Scalar((int64_t)alpha));
  } else if (at::isFloatingType(tensor.type().scalarType())) {
    double beta = unpackFloat(raw_message);
    double alpha = unpackFloat(raw_message);
    finalize(raw_message);
    at::addmv_out(tensor, src, mat, vec, beta, alpha);
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
    at::addmm_out(tensor, src, mat1, mat2, at::Scalar((int64_t)beta), at::Scalar((int64_t)alpha));
  } else if (at::isFloatingType(tensor.type().scalarType())) {
    double beta = unpackFloat(raw_message);
    double alpha = unpackFloat(raw_message);
    finalize(raw_message);
    at::addmm_out(tensor, src, mat1, mat2, beta, alpha);
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
    at::addr_out(tensor, src, vec1, vec2,at::Scalar((int64_t)beta), at::Scalar((int64_t)alpha));
  } else if (at::isFloatingType(tensor.type().scalarType())) {
    double beta = unpackFloat(raw_message);
    double alpha = unpackFloat(raw_message);
    finalize(raw_message);
    at::addr_out(tensor, src, vec1, vec2, beta, alpha);
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
    at::addbmm_out(tensor, src, batch1, batch2, at::Scalar((int64_t)beta), at::Scalar((int64_t)alpha));
  } else if (at::isFloatingType(tensor.type().scalarType())) {
    double beta = unpackFloat(raw_message);
    double alpha = unpackFloat(raw_message);
    finalize(raw_message);
    at::addbmm_out(tensor, src, batch1, batch2, beta, alpha);
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
    at::baddbmm_out(tensor, src, batch1, batch2, at::Scalar((int64_t)beta), at::Scalar((int64_t)alpha));
  } else if (at::isFloatingType(tensor.type().scalarType())) {
    double beta = unpackFloat(raw_message);
    double alpha = unpackFloat(raw_message);
    finalize(raw_message);
    at::baddbmm_out(tensor, src, batch1, batch2, beta, alpha);
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
  at::max_out(tensor, indices_, src, dimension, keepdim);
}

static void tensorMin(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor indices_ = unpackRetrieveTensor(raw_message);
  at::Tensor src = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  int keepdim = unpackInteger(raw_message);
  finalize(raw_message);
  at::min_out(tensor, indices_, src, dimension, keepdim);
}

static void tensorKthvalue(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor indices_ = unpackRetrieveTensor(raw_message);
  at::Tensor src = unpackRetrieveTensor(raw_message);
  int k = unpackInteger(raw_message);
  int dimension = unpackInteger(raw_message);
  int keepdim = unpackInteger(raw_message);
  finalize(raw_message);
  at::kthvalue_out(tensor, indices_, src, k, dimension, keepdim);
}

static void tensorMode(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor indices_ = unpackRetrieveTensor(raw_message);
  at::Tensor src = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  int keepdim = unpackInteger(raw_message);
  finalize(raw_message);
  at::mode_out(tensor, indices_, src, dimension, keepdim);
}

static void tensorMedian(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor indices_ = unpackRetrieveTensor(raw_message);
  at::Tensor src = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  int keepdim = unpackInteger(raw_message);
  finalize(raw_message);
  at::median_out(tensor, indices_, src, dimension, keepdim);
}

static void tensorSum(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  int keepdim = unpackInteger(raw_message);
  finalize(raw_message);
  at::sum_out(tensor, src, dimension, keepdim);
}

static void tensorProd(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  int keepdim = unpackInteger(raw_message);
  finalize(raw_message);
  at::prod_out(tensor, src, dimension, keepdim);
}

static void tensorCumsum(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  finalize(raw_message);
  at::cumsum_out(tensor, src, dimension);
}

static void tensorCumprod(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  finalize(raw_message);
  at::cumprod_out(tensor, src, dimension);
}

static void tensorSign(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  at::sign_out(tensor, src);
}

static void tensorTrace(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  finalize(raw_message);

  if (at::isIntegralType(tensor.type().scalarType())) {
    int64_t value = tensor.trace().toCLong();
    sendValueToMaster(value);
  } else if (at::isFloatingType(tensor.type().scalarType())) {
    double value = tensor.trace().toCDouble();
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
  at::cross_out(tensor, src1, src2, dimension);
}

static void tensorCmax(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src1 = unpackRetrieveTensor(raw_message);
  at::Tensor src2 = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  at::max_out(tensor, src1, src2);
}

static void tensorCmin(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src1 = unpackRetrieveTensor(raw_message);
  at::Tensor src2 = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  at::min_out(tensor, src1, src2);
}

/* static void tensorCmaxValue(rpc::RPCMessage& raw_message) { */
/*   at::Tensor tensor = unpackRetrieveTensor(raw_message); */
/*   at::Tensor src = unpackRetrieveTensor(raw_message); */

/*   if (at::isIntegralType(tensor.type().scalarType())) { */
/*     int64_t value = unpackInteger(raw_message); */
/*     finalize(raw_message); */
/*     at::clamp_out(tensor, src, at::Scalar((int64_t)value)); */
/*   } else if (at::isFloatingType(tensor.type().scalarType())) { */
/*     double value = unpackFloat(raw_message); */
/*     finalize(raw_message); */
/*     at::clamp_out(tensor, src, value); */
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
