
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

static void tensorAdd(rpc::RPCMessage& raw_message) {
  throw std::runtime_error("addition is not yet available");
  //THTensor& result = parse_tensor(raw_message);
  //THTensor& source = parse_tensor(raw_message);
  //double x = parse_scalar(raw_message);
  //assert_end(raw_message);
  //result.add(source, x);
}

static void tensorFree(rpc::RPCMessage& raw_message) {
  object_id_type tensor_id = unpackInteger(raw_message);
  (void)workerTensors.erase(tensor_id);
}

