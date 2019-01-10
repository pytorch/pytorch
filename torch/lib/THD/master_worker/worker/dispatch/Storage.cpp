static std::unique_ptr<at::Storage> createStorage(RPCType type) {
  if (type == RPCType::UCHAR)
    return at::getType(at::Backend::CPU, at::ScalarType::Byte).storage();
  else if (type == RPCType::CHAR)
    return at::getType(at::Backend::CPU, at::ScalarType::Char).storage();
  else if (type == RPCType::SHORT)
    return at::getType(at::Backend::CPU, at::ScalarType::Short).storage();
  else if (type == RPCType::INT)
    return at::getType(at::Backend::CPU, at::ScalarType::Int).storage();
  else if (type == RPCType::LONG)
    return at::getType(at::Backend::CPU, at::ScalarType::Long).storage();
  else if (type == RPCType::FLOAT)
    return at::getType(at::Backend::CPU, at::ScalarType::Float).storage();
  else if (type == RPCType::DOUBLE)
    return at::getType(at::Backend::CPU, at::ScalarType::Double).storage();
  throw std::invalid_argument("passed character doesn't represent a storage type");
}

static std::unique_ptr<at::Storage> createStorage(RPCType type, std::size_t size) {
  std::unique_ptr<at::Storage> storage = createStorage(type);
  storage->resize(size);
  return storage;
}

static void storageSet(rpc::RPCMessage& raw_message) {
  at::Storage *storage = unpackRetrieveStorage(raw_message);
  ptrdiff_t offset = unpackInteger(raw_message);
  RPCType type = peekType(raw_message);
  if (isInteger(type)) {
    int64_t value = unpackInteger(raw_message);
    finalize(raw_message);
    storage->set(offset, value);
  } else if (isFloat(type)) {
    double value = unpackFloat(raw_message);
    finalize(raw_message);
    storage->set(offset, value);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void storageGet(rpc::RPCMessage& raw_message) {
  at::Storage *storage = unpackRetrieveStorage(raw_message);
  ptrdiff_t offset = unpackInteger(raw_message);
  RPCType type = unpackType(raw_message);
  finalize(raw_message);
  if (isInteger(type)) {
    int64_t value = storage->get(offset).to<int64_t>();
    sendValueToMaster(value);
  } else if (isFloat(type)) {
    double value = storage->get(offset).to<double>();
    sendValueToMaster(value);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void storageNew(rpc::RPCMessage& raw_message) {
  RPCType storage_type = unpackType(raw_message);
  object_id_type storage_id = unpackStorage(raw_message);
  finalize(raw_message);
  workerStorages.emplace(
    storage_id,
    createStorage(storage_type)
  );
}

static void storageNewWithSize(rpc::RPCMessage& raw_message) {
  RPCType storage_type = unpackType(raw_message);
  object_id_type storage_id = unpackStorage(raw_message);
  int64_t size = unpackInteger(raw_message);
  finalize(raw_message);
  workerStorages.emplace(
    storage_id,
    createStorage(storage_type, size)
  );
}

static void storageNewWithSizeN(rpc::RPCMessage& raw_message, std::size_t size) {
  RPCType storage_type = unpackType(raw_message);
  object_id_type storage_id = unpackStorage(raw_message);
  std::unique_ptr<at::Storage> storage = createStorage(storage_type, size);
  RPCType value_type = peekType(raw_message);
  if (isInteger(value_type)) {
    int64_t values[size];
    for (std::size_t i = 0; i < size; i++)
      values[i] = unpackInteger(raw_message);
    finalize(raw_message);
    for (std::size_t i = 0; i < size; i++)
      storage->fast_set(i, values[i]);
  } else if (isFloat(value_type)) {
    double values[size];
    for (std::size_t i = 0; i < size; i++)
      values[i] = unpackInteger(raw_message);
    finalize(raw_message);
    for (std::size_t i = 0; i < size; i++)
      storage->fast_set(i, values[i]);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
  finalize(raw_message);
  workerStorages.emplace(
    storage_id,
    std::move(storage)
  );
}

static void storageNewWithSize1(rpc::RPCMessage& raw_message) {
  storageNewWithSizeN(raw_message, 1);
}

static void storageNewWithSize2(rpc::RPCMessage& raw_message) {
  storageNewWithSizeN(raw_message, 2);
}

static void storageNewWithSize3(rpc::RPCMessage& raw_message) {
  storageNewWithSizeN(raw_message, 3);
}

static void storageNewWithSize4(rpc::RPCMessage& raw_message) {
  storageNewWithSizeN(raw_message, 4);
}

static void storageFree(rpc::RPCMessage& raw_message) {
  object_id_type storage_id = unpackStorage(raw_message);
  finalize(raw_message);
  workerStorages.erase(storage_id);
}

static void storageResize(rpc::RPCMessage& raw_message) {
  at::Storage *storage = unpackRetrieveStorage(raw_message);
  int64_t new_size = unpackInteger(raw_message);
  finalize(raw_message);
  storage->resize(new_size);
}

static void storageFill(rpc::RPCMessage& raw_message) {
  at::Storage *storage = unpackRetrieveStorage(raw_message);
  RPCType type = peekType(raw_message);
  if (isInteger(type)) {
    int64_t val = unpackInteger(raw_message);
    finalize(raw_message);
    storage->fill(val);
  } else if (isFloat(type)) {
    double val = unpackFloat(raw_message);
    finalize(raw_message);
    storage->fill(val);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}
