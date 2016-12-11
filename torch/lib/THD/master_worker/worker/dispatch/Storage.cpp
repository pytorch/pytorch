static std::unique_ptr<Storage> createStorage(Type type) {
  if (type == Type::UCHAR)
    return std::unique_ptr<Storage>(new THStorage<unsigned char>());
  else if (type == Type::CHAR)
    return std::unique_ptr<Storage>(new THStorage<char>());
  else if (type == Type::SHORT)
    return std::unique_ptr<Storage>(new THStorage<short>());
  else if (type == Type::INT)
    return std::unique_ptr<Storage>(new THStorage<int>());
  else if (type == Type::LONG)
    return std::unique_ptr<Storage>(new THStorage<long>());
  else if (type == Type::FLOAT)
    return std::unique_ptr<Storage>(new THStorage<float>());
  else if (type == Type::DOUBLE)
    return std::unique_ptr<Storage>(new THStorage<double>());
  throw std::invalid_argument("passed character doesn't represent a storage type");
}

static std::unique_ptr<Storage> createStorage(Type type, std::size_t size) {
  std::unique_ptr<Storage> storage = createStorage(type);
  storage->resize(size);
  return storage;
}

static void storageSet(rpc::RPCMessage& raw_message) {
  Storage *storage = unpackRetrieveStorage(raw_message);
  ptrdiff_t offset = unpackInteger(raw_message);
  Type type = peekType(raw_message);
  if (isInteger(type)) {
    long long value = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<IntStorage *>(storage)->set(offset, value);
  } else if (isFloat(type)) {
    double value = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<FloatStorage *>(storage)->set(offset, value);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void storageGet(rpc::RPCMessage& raw_message) {
  Storage *storage = unpackRetrieveStorage(raw_message);
  ptrdiff_t offset = unpackInteger(raw_message);
  Type type = unpackType(raw_message);
  finalize(raw_message);
  if (isInteger(type)) {
    long long value = dynamic_cast<IntStorage *>(storage)->get(offset);
    sendValueToMaster(dynamic_cast<IntStorage *>(storage), value);
  } else if (isFloat(type)) {
    double value = dynamic_cast<FloatStorage *>(storage)->get(offset);
    sendValueToMaster(dynamic_cast<FloatStorage *>(storage), value);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void storageConstruct(rpc::RPCMessage& raw_message) {
  Type storage_type = unpackType(raw_message);
  object_id_type storage_id = unpackStorage(raw_message);
  finalize(raw_message);
  workerStorages.emplace(
    storage_id,
    createStorage(storage_type)
  );
}

static void storageConstructWithSize(rpc::RPCMessage& raw_message) {
  Type storage_type = unpackType(raw_message);
  object_id_type storage_id = unpackStorage(raw_message);
  long long size = unpackInteger(raw_message);
  finalize(raw_message);
  workerStorages.emplace(
    storage_id,
    createStorage(storage_type, size)
  );
}

static void storageConstructWithSizeN(rpc::RPCMessage& raw_message, std::size_t size) {
  Type storage_type = unpackType(raw_message);
  object_id_type storage_id = unpackStorage(raw_message);
  std::unique_ptr<Storage> storage = createStorage(storage_type, size);
  Type value_type = peekType(raw_message);
  if (isInteger(value_type)) {
    IntStorage *raw_storage = dynamic_cast<IntStorage *>(storage.get());
    long long values[size];
    for (std::size_t i = 0; i < size; i++)
      values[i] = unpackInteger(raw_message);
    finalize(raw_message);
    for (std::size_t i = 0; i < size; i++)
      raw_storage->fast_set(i, values[i]);
  } else if (isFloat(value_type)) {
    FloatStorage *raw_storage = dynamic_cast<FloatStorage *>(storage.get());
    double values[size];
    for (std::size_t i = 0; i < size; i++)
      values[i] = unpackInteger(raw_message);
    finalize(raw_message);
    for (std::size_t i = 0; i < size; i++)
      raw_storage->fast_set(i, values[i]);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
  finalize(raw_message);
  workerStorages.emplace(
    storage_id,
    std::move(storage)
  );
}

static void storageConstructWithSize1(rpc::RPCMessage& raw_message) {
  storageConstructWithSizeN(raw_message, 1);
}

static void storageConstructWithSize2(rpc::RPCMessage& raw_message) {
  storageConstructWithSizeN(raw_message, 2);
}

static void storageConstructWithSize3(rpc::RPCMessage& raw_message) {
  storageConstructWithSizeN(raw_message, 3);
}

static void storageConstructWithSize4(rpc::RPCMessage& raw_message) {
  storageConstructWithSizeN(raw_message, 4);
}

static void storageFree(rpc::RPCMessage& raw_message) {
  object_id_type storage_id = unpackStorage(raw_message);
  finalize(raw_message);
  workerStorages.erase(storage_id);
}

static void storageResize(rpc::RPCMessage& raw_message) {
  Storage *storage = unpackRetrieveStorage(raw_message);
  long long new_size = unpackInteger(raw_message);
  finalize(raw_message);
  storage->resize(new_size);
}

static void storageFill(rpc::RPCMessage& raw_message) {
  Storage *storage = unpackRetrieveStorage(raw_message);
  Type type = peekType(raw_message);
  if (isInteger(type)) {
    long long val = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<IntStorage *>(storage)->fill(val);
  } else if (isFloat(type)) {
    double val = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<FloatStorage *>(storage)->fill(val);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}
