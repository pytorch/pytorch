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
    for (std::size_t i = 0; i < size; i++)
      raw_storage->fast_set(i, unpackInteger(raw_message));
  } else if (isFloat(value_type)) {
    FloatStorage *raw_storage = dynamic_cast<FloatStorage *>(storage.get());
    for (std::size_t i = 0; i < size; i++)
      raw_storage->fast_set(i, unpackFloat(raw_message));
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
  workerTensors.erase(storage_id);
}
