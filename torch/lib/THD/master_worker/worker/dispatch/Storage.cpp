static std::unique_ptr<thpp::Storage> createStorage(thpp::Type type) {
  if (type == thpp::Type::UCHAR)
    return std::unique_ptr<thpp::Storage>(new thpp::THStorage<uint8_t>());
  else if (type == thpp::Type::CHAR)
    return std::unique_ptr<thpp::Storage>(new thpp::THStorage<int8_t>());
  else if (type == thpp::Type::SHORT)
    return std::unique_ptr<thpp::Storage>(new thpp::THStorage<int16_t>());
  else if (type == thpp::Type::INT)
    return std::unique_ptr<thpp::Storage>(new thpp::THStorage<int32_t>());
  else if (type == thpp::Type::LONG_LONG)
    return std::unique_ptr<thpp::Storage>(new thpp::THStorage<int64_t>());
  else if (type == thpp::Type::FLOAT)
    return std::unique_ptr<thpp::Storage>(new thpp::THStorage<float>());
  else if (type == thpp::Type::DOUBLE)
    return std::unique_ptr<thpp::Storage>(new thpp::THStorage<double>());
  throw std::invalid_argument("passed character doesn't represent a storage type");
}

static std::unique_ptr<thpp::Storage> createStorage(thpp::Type type, std::size_t size) {
  std::unique_ptr<thpp::Storage> storage = createStorage(type);
  storage->resize(size);
  return storage;
}

static void storageSet(rpc::RPCMessage& raw_message) {
  thpp::Storage *storage = unpackRetrieveStorage(raw_message);
  ptrdiff_t offset = unpackInteger(raw_message);
  thpp::Type type = peekType(raw_message);
  if (thpp::isInteger(type)) {
    int64_t value = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::IntStorage*>(storage)->set(offset, value);
  } else if (thpp::isFloat(type)) {
    double value = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::FloatStorage*>(storage)->set(offset, value);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void storageGet(rpc::RPCMessage& raw_message) {
  thpp::Storage *storage = unpackRetrieveStorage(raw_message);
  ptrdiff_t offset = unpackInteger(raw_message);
  thpp::Type type = unpackType(raw_message);
  finalize(raw_message);
  if (thpp::isInteger(type)) {
    int64_t value = dynamic_cast<thpp::IntStorage*>(storage)->get(offset);
    sendValueToMaster(value);
  } else if (thpp::isFloat(type)) {
    double value = dynamic_cast<thpp::FloatStorage*>(storage)->get(offset);
    sendValueToMaster(value);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void storageNew(rpc::RPCMessage& raw_message) {
  thpp::Type storage_type = unpackType(raw_message);
  object_id_type storage_id = unpackStorage(raw_message);
  finalize(raw_message);
  workerStorages.emplace(
    storage_id,
    createStorage(storage_type)
  );
}

static void storageNewWithSize(rpc::RPCMessage& raw_message) {
  thpp::Type storage_type = unpackType(raw_message);
  object_id_type storage_id = unpackStorage(raw_message);
  int64_t size = unpackInteger(raw_message);
  finalize(raw_message);
  workerStorages.emplace(
    storage_id,
    createStorage(storage_type, size)
  );
}

static void storageNewWithSizeN(rpc::RPCMessage& raw_message, std::size_t size) {
  thpp::Type storage_type = unpackType(raw_message);
  object_id_type storage_id = unpackStorage(raw_message);
  std::unique_ptr<thpp::Storage> storage = createStorage(storage_type, size);
  thpp::Type value_type = peekType(raw_message);
  if (thpp::isInteger(value_type)) {
    thpp::IntStorage* raw_storage = dynamic_cast<thpp::IntStorage*>(storage.get());
    int64_t values[size];
    for (std::size_t i = 0; i < size; i++)
      values[i] = unpackInteger(raw_message);
    finalize(raw_message);
    for (std::size_t i = 0; i < size; i++)
      raw_storage->fast_set(i, values[i]);
  } else if (thpp::isFloat(value_type)) {
    thpp::FloatStorage* raw_storage = dynamic_cast<thpp::FloatStorage*>(storage.get());
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
  thpp::Storage *storage = unpackRetrieveStorage(raw_message);
  int64_t new_size = unpackInteger(raw_message);
  finalize(raw_message);
  storage->resize(new_size);
}

static void storageFill(rpc::RPCMessage& raw_message) {
  thpp::Storage *storage = unpackRetrieveStorage(raw_message);
  thpp::Type type = peekType(raw_message);
  if (thpp::isInteger(type)) {
    int64_t val = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::IntStorage*>(storage)->fill(val);
  } else if (thpp::isFloat(type)) {
    double val = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::FloatStorage*>(storage)->fill(val);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}
