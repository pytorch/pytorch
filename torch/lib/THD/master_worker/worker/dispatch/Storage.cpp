
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

static void storageFree(rpc::RPCMessage& raw_message) {
  object_id_type storage_id = unpackStorage(raw_message);
  workerTensors.erase(storage_id);
}

