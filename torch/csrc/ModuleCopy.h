struct pair_hasher {

  size_t operator()(std::pair<PyObject *, PyObject *> types) const {
    size_t seed = ptr_hash(std::get<0>(types));
    seed ^= ptr_hash(std::get<1>(types)) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
  }

  std::hash<PyObject *> ptr_hash = std::hash<PyObject *>();
};

using THPCopyFunction = void (*)(PyObject *dst, PyObject *src);
extern std::unordered_map<std::pair<PyObject *, PyObject *>, THPCopyFunction, pair_hasher> tensor_copy_handlers;
extern std::unordered_map<std::pair<PyObject *, PyObject *>, THPCopyFunction, pair_hasher> storage_copy_handlers;
