#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "storages/generic/THStorage.cpp"
#else

template<>
THStorage<real>::THStorage(): storage(THStorage_(new)()) {}

template<>
THStorage<real>::THStorage(storage_type* storage): storage(storage) {}

template<>
THStorage<real>::THStorage(std::size_t storage_size)
  : storage(THStorage_(newWithSize)(storage_size)) {}

template<>
THStorage<real>::~THStorage() {
  THStorage_(free)(storage);
}

template<>
std::size_t THStorage<real>::elementSize() const {
  return sizeof(real);
}

template<>
std::size_t THStorage<real>::size() const {
  return storage->size;
}

template<>
void* THStorage<real>::data() {
  return storage->data;
}

template<>
const void* THStorage<real>::data() const {
  return storage->data;
}

template<>
auto THStorage<real>::retain() -> THStorage& {
  THStorage_(retain)(storage);
  return *this;
}

template<>
auto THStorage<real>::free() -> THStorage& {
  THStorage_(free)(storage);
  return *this;
}

template<>
auto THStorage<real>::resize(int64_t new_size) -> THStorage& {
  THStorage_(resize)(storage, new_size);
  return *this;
}

template<>
auto THStorage<real>::fill(scalar_type value) -> THStorage& {
  THStorage_(fill)(storage, value);
  return *this;
}

template<>
auto THStorage<real>::set(std::size_t ind, scalar_type value) -> THStorage& {
  THStorage_(set)(storage, ind, value);
  return *this;
}

template<>
auto THStorage<real>::fast_set(std::size_t ind, scalar_type value) -> THStorage& {
  storage->data[ind] = value;
  return *this;
}

template<>
auto THStorage<real>::get(std::size_t ind) -> scalar_type {
  return THStorage_(get)(storage, ind);
}

template<>
auto THStorage<real>::fast_get(std::size_t ind) -> scalar_type {
  return storage->data[ind];
}

template<>
thpp::Type THStorage<real>::type() const {
  return thpp::type_traits<real>::type;
}

template<>
bool THStorage<real>::isCuda() const {
  return false;
}

template<>
int THStorage<real>::getDevice() const {
  return -1;
}

template<>
std::unique_ptr<Tensor> THStorage<real>::newTensor() const {
  return std::unique_ptr<Tensor>(new THTensor<real>());
}

template<>
THStorage<real>::storage_type *THStorage<real>::getRaw() const {
  return storage;
}

#endif
