#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "storages/generic/THCStorage.cpp"
#else

template<>
THCStorage<real>::THCStorage(THCState* state):
    storage(THCStorage_(new)(state)), state(state) {}

template<>
THCStorage<real>::THCStorage(THCState* state, storage_type* storage):
    storage(storage), state(state) {}

template<>
THCStorage<real>::THCStorage(THCState* state, std::size_t storage_size)
  : storage(THCStorage_(newWithSize)(state, storage_size)), state(state) {}

template<>
THCStorage<real>::~THCStorage() {
  THCStorage_(free)(state, storage);
}

template<>
std::size_t THCStorage<real>::elementSize() const {
  return sizeof(real);
}

template<>
std::size_t THCStorage<real>::size() const {
  return storage->size;
}

template<>
void* THCStorage<real>::data() {
  return storage->data;
}

template<>
const void* THCStorage<real>::data() const {
  return storage->data;
}

template<>
auto THCStorage<real>::retain() -> THCStorage& {
  THCStorage_(retain)(state, storage);
  return *this;
}

template<>
auto THCStorage<real>::free() -> THCStorage& {
  THCStorage_(free)(state, storage);
  return *this;
}

template<>
auto THCStorage<real>::resize(int64_t new_size) -> THCStorage& {
  THCStorage_(resize)(state, storage, new_size);
  return *this;
}

#ifdef THC_REAL_IS_HALF
#define cast_scalar(v) THC_float2half(v)
#else
#define cast_scalar(v) v
#endif

template<>
auto THCStorage<real>::fill(scalar_type value) -> THCStorage& {
  THCStorage_(fill)(state, storage, cast_scalar(value));
  return *this;
}

template<>
auto THCStorage<real>::set(std::size_t ind, scalar_type value) -> THCStorage& {
  THCStorage_(set)(state, storage, ind, cast_scalar(value));
  return *this;
}

template<>
auto THCStorage<real>::fast_set(std::size_t ind, scalar_type value) -> THCStorage& {
  throw std::runtime_error("unsupported operation 'fast_set'");
}

template<>
auto THCStorage<real>::get(std::size_t ind) -> scalar_type {
  auto v = THCStorage_(get)(state, storage, ind);
#ifdef THC_REAL_IS_HALF
  return THC_half2float(v);
#else
  return v;
#endif
}

template<>
auto THCStorage<real>::fast_get(std::size_t ind) -> scalar_type {
  throw std::runtime_error("unsupported operation 'fast_get'");
}

template<>
thpp::Type THCStorage<real>::type() const {
  return thpp::type_traits<real>::type;
}

template<>
bool THCStorage<real>::isCuda() const {
  return true;
}

template<>
int THCStorage<real>::getDevice() const {
  return storage->device;
}

template<>
std::unique_ptr<Tensor> THCStorage<real>::newTensor() const {
  return std::unique_ptr<Tensor>(new THCTensor<real>(state));
}

template<>
THCStorage<real>::storage_type *THCStorage<real>::getRaw() const {
  return storage;
}

#undef cast_scalar

#endif
