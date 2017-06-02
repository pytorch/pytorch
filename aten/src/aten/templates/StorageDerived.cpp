#include "${Storage}.h"
#include "HalfConvert.h"

namespace tlib {

${Storage}::${Storage}(Context* context):
    storage(${THStorage}_new(${state})), context(context) {}

${Storage}::${Storage}(Context* context, ${THStorage}* storage):
    storage(storage), context(context) {}

${Storage}::${Storage}(Context* context, std::size_t storage_size)
  : storage(${THStorage}_newWithSize(${state,} storage_size)), context(context) {}

${Storage}::~${Storage}() {
  ${THStorage}_free(${state,} storage);
}

std::size_t ${Storage}::elementSize() const {
  return sizeof(${ScalarType});
}

std::size_t ${Storage}::size() const {
  return storage->size;
}

void* ${Storage}::data() {
  return storage->data;
}

const void* ${Storage}::data() const {
  return storage->data;
}

auto ${Storage}::retain() -> ${Storage}& {
  ${THStorage}_retain(${state,} storage);
  return *this;
}

auto ${Storage}::free() -> ${Storage}& {
  ${THStorage}_free(${state,} storage);
  return *this;
}

auto ${Storage}::resize(long new_size) -> ${Storage}& {
  ${THStorage}_resize(${state,} storage, new_size);
  return *this;
}

auto ${Storage}::fill(Scalar value) -> ${Storage}& {
  ${THStorage}_fill(${state,} storage, ${to_th_half}(value.to${ScalarName}()));
  return *this;
}

auto ${Storage}::set(std::size_t ind, Scalar value) -> ${Storage}& {
  ${THStorage}_set(${state,} storage, ind, ${to_th_half}(value.to${ScalarName}()));
  return *this;
}

auto ${Storage}::fast_set(std::size_t ind, Scalar value) -> ${Storage}& {
  throw std::runtime_error("unsupported operation 'fast_set'");
}

auto ${Storage}::get(std::size_t ind) -> Scalar {
  return ${to_tlib_half}(${THStorage}_get(${state,} storage, ind));
}

auto ${Storage}::fast_get(std::size_t ind) -> Scalar {
  if(${isCUDA})
    throw std::runtime_error("unsupported operation 'fast_get'");
  return ${to_tlib_half}(storage->data[ind]);
}

int ${Storage}::getDevice() const {
  ${storage_device} //storage->device;
}

Type& ${Storage}::type() const {
  throw std::runtime_error("NYI - Storage::type()");
}

}
