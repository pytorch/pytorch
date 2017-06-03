#include "${Type}.h"
#include "${Storage}.h"

namespace tlib {

${Type}::${Type}(Context* context)
: context(context) {}
ScalarType ${Type}::scalarType() {
  return ScalarType::${ScalarName};
}
Processor ${Type}::processor() {
  return Processor::${Processor};
}
bool ${Type}::isSparse() { return false; }
bool ${Type}::isDistributed() { return false; }

Storage * ${Type}::newStorage() {
  return new ${Storage}(context);
}
Storage * ${Type}::newStorage(size_t size) {
  return new ${Storage}(context,size);
}

}
