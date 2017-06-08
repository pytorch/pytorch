#include "TensorLib/Type.h"
#include "TensorLib/Tensor.h"

${type_headers}

namespace tlib {

void Type::registerAll(Context * context) {
  ${type_registrations}
}

${type_method_definitions}

}
