#include "TensorLib/Type.h"

${type_headers}

namespace tlib {

void Type::registerAll(Context * context) {
  ${type_registrations}
}

}
