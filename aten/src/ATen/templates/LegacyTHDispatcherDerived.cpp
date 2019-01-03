#include "ATen/${Dispatcher}.h"

// ${generated_comment}

namespace at {

${Dispatcher}::${Dispatcher}()
  : LegacyTHDispatcher(${Backend}TensorId()) {}

}
