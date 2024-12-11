#include <torch/csrc/utils/python_compat.h>

namespace torch::dynamo {

void register_variable_tracker_cache(PyObject* mod);

} // namespace torch::dynamo
