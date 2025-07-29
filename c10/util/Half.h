#include <torch/headeronly/util/Half.h>

// need to keep the following for BC because the APIs in here were exposed
// before migrating Half to torch/headeronly
#include <ATen/cpu/vec/vec_half.h>
