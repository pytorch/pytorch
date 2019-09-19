#pragma once

#ifdef USE_PYTORCH_QNNPACK
#include <pytorch_qnnpack.h>

struct QnnpackOperatorDeleter {
  void operator()(pytorch_qnnp_operator_t op) {
    pytorch_qnnp_delete_operator(op);
  }
};

#endif
