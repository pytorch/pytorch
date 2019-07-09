#pragma once

#ifdef USE_QNNPACK
#include <qnnpack.h>

struct QnnpackOperatorDeleter {
  void operator()(qnnp_operator_t op) {
    qnnp_delete_operator(op);
  }
};
#endif
