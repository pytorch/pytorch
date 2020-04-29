#pragma once

#include <pytorch_qnnpack.h>

namespace qnnpack {
/* Deleter for the QNNPACK operator structures.
 *
 * Usage:
 *   std::unique_ptr<pytorch_qnnp_operator, QnnpackDeleter> qnnpack_uniq_ptr(op);
 */
struct QnnpackDeleter {
  void operator()(pytorch_qnnp_operator_t op) {
    pytorch_qnnp_delete_operator(op);
  }
};

}  // namespace qnnpack
