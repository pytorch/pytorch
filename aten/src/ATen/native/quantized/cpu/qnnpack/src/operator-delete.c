/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdlib.h>

#include <pytorch_qnnpack.h>
#include <qnnpack/operator.h>

#ifndef SAFE_FREE
#define SAFE_FREE(ptr)  \
if (ptr != NULL) {      \
  free((ptr));          \
  (ptr) = NULL;         \
}
#endif

enum pytorch_qnnp_status pytorch_qnnp_delete_operator(
    pytorch_qnnp_operator_t op) {
  if (op == NULL) {
    return pytorch_qnnp_status_invalid_parameter;
  }

  SAFE_FREE(op->indirection_buffer);
  SAFE_FREE(op->packed_weights);
  SAFE_FREE(op->a_sum);
  SAFE_FREE(op->zero_buffer);
  SAFE_FREE(op->lookup_table);
  SAFE_FREE(op);
  return pytorch_qnnp_status_success;
}

#undef SAFE_FREE
