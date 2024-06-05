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

enum pytorch_qnnp_status pytorch_qnnp_delete_operator(
    pytorch_qnnp_operator_t op) {
  if (op == NULL) {
    return pytorch_qnnp_status_invalid_parameter;
  }

  free(op->indirection_buffer);
  free(op->packed_weights);
  free(op->a_sum);
  free(op->zero_buffer);
  free(op->lookup_table);
  free(op);
  return pytorch_qnnp_status_success;
}
