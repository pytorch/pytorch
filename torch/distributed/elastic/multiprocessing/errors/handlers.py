# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# Multiprocessing error-reporting module


from torch.distributed.elastic.multiprocessing.errors.error_handler import ErrorHandler


__all__ = ["get_error_handler"]


def get_error_handler() -> ErrorHandler:
    return ErrorHandler()
