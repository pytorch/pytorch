# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# TODO: remove this file when the migration of the pytree utility is done
from torch.utils._pytree import tree_map_, treespec_pprint


__all__ = ["tree_map_", "treespec_pprint"]
