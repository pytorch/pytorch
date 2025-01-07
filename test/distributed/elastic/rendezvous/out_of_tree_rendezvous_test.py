# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import pathlib
import sys
import unittest

import torch.distributed.elastic.rendezvous as rdvz


BACKEND_NAME = "testbackend"
TEST_PACKAGE_PATH = "/out_of_tree_test_package/src"


class OutOfTreeRendezvousTest(unittest.TestCase):
    def test_out_of_tree_handler_loading(self):
        current_path = str(pathlib.Path(__file__).parent.resolve())
        rdvz._register_out_of_tree_handlers()
        registry_dict = rdvz.rendezvous_handler_registry._registry

        # test backend should not be registered as a backend
        self.assertFalse(BACKEND_NAME in registry_dict)

        # Including testbackend in python path
        sys.path.append(current_path + TEST_PACKAGE_PATH)

        # Registering the out of tree handlers again
        rdvz._register_out_of_tree_handlers()

        # test backend should be registered as a backend
        self.assertTrue(BACKEND_NAME in registry_dict)

        # Removing testbackend from python path
        sys.path.remove(current_path + TEST_PACKAGE_PATH)
