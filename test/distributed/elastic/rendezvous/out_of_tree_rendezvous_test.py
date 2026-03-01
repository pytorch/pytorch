# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import unittest
from unittest.mock import MagicMock, patch

import torch.distributed.elastic.rendezvous as rdvz


BACKEND_NAME = "testbackend"


class OutOfTreeRendezvousTest(unittest.TestCase):
    def test_out_of_tree_handler_loading(self):
        rdvz._register_out_of_tree_handlers()
        registry_dict = rdvz.rendezvous_handler_registry._registry

        # test backend should not be registered as a backend
        self.assertFalse(BACKEND_NAME in registry_dict)

        # Create a mock handler function that will be returned by the entry point
        def test_handler():
            pass

        # Create a mock entry point
        mock_entry_point = MagicMock()
        mock_entry_point.name = BACKEND_NAME
        mock_entry_point.load.return_value = test_handler

        # Create a mock EntryPoints object
        mock_entry_points = MagicMock()
        mock_entry_points.__iter__.return_value = iter([mock_entry_point])
        mock_entry_points.__getitem__.return_value = mock_entry_point

        with patch(
            "torch.distributed.elastic.rendezvous.registry.entry_points"
        ) as mock_ep:
            mock_ep.return_value = mock_entry_points

            # Registering the out of tree handlers again
            rdvz._register_out_of_tree_handlers()

            # Verify entry_points was called with the correct group
            mock_ep.assert_called_once_with(group="torchrun.handlers")

        # test backend should be registered as a backend
        self.assertTrue(BACKEND_NAME in registry_dict)
