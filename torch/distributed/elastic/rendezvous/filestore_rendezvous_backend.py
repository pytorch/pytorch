# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Tuple, Optional
from torch.distributed import Store, FileStore
from .dynamic_rendezvous import RendezvousBackend, Token
from .api import RendezvousParameters


log = logging.getLogger(__name__)


class FileStoreRendezvousBackend(RendezvousBackend):
    def __init__(self, store: FileStore, run_id: str) -> None:
        self.store = store
        self.run_id = run_id

    @property
    def name(self) -> str:
        return "FileStore"

    def get_state(self) -> Optional[Tuple[bytes, Token]]:
        print("Getting state!")
        raise NotImplementedError("Not yet implemented")

    def set_state(
        self, state: bytes, token: Optional[Token] = None
    ) -> Optional[Tuple[bytes, Token, bool]]:
        print("Setting state!")
        raise NotImplementedError("Not yet implemented")

def _create_file_store(params: RendezvousParameters) -> FileStore:
    print("Creating filestore!")

    # need to ensure the processes can read and write to this directory.
    # path = f"{path_to_tmp}/tmp/torch/rdzv/{run_id}"
    # assert that min nodes == max nodes, otherwise throw error.
    # num_workers = params.min_nodes
    # return FileStore(path, num_workers)

    raise NotImplementedError("Not yet implemented")

def create_backend(params: RendezvousParameters) -> Tuple[FileStoreRendezvousBackend, Store]:
    print("Creating backend!")

    # store = _create_file_store(params)
    # return FileStoreRendezvousBackend(store, params.run_id)

    raise NotImplementedError("Not yet implemented")
