from typing import Any, Dict, Optional
from concurrent.futures import Future

from torch.distributed.checkpoint._v2._checkpointing import (
    CheckpointContext,
    CheckpointingConfig,
    Checkpointer,
    RankInfo,
    CheckpointWriter,
    ManifestBuilder,
)
from torch.distributed.checkpoint._v2._metadata import Metadata


class SyncCheckpointer(Checkpointer):

    def __init__(
        self,
        config: CheckpointingConfig,
        rank_info: RankInfo,
        writer: CheckpointWriter,
        manifest_builder: Optional[ManifestBuilder] = None,
    ):
        self._config = config
        self._rank_info = rank_info
        self._writer = writer
        self._cached_metadata: Optional[Metadata] = None
        self.manifest_builder = manifest_builder

    def save(
        self,
        state_dict: Dict[str, Any],
        context: CheckpointContext,
        root_dir: str,
        use_cached_manifest: bool = False,
    ) -> Optional[tuple[Future[None], Future[None]]]:

        if self._config.save_manifest_with_checkpoint and self.manifest_builder is not None:
            if not use_cached_manifest or self._cached_metadata is None:
                manifest = self.manifest_builder.buid_manifest(
                    state_dict=state_dict,
                    context=context,
                )
                self._cached_metadata = Metadata(manifest)

        if self._cached_metadata is None:
            self._cached_metadata = Metadata(None)

        self._writer.write_checkpoint(state_dict, self._cached_metadata, context, root_dir)
