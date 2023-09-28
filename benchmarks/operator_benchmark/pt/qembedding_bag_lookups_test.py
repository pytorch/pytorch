from typing import Optional

import numpy as np
import torch

from torch.testing._internal.common_quantization import lengths_to_offsets

import operator_benchmark as op_bench

torch.ops.load_library("//caffe2/torch/fb/sparsenn:sparsenn_operators")


embedding_bag_rowwise_offsets_short_configs = op_bench.cross_product_configs(
    num_embeddings=(80,),
    embedding_dim=(128, 256),
    num_offsets=range(2, 10),
    enable_per_sample_weights=(True, False),
    include_last_offset=(True, False),
    is_pruned_weights=(
        True,
        False,
    ),
    use_32bit_indices=(True, False),
    use_32bit_offsets=(True, False),
    tags=["short"],
)


embedding_bag_rowwise_offsets_long_configs = op_bench.cross_product_configs(
    num_embeddings=(100, 120, 1000, 10_000, 20_000),
    embedding_dim=(16, 64, 128, 256),
    num_offsets=range(10, 20),
    enable_per_sample_weights=(True, False),
    include_last_offset=(True, False),
    is_pruned_weights=(
        True,
        False,
    ),
    use_32bit_indices=(True, False),
    use_32bit_offsets=(True, False),
    tags=["long"],
)


full_configs = (
    embedding_bag_rowwise_offsets_short_configs
    + embedding_bag_rowwise_offsets_long_configs
)

four_bit_rowwise_ops = op_bench.op_list(
    attrs=(
        (
            "qembeddingbag_4bit_rowwise_offsets",
            torch.ops.quantized.embedding_bag_4bit_rowwise_offsets,
        ),
    ),
    attr_names=("op_name", "op_func"),
)

byte_rowwise_ops = op_bench.op_list(
    attrs=(
        (
            "qembeddingbag_byte_rowwise_offsets",
            torch.ops.quantized.embedding_bag_byte_rowwise_offsets,
        ),
    ),
    attr_names=("op_name", "op_func"),
)


def get_pruned_weights_and_mapping(q_weights):
    indicator = torch.from_numpy(
        np.random.uniform(low=-1.0, high=1.0, size=[q_weights.shape[0]]).astype(
            np.float32
        )
    )

    (
        q_pruned_weights,
        compressed_indices_mapping,
    ) = torch.ops.fb.embedding_bag_rowwise_prune(
        q_weights, indicator, 0.01, torch.int32
    )

    return q_pruned_weights, compressed_indices_mapping


class EmbedddingBag4BitRowwiseOffsetsTest(op_bench.TorchBenchmarkBase):
    def init(
        self,
        num_embeddings: int,
        embedding_dim: int,
        num_offsets: int,
        enable_per_sample_weights: bool,
        include_last_offset: bool,
        is_pruned_weights: bool,
        use_32bit_indices: bool,
        use_32bit_offsets: bool,
        op_func,
    ):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_offsets = num_offsets
        self.enable_per_sample_weights = enable_per_sample_weights
        self.include_last_offset = include_last_offset
        self.max_segment_length = 20
        self.num_lengths = np.random.randint(1, num_offsets + 1)
        self.lengths = np.random.randint(
            0, self.max_segment_length + 1, size=self.num_lengths
        ).astype(np.int32)
        self.num_indices = np.sum(self.lengths)
        self.is_pruned_weights = is_pruned_weights
        self.use_32bit_indices = use_32bit_indices
        self.use_32bit_offsets = use_32bit_offsets

        self.offsets = lengths_to_offsets(self.lengths)
        self.indices = torch.from_numpy(
            np.random.randint(
                low=0, high=num_embeddings, size=self.num_indices, dtype=np.int64
            )
        )

        self.indices = self.indices.int() if self.use_32bit_indices else self.indices
        self.offsets = self.offsets.int() if self.use_32bit_offsets else self.offsets

        if self.include_last_offset:
            self.offsets = torch.cat(
                (self.offsets, torch.tensor([self.indices.size(0)], dtype=torch.long)),
                0,
            )

        self.weights = torch.from_numpy(
            (
                np.random.random_sample((self.num_embeddings, self.embedding_dim)) + 1
            ).astype(np.float32)
        )
        self.indices = torch.from_numpy(
            np.random.randint(
                low=0, high=self.num_embeddings, size=self.num_indices, dtype=np.int64
            )
        )
        self.prepack_func = torch.ops.quantized.embedding_bag_4bit_prepack

        self.prepacked_weights = self.prepack_func(self.weights)
        self.per_sample_weights = (
            torch.from_numpy(
                np.random.uniform(low=0.01, high=0.5, size=[len(self.indices)]).astype(
                    np.float32
                )
            )
            if self.enable_per_sample_weights
            else None
        )

        self.compressed_indices = None

        if self.is_pruned_weights:
            (
                self.prepacked_weights,
                self.compressed_indices,
            ) = get_pruned_weights_and_mapping(self.prepacked_weights)

        self.inputs = {
            "prepacked_weights": self.prepacked_weights,
            "indices": self.indices,
            "offsets": self.offsets,
            "mode": 0,
            "per_sample_weights": self.per_sample_weights,
            "include_last_offset": self.include_last_offset,
            "is_pruned_weights": self.is_pruned_weights,
            "compressed_indices": self.compressed_indices,
        }

        self.op_func = op_func

    def forward(
        self,
        prepacked_weights,
        indices,
        offsets,
        mode: int,
        per_sample_weights: Optional[torch.Tensor],
        include_last_offset: bool,
        is_pruned_weights: bool,
        compressed_indices: Optional[torch.Tensor],
    ):
        return self.op_func(
            prepacked_weights,
            indices,
            offsets,
            mode=mode,
            per_sample_weights=per_sample_weights,
            include_last_offset=include_last_offset,
            pruned_weights=is_pruned_weights,
            compressed_indices_mapping=compressed_indices,
        )


class EmbedddingBagByteRowwiseOffsetsTest(op_bench.TorchBenchmarkBase):
    def init(
        self,
        num_embeddings: int,
        embedding_dim: int,
        num_offsets: int,
        enable_per_sample_weights: bool,
        include_last_offset: bool,
        is_pruned_weights: bool,
        use_32bit_indices: bool,
        use_32bit_offsets: bool,
        op_func,
    ):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_offsets = num_offsets
        self.enable_per_sample_weights = enable_per_sample_weights
        self.include_last_offset = include_last_offset
        self.max_segment_length = 20
        self.num_lengths = np.random.randint(1, num_offsets + 1)
        self.lengths = np.random.randint(
            0, self.max_segment_length + 1, size=self.num_lengths
        ).astype(np.int32)
        self.is_pruned_weights = is_pruned_weights
        self.use_32bit_indices = use_32bit_indices
        self.use_32bit_offsets = use_32bit_offsets

        self.num_indices = np.sum(self.lengths)
        self.offsets = lengths_to_offsets(self.lengths)
        self.indices = torch.from_numpy(
            np.random.randint(
                low=0, high=num_embeddings, size=self.num_indices, dtype=np.int64
            )
        )

        self.indices = self.indices.int() if self.use_32bit_indices else self.indices
        self.offsets = self.offsets.int() if self.use_32bit_offsets else self.offsets

        if include_last_offset:
            self.offsets = torch.cat(
                (self.offsets, torch.tensor([self.indices.size(0)], dtype=torch.long)),
                0,
            )

        self.weights = torch.from_numpy(
            (
                np.random.random_sample((self.num_embeddings, self.embedding_dim)) + 1
            ).astype(np.float32)
        )
        self.indices = torch.from_numpy(
            np.random.randint(
                low=0, high=self.num_embeddings, size=self.num_indices, dtype=np.int64
            )
        )

        self.prepack_func = torch.ops.quantized.embedding_bag_byte_prepack

        self.prepacked_weights = self.prepack_func(self.weights)
        self.per_sample_weights = (
            torch.from_numpy(
                np.random.uniform(low=0.01, high=0.5, size=[len(self.indices)]).astype(
                    np.float32
                )
            )
            if self.enable_per_sample_weights
            else None
        )

        self.compressed_indices = None

        if self.is_pruned_weights:
            (
                self.prepacked_weights,
                self.compressed_indices,
            ) = get_pruned_weights_and_mapping(self.prepacked_weights)

        self.inputs = {
            "prepacked_weights": self.prepacked_weights,
            "indices": self.indices,
            "offsets": self.offsets,
            "mode": 0,
            "per_sample_weights": self.per_sample_weights,
            "include_last_offset": self.include_last_offset,
            "is_pruned_weights": self.is_pruned_weights,
            "compressed_indices": self.compressed_indices,
        }

        self.op_func = op_func

    def forward(
        self,
        prepacked_weights,
        indices,
        offsets,
        mode: int,
        per_sample_weights: Optional[torch.Tensor],
        include_last_offset: bool,
        is_pruned_weights: bool,
        compressed_indices: Optional[torch.Tensor],
    ):
        return self.op_func(
            prepacked_weights,
            indices,
            offsets,
            mode=0,
            per_sample_weights=per_sample_weights,
            include_last_offset=self.include_last_offset,
            pruned_weights=self.is_pruned_weights,
            compressed_indices_mapping=self.compressed_indices,
        )


op_bench.generate_pt_tests_from_op_list(
    four_bit_rowwise_ops, full_configs, EmbedddingBag4BitRowwiseOffsetsTest
)
op_bench.generate_pt_tests_from_op_list(
    byte_rowwise_ops, full_configs, EmbedddingBagByteRowwiseOffsetsTest
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
