#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torchvision.models as models
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower


def main() -> None:
    model = models.segmentation.deeplabv3_resnet101(weights='DEFAULT').eval()
    sample_inputs = (torch.randn(1, 3, 224, 224), )

    et_program = to_edge_transform_and_lower(
        torch.export.export(model, sample_inputs),
        partitioner=[XnnpackPartitioner()],
    ).to_executorch()

    with open("dl3_xnnpack_fp32.pte", "wb") as file:
        et_program.write_to_file(file)


if __name__ == "__main__":
    main()