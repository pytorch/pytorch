# Owner(s): ["oncall: export"]
import io
import random
import unittest
from dataclasses import dataclass
from typing import Any

import numpy as np

import torch
from torch._dynamo.utils import same
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_MACOS,
    parametrize,
    run_tests,
    TestCase,
)


if not IS_MACOS:
    from torch.testing._internal.distributed.fake_pg import FakeStore


def reset_rng_state():
    torch.manual_seed(1337)
    random.seed(1337)
    np.random.seed(1337)


def run_export_workflow(
    mod: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> torch.nn.Module:
    ep_train = torch.export.export_for_training(mod, args, kwargs)
    buffer = io.BytesIO()
    torch.export.save(ep_train, buffer)
    buffer.seek(0)
    loaded_ep = torch.export.load(buffer)
    unflattened = torch.export.unflatten(loaded_ep)
    return unflattened


@dataclass
class ModelTest:
    model_name: str
    model: torch.nn.Module
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


class ToyModel(torch.nn.Module):
    def __init__(self, in_feat=10, hidden_feat=5000, out_feat=5):
        super().__init__()
        self.net = torch.nn.Sequential(
            *[torch.nn.Linear(in_feat, hidden_feat), torch.nn.ReLU()]
            + [torch.nn.Linear(hidden_feat, hidden_feat), torch.nn.ReLU()]
            + [torch.nn.Linear(hidden_feat, hidden_feat), torch.nn.ReLU()]
            + [torch.nn.Linear(hidden_feat, out_feat), torch.nn.ReLU()]
        )
        self.reset_parameters()

    def forward(self, inputs):
        return self.net(inputs)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)


def get_model_tests() -> list[ModelTest]:
    device = "cuda"

    model_tests = [
        ModelTest(
            model_name="toy",
            model=ToyModel(10, 5000, 5).to(device=device),
            args=(torch.rand(20, 10).to(device=device),),
            kwargs={},
        ),
    ]

    try:
        from transformers import AutoModelForMaskedLM, BertConfig

        batch_size, max_length, config, device = 4, 512, BertConfig(), f"cuda:{0}"
        input_ids = torch.randint(0, config.vocab_size, (batch_size, max_length)).to(
            device
        )
        decoder_ids = torch.randint(0, config.vocab_size, (batch_size, max_length)).to(
            device
        )
        model_tests.append(
            ModelTest(
                model_name="hf_bert",
                model=AutoModelForMaskedLM.from_config(config).to(device),
                args=(),
                kwargs={"input_ids": input_ids, "labels": decoder_ids},
            )
        )
    except ImportError:
        pass

    return model_tests


@unittest.skipIf(IS_MACOS, "Distributed not packaged in macos")
@unittest.skipIf(not torch.cuda.is_available(), "Skip because CUDA is not available")
class TestDistributed(TestCase):
    def check_export_ddp(self, mod, args, kwargs=None) -> None:
        kwargs = kwargs or {}

    @parametrize(
        "model_test",
        get_model_tests(),
        name_fn=lambda model_test: model_test.model_name,
    )
    def test_export_ddp(self, model_test):
        model, args, kwargs = model_test.model, model_test.args, model_test.kwargs

        try:
            torch.distributed.init_process_group(
                backend="fake",
                world_size=2,
                rank=0,
                store=FakeStore(),
            )

            reset_rng_state()
            correct_outputs = model(*args, **kwargs)

            reset_rng_state()
            exported = run_export_workflow(model, args, kwargs)
            m_ddp = DDP(exported, device_ids=[0])
            new_outputs = m_ddp(*args, **kwargs)

            self.assertTrue(same(correct_outputs, new_outputs))

        finally:
            torch.distributed.destroy_process_group()


instantiate_parametrized_tests(TestDistributed)
if __name__ == "__main__":
    run_tests()
