# Owner(s): ["oncall: distributed"]

import copy
import gc
import sys
import weakref

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    TEST_SKIPS,
)
from torch.testing._internal.common_utils import parametrize, run_tests
from torch.utils.checkpoint import checkpoint


class HiddenOutput:
    def __init__(self, tensor):
        self.tensor = tensor


class RecoveryModel(nn.Module):
    def __init__(self, checkpoint_live=False):
        super().__init__()
        self.trunk = nn.Linear(8, 8)
        self.live = nn.Linear(8, 8)
        self.dead = nn.Linear(8, 8)
        self.checkpoint_live = checkpoint_live

    def forward(self, x):
        h = self.trunk(x).relu()
        dead = self.dead(h)
        if self.checkpoint_live:
            live = checkpoint(self.live, h, use_reentrant=True)
            live = live + checkpoint(self.live, 2 * h, use_reentrant=True)
        else:
            live = self.live(h)
        return live, HiddenOutput(dead)


class CheckpointedBranches(nn.Module):
    def __init__(self):
        super().__init__()
        self.live = nn.Linear(8, 8)
        self.dead = nn.Linear(8, 8)

    def forward(self, x):
        live = checkpoint(self.live, x, use_reentrant=True)
        return live, HiddenOutput(checkpoint(self.dead, x, use_reentrant=True))


class CPUOutputCheckpoints(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)

    def forward(self, x):
        def run(x):
            return self.linear(x).cpu()

        live = checkpoint(run, x, use_reentrant=True)
        return live + checkpoint(run, 2 * x, use_reentrant=True)


class ReentrantBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, module, fail):
        ctx.save_for_backward(x)
        ctx.module = module
        ctx.fail = fail
        with torch.no_grad():
            return module(x)

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        x = x.detach().requires_grad_()
        with torch.enable_grad():
            torch.autograd.backward(ctx.module(x), grad)
        if ctx.fail[0]:
            raise RuntimeError("outer backward failed")
        return x.grad, None, None


class FailingReentrant(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)
        self.fail = [False]
        self.last_output = None

    def forward(self, x):
        # Lets the test run backward without DDP's output sink.
        self.last_output = ReentrantBackward.apply(x, self.linear, self.fail)
        return self.last_output


class TestDDPStaticGraphRecovery(MultiProcContinuousTest):
    world_size = 2

    @classmethod
    def backend_str(cls):
        return dist.get_default_backend_for_device(cls.device_type)

    @classmethod
    def _init_pg(cls, rank, world_size, rdvz_file):
        if cls.device_type != "cpu":
            device_module = torch.get_device_module(cls.device_type)
            if device_module.device_count() < world_size:
                sys.exit(TEST_SKIPS[f"multi-device-{world_size}"].exit_code)
            device_module.set_device(rank)
        super()._init_pg(rank, world_size, rdvz_file)

    def _rank_device(self, device):
        # Device arguments are fixed before workers receive their ranks.
        if self.device_type == "cpu":
            return device
        return f"{self.device_type}:{self.rank}"

    def _input(self, device, rank, step=0):
        return (torch.arange(32, device=device).reshape(4, 8) + rank + step) / 16

    def _assert_grads(self, model, reference):
        for param, expected in zip(model.parameters(), reference.parameters()):
            self.assertEqual(param.grad, expected.grad)

    @parametrize("gradient_as_bucket_view", [False, True])
    @parametrize("asymmetric", [False, True])
    @parametrize("manual", [False, True])
    @torch._dynamo.config.patch(
        optimize_ddp=False, recompile_limit=1, skip_nnmodule_hook_guards=False
    )
    def test_compiled_eager_fallback(
        self, device, gradient_as_bucket_view, asymmetric, manual
    ):
        device = self._rank_device(device)
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)
        model = RecoveryModel().to(device)
        reference = copy.deepcopy(model)
        ddp = DDP(
            model,
            static_graph=True,
            bucket_cap_mb=0.0003,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )
        ddp._set_static_graph_recovery()
        ddp.require_manual_backward_finalization = manual
        compiled = torch.compile(ddp, backend="aot_eager")
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        ref_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.01)
        fired = []
        model.dead.weight.register_post_accumulate_grad_hook(lambda _: fired.append(1))
        hook = None

        for step in range(4):
            falls_back = step == 2 and (not asymmetric or self.rank == 0)
            if falls_back:
                hook = model.live.register_forward_hook(lambda mod, args, out: None)
            elif hook is not None:
                hook.remove()
                hook = None
            optimizer.zero_grad()
            ref_optimizer.zero_grad()
            fired.clear()
            compiled(self._input(device, self.rank, step))[0].sum().backward()
            if manual:
                self.assertTrue(ddp.should_finalize_after_backward)
                ddp.finalize_backward()
            self.assertEqual(len(fired), int(not falls_back))
            inputs = torch.cat(
                [self._input(device, rank, step) for rank in range(self.world_size)]
            )
            live, dead = reference(inputs)
            ((live.sum() + 0 * dead.tensor.sum()) / self.world_size).backward()
            self._assert_grads(model, reference)
            ddp._check_reducer_finalized()
            optimizer.step()
            ref_optimizer.step()
            self.assertEqual(model.state_dict(), reference.state_dict())
            self.assertEqual(optimizer.state_dict(), ref_optimizer.state_dict())

    @torch._dynamo.config.patch(
        optimize_ddp=True, recompile_limit=1, skip_nnmodule_hook_guards=False
    )
    def test_ddp_optimizer_fallback(self, device):
        device = self._rank_device(device)
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)
        model = RecoveryModel().to(device)
        reference = copy.deepcopy(model)
        ddp = DDP(model, static_graph=True)
        ddp._set_static_graph_recovery()
        compiled = torch.compile(ddp, backend="aot_eager")
        for step in range(3):
            if step == 2:
                model.live.register_forward_hook(lambda mod, args, out: None)
            ddp.zero_grad()
            reference.zero_grad()
            compiled(self._input(device, self.rank, step))[0].sum().backward()
            inputs = torch.cat(
                [self._input(device, rank, step) for rank in range(self.world_size)]
            )
            live, dead = reference(inputs)
            ((live.sum() + 0 * dead.tensor.sum()) / self.world_size).backward()
            self._assert_grads(model, reference)
            ddp._check_reducer_finalized()

    @parametrize("gradient_as_bucket_view", [False, True])
    def test_accumulated_grad(self, device, gradient_as_bucket_view):
        device = self._rank_device(device)
        model = RecoveryModel().to(device)
        reference = copy.deepcopy(model)
        ddp = DDP(
            model,
            static_graph=True,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )
        ddp._set_static_graph_recovery()
        for _ in range(2):
            ddp.zero_grad()
            live, dead = ddp(self._input(device, self.rank))
            (live.sum() + 0 * dead.tensor.sum()).backward()
        ddp.zero_grad()

        with ddp.no_sync():
            live, dead = ddp(self._input(device, self.rank))
            (live.sum() + dead.tensor.sum()).backward()
        ddp(self._input(device, self.rank, 1))[0].sum().backward()
        for step in range(2):
            inputs = torch.cat(
                [self._input(device, rank, step) for rank in range(self.world_size)]
            )
            live, dead = reference(inputs)
            loss = live.sum() + (dead.tensor.sum() if step == 0 else 0)
            (loss / self.world_size).backward()
        self._assert_grads(model, reference)
        ddp._check_reducer_finalized()

    @parametrize("gradient_as_bucket_view", [False, True])
    def test_reentrant_checkpoint(self, device, gradient_as_bucket_view):
        device = self._rank_device(device)
        model = RecoveryModel(checkpoint_live=True).to(device)
        reference = copy.deepcopy(model)
        ddp = DDP(
            model,
            static_graph=True,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )
        ddp._set_static_graph_recovery()
        for step in range(3):
            ddp.zero_grad()
            reference.zero_grad()
            live, dead = ddp(self._input(device, self.rank, step))
            loss = live.sum() + (0 * dead.tensor.sum() if step == 0 else 0)
            loss.backward()
            inputs = torch.cat(
                [self._input(device, rank, step) for rank in range(self.world_size)]
            )
            live, dead = reference(inputs)
            ((live.sum() + 0 * dead.tensor.sum()) / self.world_size).backward()
            self._assert_grads(model, reference)
            ddp._check_reducer_finalized()

    @parametrize("checkpointed", [False, True])
    @parametrize("asymmetric", [False, True])
    def test_hidden_output_only_loss(self, device, asymmetric, checkpointed):
        device = self._rank_device(device)
        model = (CheckpointedBranches() if checkpointed else RecoveryModel()).to(device)
        reference = copy.deepcopy(model)
        ddp = DDP(model, static_graph=True)
        ddp._set_static_graph_recovery()
        for step in range(3):
            ddp.zero_grad()
            reference.zero_grad()
            uses_live = [
                step < 2 or (asymmetric and rank != 0)
                for rank in range(self.world_size)
            ]
            x = self._input(device, self.rank, step).requires_grad_()
            live, dead = ddp(x)
            loss = dead.tensor.sum()
            if uses_live[self.rank]:
                loss = loss + live.sum()
            loss.backward()
            for rank in range(self.world_size):
                x = self._input(device, rank, step).requires_grad_()
                live, dead = reference(x)
                loss = uses_live[rank] * live.sum() + dead.tensor.sum()
                (loss / self.world_size).backward()
            self._assert_grads(model, reference)
            ddp._check_reducer_finalized()

    def test_reentrant_backward_ends_on_other_device(self, device):
        device = self._rank_device(device)
        if self.device_type == "cpu":
            self.skipTest("Needs an accelerator.")
        model = CPUOutputCheckpoints().to(device)
        reference = copy.deepcopy(model)
        ddp = DDP(model, static_graph=True)
        ddp._set_static_graph_recovery()
        for step in range(3):
            ddp.zero_grad()
            reference.zero_grad()
            ddp(self._input(device, self.rank, step).requires_grad_()).sum().backward()
            for rank in range(self.world_size):
                x = self._input(device, rank, step).requires_grad_()
                (reference(x).sum() / self.world_size).backward()
            self._assert_grads(model, reference)
            ddp._check_reducer_finalized()

    def test_deferred_hook_after_reducer_deleted(self, device):
        device = self._rank_device(device)
        model = FailingReentrant().to(device)
        ddp = DDP(model, static_graph=True)
        ddp._set_static_graph_recovery()
        x = self._input(device, self.rank).requires_grad_()
        ddp(x).sum().backward()
        model.fail[0] = True
        ddp(x)
        # Skip the output sink so the inner backward installs the deferred hook.
        with self.assertRaisesRegex(RuntimeError, "outer backward failed"):
            model.last_output.sum().backward(retain_graph=True)
        ddp_ref = weakref.ref(ddp)
        del ddp
        gc.collect()
        self.assertIsNone(ddp_ref())
        model.fail[0] = False
        model.last_output.sum().backward()

    def test_comm_hook_once_per_bucket(self, device):
        device = self._rank_device(device)
        model = RecoveryModel().to(device)
        ddp = DDP(model, static_graph=True, bucket_cap_mb=0.0003)
        ddp._set_static_graph_recovery()
        calls = []

        def allreduce_hook(state, bucket):
            calls.append(bucket.index())
            buf = bucket.buffer().div_(self.world_size)
            work = dist.all_reduce(buf, async_op=True)
            return work.get_future().then(lambda future: future.value()[0])

        ddp.register_comm_hook(None, allreduce_hook)
        for step in range(4):
            ddp.zero_grad()
            calls.clear()
            live, dead = ddp(self._input(device, self.rank))
            (live.sum() + (0 * dead.tensor.sum() if step < 2 else 0)).backward()
            num_buckets = len(ddp.reducer._get_zeros_like_grad_buckets())
            self.assertEqual(len(calls), num_buckets)
            if step > 0:
                self.assertEqual(calls, list(range(len(calls))))
            if step > 1:
                self.assertGreater(len(calls), 1)
            ddp._check_reducer_finalized()

    @torch._dynamo.config.patch(optimize_ddp=False)
    def test_default_stays_strict(self, device):
        device = self._rank_device(device)
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)
        ddp = DDP(RecoveryModel().to(device), static_graph=True)
        compiled = torch.compile(ddp, backend="aot_eager")
        for _ in range(2):
            ddp.zero_grad()
            compiled(self._input(device, self.rank))[0].sum().backward()
        ddp.zero_grad()
        ddp(self._input(device, self.rank))[0].sum().backward()
        error = "Expected to have finished reduction"
        with self.assertRaisesRegex(RuntimeError, error):
            ddp._check_reducer_finalized()

    @torch._dynamo.config.patch(optimize_ddp=False)
    def test_unused_becomes_used_stays_strict(self, device):
        device = self._rank_device(device)
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)
        ddp = DDP(RecoveryModel().to(device), static_graph=True)
        ddp._set_static_graph_recovery()
        for _ in range(2):
            ddp.zero_grad()
            ddp(self._input(device, self.rank))[0].sum().backward()
        ddp.zero_grad()
        compiled = torch.compile(ddp, backend="aot_eager")
        with self.assertRaisesRegex(RuntimeError, "Your training graph has changed"):
            compiled(self._input(device, self.rank))[0].sum().backward()

    def test_requires_static_graph_before_forward(self, device):
        device = self._rank_device(device)
        model = RecoveryModel().to(device)
        ddp = DDP(model)
        with self.assertRaisesRegex(RuntimeError, "before the first forward"):
            ddp._set_static_graph_recovery()
        ddp = DDP(model, static_graph=True)
        live, dead = ddp(self._input(device, self.rank))
        (live.sum() + 0 * dead.tensor.sum()).backward()
        with self.assertRaisesRegex(RuntimeError, "before the first forward"):
            ddp._set_static_graph_recovery()

    @parametrize("feature", ["sparse", "delayed", "skip"])
    def test_rejects_unsupported_features(self, device, feature):
        device = self._rank_device(device)
        model = RecoveryModel()
        if feature == "sparse":
            model = nn.Embedding(8, 8, sparse=True)
        model = model.to(device)
        options = {}
        if feature == "delayed":
            options["delay_all_reduce_named_params"] = [
                ("dead.weight", model.dead.weight),
                ("dead.bias", model.dead.bias),
            ]
            options["param_to_hook_all_reduce"] = model.live.weight
        elif feature == "skip":
            options["skip_all_reduce_unused_params"] = True
        ddp = DDP(model, static_graph=True, bucket_cap_mb=25, **options)
        with self.assertRaisesRegex(RuntimeError, "does not support"):
            ddp._set_static_graph_recovery()

    @parametrize("leaf_output", [False, True])
    def test_no_parameter_hooks(self, device, leaf_output):
        device = self._rank_device(device)
        model = nn.Linear(8, 8).to(device)
        ddp = DDP(model, static_graph=True)
        ddp._set_static_graph_recovery()
        ddp(self._input(device, self.rank)).sum().backward()
        model.forward = (lambda x: x) if leaf_output else (lambda x: x.sin())
        x = self._input(device, self.rank).requires_grad_()
        for _ in range(2):
            ddp.zero_grad()
            ddp(x).sum().backward()
            for param in model.parameters():
                self.assertEqual(param.grad, torch.zeros_like(param))
            ddp._check_reducer_finalized()

    def test_copy_preserves_recovery(self, device):
        device = self._rank_device(device)
        ddp = DDP(RecoveryModel().to(device), static_graph=True)
        ddp._set_static_graph_recovery()
        ddp = copy.deepcopy(ddp)
        for step in range(3):
            ddp.zero_grad()
            live, dead = ddp(self._input(device, self.rank))
            (live.sum() + (0 * dead.tensor.sum() if step < 2 else 0)).backward()
            weight = ddp.module.dead.weight
            self.assertEqual(weight.grad, torch.zeros_like(weight))
            ddp._check_reducer_finalized()

    def test_cuda_graph_replay(self, device):
        device = self._rank_device(device)
        if self.device_type != "cuda":
            self.skipTest("CUDA graph capture requires CUDA.")
        model = RecoveryModel().to(device)
        reference = copy.deepcopy(model)
        x = self._input(device, self.rank)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            ddp = DDP(model, static_graph=True, gradient_as_bucket_view=True)
            ddp._set_static_graph_recovery()
            # Finish DDP's initial timing samples before capture.
            for _ in range(12):
                ddp.zero_grad(set_to_none=False)
                live, dead = ddp(x)
                (live.sum() + 0 * dead.tensor.sum()).backward()
        torch.cuda.current_stream().wait_stream(stream)
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            ddp.zero_grad(set_to_none=False)
            ddp(x)[0].sum().backward()
        torch.cuda.current_stream().wait_stream(stream)
        for step in range(3):
            x.copy_(self._input(device, self.rank, step))
            graph.replay()
            reference.zero_grad()
            inputs = torch.cat(
                [self._input(device, rank, step) for rank in range(self.world_size)]
            )
            live, dead = reference(inputs)
            ((live.sum() + 0 * dead.tensor.sum()) / self.world_size).backward()
            self._assert_grads(model, reference)
        ddp._check_reducer_finalized()


instantiate_device_type_tests(
    TestDDPStaticGraphRecovery,
    globals(),
    only_for=("cpu", "cuda", "xpu"),
    allow_xpu=True,
)


if __name__ == "__main__":
    run_tests()
