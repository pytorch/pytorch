# Owner(s): ["module: dynamo"]

import copy
import os
import sys
import tempfile

import torch
import torch.distributed as dist
from torch._guards import tracing, TracingContext
from torch._inductor._functionalize_collectives import (
    _functionalize_inplace_collectives,
    _unbox_process_group_torchbinds,
)
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.graph_module import _share_torchbind_and_process_group_on_deepcopy
from torch.fx.passes.regional_inductor import regional_inductor
from torch.testing._internal.common_utils import run_tests, TestCase


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)


def _f(t):
    t = t.clone()
    dist.all_reduce(t)
    return t + 1


def _make_fx_with_allreduce():
    return make_fx(_f)(torch.ones(4))


class TestInductorCompileCollectives(TestCase):
    def setUp(self):
        super().setUp()
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29516")
        with tempfile.NamedTemporaryFile(
            prefix="inductor_compile_collectives_store_", delete=False
        ) as fd:
            self._store_path = fd.name
        dist.init_process_group(
            backend="gloo",
            rank=0,
            world_size=1,
            store=dist.FileStore(self._store_path, 1),
        )

    def tearDown(self):
        try:
            dist.destroy_process_group()
        except AssertionError:
            pass
        if os.path.exists(self._store_path):
            os.unlink(self._store_path)
        super().tearDown()

    def test_functionalize_inplace_allreduce(self):
        gm = _make_fx_with_allreduce()
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, t_1):
    clone = torch.ops.aten.clone.default(t_1);  t_1 = None
    _torchbind_obj0 = self._torchbind_obj0
    _torchbind_obj1 = self._torchbind_obj1
    allreduce_ = torch.ops.c10d.allreduce_.default([clone], _torchbind_obj0, _torchbind_obj1, None, False);  clone = _torchbind_obj0 = _torchbind_obj1 = None
    getitem = allreduce_[0]
    getitem_1 = getitem[0];  getitem = None
    getitem_2 = allreduce_[1];  allreduce_ = getitem_2 = None
    add = torch.ops.aten.add.Tensor(getitem_1, 1);  getitem_1 = None
    return add""",
        )

        _functionalize_inplace_collectives(gm)

        # ReduceOp is baked as ``'sum'`` (its int value is read at rewrite
        # time) and its now-dead ``_torchbind_obj1`` get_attr / module attr
        # is stripped; the ProcessGroup ``get_attr`` is still referenced by
        # the new functional call, so it stays for the unbox step.
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, t_1):
    clone = torch.ops.aten.clone.default(t_1);  t_1 = None
    _torchbind_obj0 = self._torchbind_obj0
    all_reduce_default = torch.ops._c10d_functional.all_reduce.default(clone, 'sum', _torchbind_obj0);  _torchbind_obj0 = None
    wait_tensor_default = torch.ops._c10d_functional.wait_tensor.default(all_reduce_default);  all_reduce_default = None
    copy__default = torch.ops.aten.copy_.default(clone, wait_tensor_default);  clone = copy__default = None
    add = torch.ops.aten.add.Tensor(wait_tensor_default, 1);  wait_tensor_default = None
    return add""",
        )

    def test_unbox_process_group_torchbinds(self):
        # ``_unbox_process_group_torchbinds`` flips the gm attr from a
        # torchbind ScriptObject to a Python ``dist.ProcessGroup`` (the
        # form Inductor's collective lowering and runtime ops accept).
        # The FX graph itself is unchanged.
        gm = _make_fx_with_allreduce()
        _functionalize_inplace_collectives(gm)
        self.assertIsInstance(gm._torchbind_obj0, torch.ScriptObject)

        _unbox_process_group_torchbinds(gm)

        self.assertNotIsInstance(gm._torchbind_obj0, torch.ScriptObject)
        self.assertIsInstance(gm._torchbind_obj0, dist.ProcessGroup)
        x = torch.arange(4, dtype=torch.float32)
        self.assertEqual(gm(x), _f(x))

    def test_post_pass_gm_deepcopy(self):
        # After functionalize + unbox the gm holds a Python ``dist.ProcessGroup``
        # — still not pickleable, but ``_share_torchbind_and_process_group_on_deepcopy()``
        # makes the gm deepcopy-safe by sharing the PG by reference.
        # ``standalone_compile``'s own deepcopy of the gm relies on this.
        gm = _make_fx_with_allreduce()
        _functionalize_inplace_collectives(gm)
        _unbox_process_group_torchbinds(gm)
        self.assertIsInstance(gm._torchbind_obj0, dist.ProcessGroup)

        with _share_torchbind_and_process_group_on_deepcopy():
            gm2 = copy.deepcopy(gm)
        self.assertIs(gm._torchbind_obj0, gm2._torchbind_obj0)

    def test_multi_allreduce_make_fx_and_compile(self):
        # Two ``dist.all_reduce`` calls. Verify that:
        #   1) our pass on a make_fx-traced graph produces N independent
        #      ``_c10d_functional.all_reduce`` chains, and
        #   2) Dynamo + AOT autograd produce the same per-call structure
        #      (the canonical shape — no batched form).
        def g(t0, t1):
            t0 = t0.clone()
            t1 = t1.clone()
            dist.all_reduce(t0)
            dist.all_reduce(t1)
            return t0 + t1

        # (1) make_fx + our pass.
        gm = make_fx(g)(torch.ones(4), torch.ones(4) * 2)
        _functionalize_inplace_collectives(gm)
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, t0_1, t1_1):
    clone = torch.ops.aten.clone.default(t0_1);  t0_1 = None
    clone_1 = torch.ops.aten.clone.default(t1_1);  t1_1 = None
    _torchbind_obj0 = self._torchbind_obj0
    all_reduce_default = torch.ops._c10d_functional.all_reduce.default(clone, 'sum', _torchbind_obj0);  _torchbind_obj0 = None
    wait_tensor_default = torch.ops._c10d_functional.wait_tensor.default(all_reduce_default);  all_reduce_default = None
    copy__default = torch.ops.aten.copy_.default(clone, wait_tensor_default);  clone = copy__default = None
    _torchbind_obj2 = self._torchbind_obj2
    all_reduce_default_1 = torch.ops._c10d_functional.all_reduce.default(clone_1, 'sum', _torchbind_obj2);  _torchbind_obj2 = None
    wait_tensor_default_1 = torch.ops._c10d_functional.wait_tensor.default(all_reduce_default_1);  all_reduce_default_1 = None
    copy__default_1 = torch.ops.aten.copy_.default(clone_1, wait_tensor_default_1);  clone_1 = copy__default_1 = None
    add = torch.ops.aten.add.Tensor(wait_tensor_default, wait_tensor_default_1);  wait_tensor_default = wait_tensor_default_1 = None
    return add""",
        )

        # (2) torch.compile path: capture AOT fwd graph via aot_autograd.
        from functorch.compile import min_cut_rematerialization_partition
        from torch._dynamo.backends.common import aot_autograd

        captured: list[str] = []

        def fw_compiler(fx_g, _):
            captured.append(fx_g.code.strip())
            return fx_g

        backend = aot_autograd(
            fw_compiler=fw_compiler,
            partition_fn=min_cut_rematerialization_partition,
        )
        torch.compile(g, backend=backend, fullgraph=True)(
            torch.ones(4), torch.ones(4) * 2
        )
        self.assertEqual(len(captured), 1)
        self.assertExpectedInline(
            captured[0],
            """\
def forward(self, arg0_1, arg1_1):
    clone = torch.ops.aten.clone.default(arg0_1);  arg0_1 = None
    clone_1 = torch.ops.aten.clone.default(arg1_1);  arg1_1 = None
    all_reduce = torch.ops._c10d_functional.all_reduce.default(clone, 'sum', '0')
    wait_tensor = torch.ops._c10d_functional.wait_tensor.default(all_reduce);  all_reduce = None
    copy = torch.ops.aten.copy.default(clone, wait_tensor);  clone = wait_tensor = None
    all_reduce_1 = torch.ops._c10d_functional.all_reduce.default(clone_1, 'sum', '0')
    wait_tensor_1 = torch.ops._c10d_functional.wait_tensor.default(all_reduce_1);  all_reduce_1 = None
    copy_1 = torch.ops.aten.copy.default(clone_1, wait_tensor_1);  clone_1 = wait_tensor_1 = None
    add = torch.ops.aten.add.Tensor(copy, copy_1);  copy = copy_1 = None
    return (add,)""",
        )

    def test_regional_inductor_with_dist_all_reduce(self):
        # End-to-end via ``regional_inductor`` → ``standalone_compile``,
        # which now wires functionalize + unbox + the deepcopy hook.
        gm = _make_fx_with_allreduce()
        for node in gm.graph.nodes:
            if node.op not in ("placeholder", "output"):
                node.meta.setdefault("custom", {})["compile_with_inductor"] = {
                    "inductor_configs": {}
                }
        fake_mode = next(
            n.meta["val"].fake_mode
            for n in gm.graph.nodes
            if n.op == "placeholder" and isinstance(n.meta.get("val"), torch.Tensor)
        )

        with tracing(TracingContext(fake_mode)):
            compiled_gm = regional_inductor(gm)

        x = torch.arange(4, dtype=torch.float32)
        self.assertEqual(compiled_gm([x]), _f(x))


if __name__ == "__main__":
    run_tests()
