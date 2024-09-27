# Owner(s): ["oncall: export"]

import torch
from torch.export import Dim
from torch.export._draft_export import draft_export
from torch.testing._internal.common_utils import run_tests, TestCase


class TestDraftExport(TestCase):
    def test_missing_meta_kernel(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo",
                "(Tensor a, Tensor b) -> Tensor",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo", "cpu", lib=lib)
            def foo_impl(a, b):
                return a + b

            # @torch.library.register_fake("mylib::foo")
            # def mylib_foo_default_fake(*args, **kwargs):
            #     ctx = torch.library.get_ctx()
            #     fake_shape = [ctx.new_dynamic_size() for _ in range(2)]
            #     return torch.empty(fake_shape, dtype=torch.float32, device="cpu")

            class M(torch.nn.Module):
                def forward(self, a, b):
                    res = torch.ops.mylib.foo(a, b)
                    return res

            inp = (torch.ones(3, 3), torch.ones(3, 3))

            ep, report = draft_export(
                M(), inp, fake_tensor_propagate_real_tensors=False
            )
            print(ep)
            print(report)

    def test_data_dependent_failure(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo1",
                "(Tensor a, Tensor b) -> Tensor",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo1", "cpu", lib=lib)
            def foo_impl(a, b):
                return a + b

            @torch.library.register_fake("mylib::foo1")
            def mylib_foo_default_fake(*args, **kwargs):
                ctx = torch.library.get_ctx()
                fake_shape = [ctx.new_dynamic_size() for _ in range(2)]
                return torch.empty(fake_shape, dtype=torch.float32, device="cpu")

            class M(torch.nn.Module):
                def forward(self, a, b, c):
                    res = torch.ops.mylib.foo1(a, b)

                    c_item = c.item()
                    return res[:c_item]

            inp = (torch.ones(3, 3), torch.ones(3, 3), torch.tensor(3))

            ep, report = draft_export(M(), inp)
            print(ep)
            print(report)

    def test_shape_failure(self):
        class M(torch.nn.Module):
            def forward(self, a):
                assert a.shape[0] == 3
                return a * a

        inp = (torch.ones(3, 3),)

        ep, report = draft_export(M(), inp, dynamic_shapes={"a": {0: Dim("a0")}})
        print(ep)
        print(report)


if __name__ == "__main__":
    run_tests()
