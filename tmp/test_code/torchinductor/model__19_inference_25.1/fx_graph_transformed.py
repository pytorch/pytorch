class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f16[2048, 4096]", arg1_1: "f16[4096, 512]", arg2_1: "f16[2048, 512]"):
        # File: /data/users/jezng/pytorch/test/inductor/test_cutlass_backend.py:522 in mm, code: return (a @ b) - 3.3 * c
        mm: "f16[2048, 512]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        mul: "f16[2048, 512]" = torch.ops.aten.mul.Tensor(arg2_1, 3.3);  arg2_1 = None
        sub: "f16[2048, 512]" = torch.ops.aten.sub.Tensor(mm, mul);  mm = mul = None
        return (sub,)
        