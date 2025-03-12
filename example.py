# https://github.com/Xilinx/brevitas/blob/4617f7bd136e96fa21c7f76e3c7e2e37fe563837/src/brevitas/export/onnx/qonnx/function.py#L36C1-L47C19
import torch
import torch.onnx.ops


# def symbolic(g, x, scale, zero_point, bit_width, narrow_range, signed, rounding_mode):
#     ret = g.op(
#         f'{DOMAIN_STRING}::Quant',
#         x,
#         scale,
#         zero_point,
#         bit_width,
#         rounding_mode_s=rounding_mode,
#         signed_i=int(signed),
#         narrow_i=int(narrow_range))
#     ret.setType(x.type())
#     return ret
# def forward(ctx, x, scale, zero_point, bit_width, narrow_range, signed, rounding_mode):
#     float_to_int_impl = solve_float_to_int_impl_from_enum(rounding_mode)
#     quant = IntQuant(
#         float_to_int_impl=float_to_int_impl(),
#         tensor_clamp_impl=TensorClamp(),
#         input_view_impl=Identity(),  #TODO: Update this when QONNX support Groupwise export
#         narrow_range=narrow_range,
#         signed=signed)
#     y = quant(scale, zero_point, bit_width, x)
#     return y

DOMAIN_STRING = "onnx.brevitas"


class Quant(torch.nn.Module):
    def forward(
        self,
        x,
        scale,
        zero_point,
        bit_width,
        narrow_range: bool,
        signed: bool,
        rounding_mode: str,
    ):
        if torch.onnx.is_in_onnx_export():
            return torch.onnx.ops.symbolic(
                f"{DOMAIN_STRING}Quant",
                (x, scale, zero_point, bit_width),
                dict(
                    signed=signed,
                    narrow_range=narrow_range,
                    rounding_mode=rounding_mode,
                ),
                dtype=x.dtype,
                shape=x.shape,
            )
        else:
            float_to_int_impl = solve_float_to_int_impl_from_enum(rounding_mode)
            quant = IntQuant(
                float_to_int_impl=float_to_int_impl(),
                tensor_clamp_impl=TensorClamp(),
                input_view_impl=Identity(),
                narrow_range=narrow_range,
                signed=signed,
            )
            y = quant(scale, zero_point, bit_width, x)
            return y


# https://github.com/microsoft/Olive/blob/7df3c0353b3b754dec23faee2d73f22af9526155/olive/passes/pytorch/tensor_parallel_layers.py#L19C1-L29C69

# class AllReduce(torch.autograd.Function):  # pylint: disable=abstract-method
#     @staticmethod
#     def forward(ctx, x: torch.Value) -> torch.Value:  # pylint: disable=arguments-differ
#         if torch.onnx.is_in_onnx_export():
#             return x
#         dist.all_reduce(x, op=dist.ReduceOp.SUM)
#         return x

#     @staticmethod
#     def symbolic(g: torch.Graph, x: torch.Value) -> torch.Value:
#         return g.op("com.microsoft::AllReduce", x).setType(x.type())


class AllReduce(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.onnx.is_in_onnx_export():
            return torch.onnx.ops.symbolic(
                "com.microsoft::AllReduce",
                (x,),
                {},
                dtype=x.dtype,
                shape=x.shape,
            )
        torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.SUM)
        return x


# https://github.com/microsoft/Olive/blob/7df3c0353b3b754dec23faee2d73f22af9526155/olive/common/hf/quant.py#L122

# class QuantLinearTorchFunction(torch.autograd.Function):
#     """Used to export the quantized linear layer to onnx using the contrib operator MatMulNBits."""

#     @staticmethod
#     def symbolic(g, x, qweight, scales, qzeros, g_idx, bits, group_size, in_features, out_features):
#         tensor_args = [x, qweight, scales, qzeros]
#         if g_idx is not None:
#             tensor_args.append(g_idx)
#         attrs = {
#             "K_i": in_features,
#             "N_i": out_features,
#             "bits_i": bits,
#             "block_size_i": group_size,
#         }

#         output = g.op(
#             "com.microsoft::MatMulNBits",
#             *tensor_args,
#             # what does this outputs do?
#             outputs=1,
#             **attrs,
#         )
#         input_shape = x.type().varyingSizes()
#         if input_shape is not None and hasattr(x.type(), "with_sizes"):
#             output_type = x.type().with_sizes(input_shape[:-1] + [qweight.type().varyingSizes()[0]])
#             output.setType(output_type)

#         return output

#     @staticmethod
#     def forward(ctx, x, qweight, scales, qzeros, g_idx, bits, group_size, in_features, out_features):
#         if torch.onnx.is_in_onnx_export():
#             return torch.zeros(x.shape[:-1] + (out_features,), dtype=x.dtype, device=x.device)
#         raise NotImplementedError("QuantLinearTorchFunction forward is only implemented for onnx export")


class QuantLinearTorchFunction(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        qweight,
        scales,
        qzeros,
        g_idx,
        bits: int,
        group_size: int,
        in_features: int,
        out_features: int,
    ):
        if torch.onnx.is_in_onnx_export():
            tensor_args = [x, qweight, scales, qzeros]
            if g_idx is not None:
                tensor_args.append(g_idx)
            return torch.onnx.ops.symbolic(
                "com.microsoft::QuantLinear",
                tensor_args,
                dict(
                    bits=bits,
                    block_size=group_size,
                    K=in_features,
                    N=out_features,
                ),
                dtype=x.dtype,
                shape=(*x.shape[:-1], out_features),
            )
        raise NotImplementedError(
            "QuantLinearTorchFunction forward is only implemented for onnx export"
        )


from torch.onnx.ops._impl import _symbolic


class TestModule(torch.nn.Module):
    def forward(self, x):
        return _symbolic(
            [x],
            "Add",
            1,
            [torch.tensor(42)],
            shape=x.size(),
            attr_keys=["key"],
            attr_types=["i"],
            attr_pos=[(0, 1)],
            attr_ints=[1],
            attr_floats=[1.0],
            attr_strs=["attr"],
            attr_bools=[True],
            metadata_props_keys=["meta_key"],
            metadata_props_values=["meta_value"],
            domain="com.microsoft",
            version=1,
        )


batch = torch.export.Dim("batch")
ep = torch.export.export(
    TestModule(), (torch.ones(2, 1),), dynamic_shapes=({0: batch},), strict=False
)
print(ep)

# ExportedProgram:
#     class GraphModule(torch.nn.Module):
#         def forward(self, c_lifted_tensor_0: "i64[]", x: "f32[s0, 1]"):
#              #
#             sym_size_int_1: "Sym(s0)" = torch.ops.aten.sym_size.int(x, 0)

#              # File: /home/justinchu/dev/pytorch/example.py:174 in forward, code: [torch.tensor(42)],
#             lift_fresh_copy: "i64[]" = torch.ops.aten.lift_fresh_copy.default(c_lifted_tensor_0);  c_lifted_tensor_0 = None
#             detach_: "i64[]" = torch.ops.aten.detach_.default(lift_fresh_copy);  lift_fresh_copy = None

#              # File: /home/justinchu/dev/pytorch/example.py:170 in forward, code: return _symbolic(
#             _symbolic: "f32[s0, 1]" = torch.ops.onnx_symbolic._symbolic.default([x], 'Add', 1, [detach_], shape = [sym_size_int_1, 1], attr_keys = ['key'], attr_ints = [1], attr_floats = [1.0], attr_strs = ['attr'], attr_bools = [True], metadata_props_keys = ['meta_key'], metadata_props_values = ['meta_value'], domain = 'com.microsoft', version = 1);  x = detach_ = sym_size_int_1 = None
#             return (_symbolic,)
