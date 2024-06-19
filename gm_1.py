class GraphModule(torch.nn.Module):
    def forward(self, primals_4: "i64[32, 1024]", view: "bf16[32768, 768]", addmm_default: "bf16[32768, 50264]", amax: "f32[32768, 1]", log: "f32[32768, 1]", convert_element_type_5: "bf16[]", tangents_1: "bf16[]"):
         # File: /home/shunting/ws/pytorch/t.py:29 in f, code: ce(model(x).view(-1, V), label.view(-1)).backward()
        div_1: "bf16[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type_5);  tangents_1 = convert_element_type_5 = None
        view_3: "i64[32768]" = torch.ops.aten.reshape.default(primals_4, [-1]);  primals_4 = None
        unsqueeze_1: "i64[32768, 1]" = torch.ops.aten.unsqueeze.default(view_3, 1);  view_3 = None
        ne_3: "b8[32768, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_1, -100)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "i64[32768, 1]" = torch.ops.aten.where.self(ne_3, unsqueeze_1, full_default);  unsqueeze_1 = full_default = None

        full_default_3: "bf16[32768, 50257]" = torch.ops.aten.full.default([32768, 50257], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        scatter: "bf16[32768, 50257]" = torch.ops.aten.scatter.value(full_default_3, 1, where_2, -1.0);  full_default_3 = where_2 = None

        full_default_1: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "bf16[32768, 1]" = torch.ops.aten.where.self(ne_3, div_1, full_default_1);  ne_3 = div_1 = full_default_1 = None

        mul: "bf16[32768, 50257]" = torch.ops.aten.mul.Tensor(scatter, where_3);  scatter = where_3 = None

        convert_element_type_6: "f32[32768, 50257]" = torch.ops.prims.convert_element_type.default(mul, torch.float32);  mul = None
        
        # No stacktrace found for following nodes
        slice_tensor_1: "bf16[32768, 50257]" = torch.ops.aten.slice.Tensor(addmm_default, 1, 0, -7);  addmm_default = None
        
         # File: /home/shunting/ws/pytorch/t.py:22 in forward, code: return self.linear(x)
        view_1: "bf16[32, 1024, 50257]" = torch.ops.aten.reshape.default(slice_tensor_1, [32, 1024, 50257]);  slice_tensor_1 = None
        
         # File: /home/shunting/ws/pytorch/t.py:29 in f, code: ce(model(x).view(-1, V), label.view(-1)).backward()
        view_2: "bf16[32768, 50257]" = torch.ops.aten.reshape.default(view_1, [-1, 50257]);  view_1 = None
        convert_element_type_3: "f32[32768, 50257]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        sub: "f32[32768, 50257]" = torch.ops.aten.sub.Tensor(convert_element_type_3, amax);  convert_element_type_3 = amax = None
        sub_1: "f32[32768, 50257]" = torch.ops.aten.sub.Tensor(sub, log);  sub = log = None
        convert_element_type_4: "bf16[32768, 50257]" = torch.ops.prims.convert_element_type.default(sub_1, torch.bfloat16);  sub_1 = None
        convert_element_type_7: "f32[32768, 50257]" = torch.ops.prims.convert_element_type.default(convert_element_type_4, torch.float32);  convert_element_type_4 = None
        exp_1: "f32[32768, 50257]" = torch.ops.aten.exp.default(convert_element_type_7);  convert_element_type_7 = None
        sum_4: "f32[32768, 1]" = torch.ops.aten.sum.dim_IntList(convert_element_type_6, [1], True)
        mul_1: "f32[32768, 50257]" = torch.ops.aten.mul.Tensor(exp_1, sum_4);  exp_1 = sum_4 = None

        sub_2: "f32[32768, 50257]" = torch.ops.aten.sub.Tensor(convert_element_type_6, mul_1);  convert_element_type_6 = mul_1 = None

        convert_element_type_8: "bf16[32768, 50257]" = torch.ops.prims.convert_element_type.default(sub_2, torch.bfloat16);  sub_2 = None
        view_4: "bf16[32, 1024, 50257]" = torch.ops.aten.reshape.default(convert_element_type_8, [32, 1024, 50257]);  convert_element_type_8 = None
        
         # File: /home/shunting/ws/pytorch/t.py:22 in forward, code: return self.linear(x)
        view_5: "bf16[32768, 50257]" = torch.ops.aten.reshape.default(view_4, [32768, 50257]);  view_4 = None
        permute_1: "bf16[50257, 32768]" = torch.ops.aten.permute.default(view_5, [1, 0])
        
        # No stacktrace found for following nodes
        constant_pad_nd_default: "bf16[50264, 32768]" = torch.ops.aten.constant_pad_nd.default(permute_1, [0, 0, 0, 7]);  permute_1 = None
        mm_default: "bf16[50264, 768]" = torch.ops.aten.mm.default(constant_pad_nd_default, view);  constant_pad_nd_default = view = None
        slice_tensor: "bf16[50257, 768]" = torch.ops.aten.slice.Tensor(mm_default, 0, 0, -7);  mm_default = None
        
         # File: /home/shunting/ws/pytorch/t.py:22 in forward, code: return self.linear(x)
        permute_2: "bf16[768, 50257]" = torch.ops.aten.permute.default(slice_tensor, [1, 0]);  slice_tensor = None
        sum_5: "bf16[1, 50257]" = torch.ops.aten.sum.dim_IntList(view_5, [0], True);  view_5 = None
        view_6: "bf16[50257]" = torch.ops.aten.reshape.default(sum_5, [50257]);  sum_5 = None
        permute_3: "bf16[50257, 768]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        return [permute_3, view_6, None, None]
        