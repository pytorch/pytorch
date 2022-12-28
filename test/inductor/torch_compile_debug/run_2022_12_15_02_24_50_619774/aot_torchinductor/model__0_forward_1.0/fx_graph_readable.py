class GraphModule(torch.nn.Module):
    def forward(self, primals_1: f32[]):
        # File: /home/xiaobing/pytorch-offical/torch/testing/_internal/common_methods_invocations.py:14444, code: op=lambda x, *args, **kwargs: x.half(*args, **kwargs),
        convert_element_type: f16[] = torch.ops.prims.convert_element_type.default(primals_1, torch.float16);  primals_1 = None
        return [convert_element_type]
        