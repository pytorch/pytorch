import torch

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1)
        self.relu = torch.nn.ReLU()
        # self.ff = torch.nn.quantized.modules.functional_modules.FloatFunctional()

    def forward(self, x):
        # return self.ff.add(self.relu(self.conv(x)), x)
        return self.relu(self.conv(x)) + x

model_fp32 = M()

model_fp32.eval()

# model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')
model_fp32.qconfig = torch.quantization.QConfig(activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8),
                                                weight=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8))

model_fp32_fused = torch.quantization.fuse_modules(model_fp32, [['conv', 'relu']])

model_fp32_prepared = torch.quantization.prepare(model_fp32_fused)
# print(model_fp32_prepared)

input_fp32 = torch.randn(4, 1, 4, 4)
model_fp32_prepared(input_fp32)

input_fp32 = torch.randn(4, 1, 4, 4)
model_fp32_prepared(input_fp32)

model_int8 = torch.quantization.convert(model_fp32_prepared)
input_q = torch.quantize_per_tensor(input_fp32, 0.1, 0, torch.quint8)
model_int8(input_q)

input_q2 = torch.quantize_per_tensor(torch.randn(4, 1, 4, 4), 0.1, 0, torch.quint8)
out = model_int8(input_q2)

traced_model_int8 = torch.jit.trace(model_int8, (input_q,), check_trace=False)
# confirm quantized::add op
traced_out = traced_model_int8(input_q2)
assert torch.all(traced_out.int_repr() == out.int_repr())

rewritten = model_int8.rewrite()
rewritten_out = rewritten(input_q2)
assert torch.all(rewritten_out.int_repr() == out.int_repr())

scripted_rewritten = torch.jit.script(rewritten)
scripted_rewritten_out = scripted_rewritten(input_q2)
assert torch.all(scripted_rewritten_out.int_repr() == out.int_repr())

traced_rewritten = torch.jit.trace(rewritten, (input_q,), check_trace=False)
traced_rewritten_out = traced_rewritten(input_q2)
assert torch.all(traced_rewritten_out.int_repr() == out.int_repr())

# scripted = torch.jit.script(model_int8)
# scripted_out = scripted(input_q2)
# assert torch.all(scripted_out.int_repr() == out.int_repr())

# Test control flow

# class Looper(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.i2h = torch.nn.Linear(5, 5)
#         self.h2h = torch.nn.Linear(5, 5)

#     def forward(self, x):
#         h = torch.zeros(x.shape[1:])
#         for i in range(x.shape[0]):
#             i2h = self.i2h(x[0])
#             h2h = self.h2h(h)
#             h = i2h + h2h
#         return h

# l = Looper().eval()
# x = torch.randn(10, 5, 5)

# # l(x)

# l.qconfig = torch.quantization.QConfig(activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8),
#                                        weight=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8))

# l_prepared = torch.quantization.prepare(l)

# l_prepared(torch.randn(7, 5, 5))
# l_prepared(torch.randn(13, 5, 5))
