import torch
import torchvision

model_orig = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
model = model_orig
model.eval()

model = (torch.jit.script(model))
# import pdb; pdb.set_trace()
# torch._C._jit_pass_fold_convbn(model._c)
model = torch.jit.freeze(model)
import pdb; pdb.set_trace()
torch._C._jit_pass_fold_batch_conv(model.graph)
torch._C._jit_pass_lint(model.graph)
# print(model.graph)

# inp = torch.rand([N, 3, 480, 480], device=device)
# inp = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
inp = torch.ones(1, 3, 224, 224)

for m in [model_orig, model]:
    import time
    tic = time.clock()
    for _ in range(10):
        m(inp)
    toc = time.clock()
    print("HERE WE GO", toc - tic)


g = model.graph
convs = g.findAllNodes("aten::conv2d")

# def bn_folding(self, conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
#     if conv_b is None:
#         conv_b = bn_rm.new_zeros(bn_rm.shape)
#     bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

#     w_fold = conv_w * (bn_w * bn_var_rsqrt).view(-1, 1, 1, 1)
#     b_fold = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b

#     return torch.nn.Parameter(w_fold), torch.nn.Parameter(b_fold)

# # weight is second input, bias is third input
# def fold_conv_bn_eval(conv, bn):
#     assert(not (conv.training or bn.training)), "Fusion only for eval!"
#     fused_conv = copy.deepcopy(conv)

#     fused_conv.weight, fused_conv.bias = self.bn_folding(fused_conv.weight, fused_conv.bias,
#                                 bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)

#     return fused_conv
