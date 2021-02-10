import torch
import torchvision.models as models
import torch.fx
import torch.fx.experimental.fuser
import time

from torch.utils import mkldnn as mkldnn_utils
# g = (torch.jit.freeze(torch.jit.script(models.resnet50().eval())).graph)
# torch._C._jit_pass_convert_frozen_ops_to_mkldnn(g)
# print(g)



print('model_name', 'num_threads', 'elapsed_unfused_sec', 'elapsed_fused_sec', 'speedup_pct', sep=',')

for freeze_threads in [False, True]:
    if freeze_threads:
       torch.set_num_threads(1)
    models = [
        ('resnet18', models.resnet18()),
        ('resnet50', models.resnet50()),
        ('densenet161', models.densenet161()),
        ('inception_v3', models.inception_v3()),
        ('vgg', models.vgg16()),
        ('shufflenet', models.shufflenet_v2_x1_0()),
        ('resnext50_32x4d', models.resnext50_32x4d()),
        ('alexnet', models.alexnet()),
        ('maskrcnn', models.detection.maskrcnn_resnet50_fpn()),
        ('keypoint_rcnn', models.detection.keypointrcnn_resnet50_fpn()),
    ]

    for model_name, pt_model in models:


        fused = torch.jit.freeze(torch.jit.script(pt_model.eval()))
        rn18 = mkldnn_utils.to_mkldnn(pt_model.eval())
        # rn18 = torch.jit.freeze(rn18, optimize=False)
        # torch._C._jit_pass_convert_frozen_ops_to_mkldnn(rn18.graph)
        # rn18 = torch.jit.freeze(rn18)
        torch._C._jit_pass_convert_frozen_ops_to_mkldnn(fused.graph)
        # import pdb; pdb.set_trace()
        # print(fused.graph)
        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()

        # Example input for benchmarking TODO: sweep over batch size
        N, C, H, W, = 10, 3, 224, 224
        inp = torch.randn(N, C, H, W)
        if 'rcnn' in model_name:
            inp = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]

        # Warm-up runs. Makes sure we're not measuring one-time initialization costs
        out1 = rn18(inp)
        out2 = fused(inp)
        # print(model_name)
        # if isinstance(out1, torch.Tensor):
        #     assert torch.allclose(out1, out2, rtol=1e-05, atol=10000)
        # else:
        #     if isinstance(out1, (list, tuple)):
        #         for elem1, elem2 in zip(out1, out2):
        #             if elem1 is None:
        #                 assert elem2 is None
        #             else:
        #                 assert torch.allclose(elem1, elem2, rtol=1e-05, atol=10000)
        #     else:
        #         print(type(out1), model_name)

        # continue

        NITER = 5

        # Time unfused execution
        s = time.time()
        for _ in range(NITER):
            rn18(inp)
        e = time.time()
        elapsed_unfused_sec = (e - s) / NITER


        # Time fused execution
        s = time.time()
        for _ in range(NITER):
            fused(inp)
        e = time.time()
        elapsed_fused_sec = (e - s) / NITER

        speedup_pct = (elapsed_unfused_sec - elapsed_fused_sec) / elapsed_unfused_sec * 100

        print(model_name, torch.get_num_threads(), elapsed_unfused_sec, elapsed_fused_sec, speedup_pct, sep=',')
