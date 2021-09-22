# used by the benchmarking program to wrap cpu models for GPU use
import torch
from copy import deepcopy

def to_device(i, d):
    if isinstance(i, torch.Tensor):
        return i.to(device=d)
    elif isinstance(i, (tuple, list)):
        return tuple(to_device(e, d) for e in i)
    else:
        raise RuntimeError('inputs are weird')

class GPUWrapper(torch.nn.Module):
    def __init__(self, root):
        super().__init__()
        self.models = []
        self.streams = {}
        for i in range(torch.cuda.device_count()):
            m = deepcopy(root) if i != 0 else root
            d = f'cuda:{i}'
            m.to(device=d)
            self.models.append((m, d))

    def __getstate__(self):
        return self.models

    def __setstate__(self, models):
        super().__init__()
        self.models = models
        self.streams = {}
        for m, d in models:
            torch.cuda.synchronize(d)

    # roi_align, 2210 count, ROIAlign_cuda.cu: add threadsync: problem goes away, return rand problem goes away,
    # use different streams here, problem goes away.
    def forward(self, tid, *args):
        m, d = self.models[tid % len(self.models)]
        if tid not in self.streams:
            self.streams[tid] = torch.cuda.Stream(d)
        s = self.streams[tid]
        with torch.cuda.stream(s):
            iput = to_device(args, d)
            r = to_device(m(*iput), 'cpu')
            return r


if __name__ == '__main__':
    def check_close(a, b):
        if isinstance(a, (list, tuple)):
            for ae, be in zip(a, b):
                check_close(ae, be)
        else:
            print(torch.max(torch.abs(a - b)))
            assert torch.allclose(a, b)

    import sys
    from torch.package import PackageImporter
    i = PackageImporter(sys.argv[1])
    torch.version.interp = 0
    model = i.load_pickle('model', 'model.pkl')
    eg = i.load_pickle('model', 'example.pkl')
    r = model(*eg)

    gpu_model = GPUWrapper(model)
    r2 = gpu_model(*eg)
    check_close(r, r2)
