import torch
from torch.utils.mobile_optimizer import optimize_for_mobile


class QD(torch.nn.Module):
    def __init__(self):
        super(QD, self).__init__()

    def forward(self, x):
        q = torch.quantize_per_tensor(x, 0.0752439, 130, torch.quint8)
        return q.dequantize()


class QD2(torch.nn.Module):
    def __init__(self):
        super(QD2, self).__init__()

    def forward(self, x, scale: float, zero: int):
        test = torch.randn(3, 5)
        q = torch.quantize_per_tensor(x, scale, zero, torch.quint8)
        return q.dequantize()

def do(model, name):
    model.eval()
    model_script = torch.jit.script(model)
    #model_opt = optimize_for_mobile(model_script)
    model_opt = model_script
    print(model_opt.graph)
    model_opt.save("{}.pt".format(name))
    model_opt._save_for_lite_interpreter("{}.ptl".format(name))

def main():
    do(QD(), "quant-dequant")
    do(QD2(), "quant-dequant-3arg")


if __name__ == '__main__':
    main()


