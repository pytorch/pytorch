"""
Generate the example files that torchpy_test uses.
"""
from pathlib import Path
import torch
import argparse

from torch.package import PackageExporter

try:
    from .examples import Simple, resnet18
except ImportError:
    from examples import Simple, resnet18

def save(name, model, model_jit, eg):
    with PackageExporter(str(p / name)) as e:
        e.mock('iopath.**')
        e.save_pickle('model', 'model.pkl', model)
        e.save_pickle('model', 'example.pkl', eg)
    model_jit.save(str(p / (name + '_jit')))


parser = argparse.ArgumentParser(description="Generate Examples")
parser.add_argument("--install_dir", help="Root directory for all output files")
parser.add_argument("--fbcode_dir", help="fbcode passes this to all binaries, so we accept it")


if __name__ == "__main__":
    args = parser.parse_args()
    if args.install_dir is None:
        p = Path(__file__).parent / "generated"
        p.mkdir(exist_ok=True)
    else:
        p = Path(args.install_dir)

    resnet = resnet18()
    resnet.eval()
    resnet_eg = torch.rand(1, 3, 224, 224)
    resnet_traced = torch.jit.trace(resnet, resnet_eg)
    save('resnet', resnet, resnet_traced, (resnet_eg,))

    simple = Simple(10, 20)
    save('simple', simple, torch.jit.script(simple), (torch.rand(10, 20),))
