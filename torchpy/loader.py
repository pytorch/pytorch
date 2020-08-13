import os
import pickle
import sys
import torch
import types
import zipfile

from torch import Tensor  # noqa

# https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/docs/serialization.md

# Hack
class Module:
    pass


class Loader(types.ModuleType):
    def __init__(self, base, zf, globals_, name=""):
        super().__init__(name)
        self.base = base
        self.zf = zf
        self.globals = globals_.copy() if globals_ is not None else globals()
        self.fresh_module = False
        self.compiled = False
        self.unmangle = ['_ResNet', '_Sequential', '_BasicBlock', '_Conv2d']

        # special case for now
        if name == "__torch__":
            self.globals['__torch__'] = self
            filename = self.base + '.py'
            if filename in self.zf.namelist():
                f = self.zf.read(filename)
                self.fresh_module = True
                exec(compile(f, name, 'exec'), self.globals) 
                self.fresh_module = False

    def __getattr__(self, name):
        for prefix in self.unmangle:
            if name.find(prefix) == 0:
                name = name[len(prefix):]
        dirname = os.path.join(self.base, name)
        filename = dirname + '.py'
        # print("{} -> {}".format(self.base, name))

        if name in self.globals and name != 'torch':
            # Stuff in the module by this name takes precedence, then subdirs
            # print("   Return existing globals for {}".format(name))
            return self.globals[name]

        if filename in self.zf.namelist() and not self.compiled:
            # If the filename names a module, compile it
            #   - the same name can also be a directory leading to other modules
            ldr = Loader(dirname, self.zf, self.globals)
            ldr.fresh_module = True
            setattr(self, name, ldr)  # Temporary class to allow annotations to work
            self.globals[name] = ldr
            f = self.zf.read(filename)
            # print("   Compile: ", name)
            exec(compile(f, name, 'exec'), ldr.globals)
            ldr.fresh_module = False
            ldr.compiled = True
            # print("   Return finished loader for module {}".format(name))
            return ldr

        if any([dirname in x for x in self.zf.namelist()]):
            attr = Loader(dirname, self.zf, self.globals)
            setattr(self, name, attr)
            # print("   Return loader for dir {}".format(name))
            return attr

        if self.fresh_module:
            # print("   Return Dummy Class for {}".format(name))
            return types.new_class("Dummy Class")

        raise AttributeError("{} not found in {}".format(name, self.base))


class TorchUnpickler(pickle.Unpickler):
    def __init__(self, base_name, zipfile, resolver, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resolver = resolver
        self.zf = zipfile
        self.base_name = base_name

    def persistent_load(self, id):
        if 'storage' == id[0]:
            _, storage_cls, idx, device, elts = id
            data = self.zf.read('{}/data/{}'.format(self.base_name, idx))
            storage = storage_cls.from_buffer(data, 'little')
            assert storage.device.type == device, "should handle cpu()/cuda()"
            assert len(storage) == elts
            return storage

        raise pickle.UnpicklingError("unsupported persistent object")

    def find_class(self, module, name):
        parts = module.split('.')
        if parts[0] == "__torch__":
            mod = self.resolver
            for part in parts[1:]:
                mod = getattr(mod, part)
            return getattr(mod, name)
        else:
            return super().find_class(module, name)

def load(filename):

    # glob = {'torch': torch}
    glob = None
    # exec(compile('from torch import Tensor', 'import tensor', 'exec'), glob)
    # exec(compile("class Module:  pass", 'module.py', 'exec'), glob)

    filename_no_ext = os.path.splitext(os.path.split(filename)[-1])[0]
    with zipfile.ZipFile(filename, 'r') as f:
        resolver = Loader('{}/code/__torch__'.format(filename_no_ext), f, glob, name="__torch__") 
        # glob['__torch__'] = resolver
        # glob['__torch__'].globals['__torch__'] = resolver
        globals()['__torch__'] = resolver
        # glob['torch'] = torch
        sys.modules['__torch__'] = resolver

        # TODO where would constants be used? not in resnet aparently..
        constants = pickle.load(f.open(os.path.join(filename_no_ext, 'constants.pkl')))
        data_fname = os.path.join(filename_no_ext, 'data.pkl')
        unpickler = TorchUnpickler(filename_no_ext, f, resolver, f.open(data_fname))
        model = unpickler.load()

    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("load_file", help="File to load model from")
    parser.add_argument("--forward_with_shape", type=int, nargs="+", help="shape of empty tensor to create as input to forward")
    args = parser.parse_args()

    model = load(args.load_file)

    if args.forward_with_shape:
        input = torch.empty(args.forward_with_shape).uniform_()

        # HACK, but still doesn't help bc gets called with shapes 1,3,17,17 and 1,3,56,56?
        torch.add_ = torch.add

        out = model.forward(input)
        print(out)
