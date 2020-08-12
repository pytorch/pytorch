import os
import pickle
import types
import zipfile

class Loader:
    def __init__(self, base, zf, globals):
        self.base = base
        self.zf = zf
        self.globals = globals.copy()
        self.fresh = True

    def __getattr__(self, name):
        if name[:7] == "_ResNet":
            name = name[7:]
        # if name == '_ResNet___torch_mangle_29':
            # import ipdb; ipdb.set_trace()
        print("{} -> {}".format(self.base, name))
        dirname = os.path.join(self.base, name)
        filename = dirname + '.py'
        if filename in self.zf.namelist():
            # the Loader for 'module' is loading 'linear.py' which should define some stuff (Linear? others?).  
            # We would like to eval it into a 'module' object and setattr the new module
            # but we can't eval it bc it depends on itself for annotations 
            # what if i can somehow eval into myself instead of this 'globals' thing?  i need Linear to exist inside 'linear'.
            ldr = Loader(dirname, self.zf, self.globals)
            setattr(self, name, ldr) # Temporary class to allow annotations to work
            self.globals[name] = ldr
            f = self.zf.read(filename)
            print("   Compile: ", name)
            exec(compile(f, name, 'exec'), ldr.globals)
            ldr.fresh = False
            print("   Return finished loader for module {}".format(name))
            return ldr

        if any([dirname in x for x in self.zf.namelist()]):
            attr = Loader(dirname, self.zf, self.globals)
            setattr(self, name, attr)
            print("   Return loader for dir {}".format(name))
            return attr


        # Need to find a way to return a dummy that 
        # - can be used recursively
        #   - e.g. while compiling 'resnet' module, 
        #          dummy 'resnet._Sequential___torch_mangle_13' gets returned
        # .        and BasicBlock gets looked up on that dummy
        # - could also be terminal
        #   - I am replacing the terminal ones above, which will break the recursive part

        # i'm looking up 'Sequential' class correctly from 'container.py' module
        # but failing to treat '_ResNet___torch_mangle_29' as a module name in 'container' dir
        # since container already exists as a .py module
        # how do I make a correct distinction?
        if self.fresh:
            # return types.new_class("Dummy Class")
            print("   Return sub- loader for {}".format(name))
            return Loader(dirname, self.zf, self.globals)
        else:
            if name in self.globals:
                print("   Return globals for {}".format(name))
                return self.globals[name]
            raise AttributeError("{} not found in {}".format(name, self.base))


class TorchUnpickler(pickle.Unpickler):
    def __init__(self, resolver, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resolver = resolver

    def find_class(self, module, name):
        print("find_class {}:{}".format(module, name))
        for step in module.split('.'):
            if step == '__torch__':
                mod = self.resolver
            else:
                mod = getattr(mod, step)
        
        import ipdb; ipdb.set_trace()
        print("foo")


def load(filename):

    glob = {}
    # glob['pickle'] = pickle
    exec(compile('from torch import Tensor', 'import tensor', 'exec'), glob)
    with open('torchpy/module.py', 'r') as f:
        exec(compile(f.read(), 'module.py', 'exec'), glob)


    filename_no_ext = 'resnet'
    with zipfile.ZipFile(filename, 'r') as f:
        resolver = Loader('{}/code/__torch__'.format(filename_no_ext), f, glob) 
        glob['__torch__'] = resolver
        glob['__torch__'].globals['__torch__'] = resolver
        
        # with open('/home/whc/local/pytorch/torchpy/example/resnet_extracted/resnet/code/__torch__/torch/nn/modules/linear.py', 'r') as f:
            # exec(compile(f.read(), 'module.py', 'exec'), glob)
        # with open('/home/whc/local/pytorch/torchpy/example/resnet_extracted/resnet/code/__torch__/torchvision/models/resnet.py', 'r') as f:
            # exec(compile(f.read(), 'module.py', 'exec'), glob)

        for src in ['data.pkl']: #['constants.pkl', 'data.pkl']:
            fname = os.path.join(filename_no_ext, src)
            upk = TorchUnpickler(resolver, f.open(fname))
            upk.load()
            print("blah")

    return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("load_file", help="File to load model from")
    args = parser.parse_args()

    model = load(args.load_file)
