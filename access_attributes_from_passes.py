import torch

def printModuleRecursively(name, m):
    print('=== %s ===' % name)
    for f in m._get_functions():
        print(f)
        print(f.graph)
    for name, sm in m._get_modules():
        printModuleRecursively(name, sm)

def populateQConfMap(m, qconf_map):
    if m._has_attribute('quant_config'):
        qconf = m._get_attribute('quant_config')
        for f in m._get_functions():
            qconf_map[f] = qconf

    for name, sm in m._get_modules():
        populateQConfMap(sm, qconf_map)

def script_with_submodules():
    class SubModule_MatMul(torch.jit.ScriptModule):
        def __init__(self):
            super(SubModule_MatMul, self).__init__()
            self.weight = torch.jit.Parameter(torch.Tensor(3, 3))
            self.quant_config = torch.jit.Attribute(111, int)

        @torch.jit.script_method
        def forward(self, input):
            return input.matmul(self.weight.t())

    class SubModule_Add(torch.jit.ScriptModule):
        def __init__(self):
            super(SubModule_Add, self).__init__()
            self.weight = torch.jit.Parameter(torch.Tensor(3, 3))
            self.quant_config = torch.jit.Attribute(222, int)

        @torch.jit.script_method
        def forward(self, input):
            return input + self.weight

    class MyModule(torch.jit.ScriptModule):
        def __init__(self):
            super(MyModule, self).__init__()
            self.submodule_matmul = SubModule_MatMul()
            self.submodule_add = SubModule_Add()
            self.quant_config = torch.jit.Attribute(333, int)

        @torch.jit.script_method
        def forward(self, a, b):
            d = self.submodule_matmul(a) * self.submodule_add(b)
            return d

    scripted = MyModule()
    qconf_map = {}
    populateQConfMap(scripted._c, qconf_map)
    print('Before passes:')
    printModuleRecursively('top', scripted._c)
    print('--------------------------------------')
    torch._C._jit_pass_propagate_qinfo(scripted._c, qconf_map)
    print('--------------------------------------')
    print('After passes:')
    printModuleRecursively('top', scripted._c)

def main():
    script_with_submodules()

if __name__ == '__main__':
    main()
