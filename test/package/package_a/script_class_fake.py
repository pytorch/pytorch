import torch

class MyScriptClass:
    def __init__(self, x):
        self.foo = x

    def getFooTest(self):
        return self.foo

def uses_script_class(x):
    foo = MyScriptClass(x)
    return foo.foo
