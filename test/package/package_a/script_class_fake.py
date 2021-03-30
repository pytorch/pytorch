class MyScriptClass:
    """Intended to be scripted."""
    def __init__(self, x):
        self.foo = x

    def getFooTest(self):
        return self.foo

def uses_script_class(x):
    """Intended to be scripted."""
    foo = MyScriptClass(x)
    return foo.foo
