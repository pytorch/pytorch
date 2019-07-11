
class FunctionBackend(object):

    def __init__(self):
        self.function_classes = {}

    def __getattr__(self, name):
        fn = self.function_classes.get(name)
        if fn is None:
            raise NotImplementedError
        return fn

    def register_function(self, name, function_class):
        if self.function_classes.get(name):
            raise RuntimeError("Trying to register second function under name " + name + " in " + type(self).__name__)
        self.function_classes[name] = function_class
