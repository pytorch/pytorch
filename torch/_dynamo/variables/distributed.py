from .base import VariableTracker


class ProcessGroupVariable(VariableTracker):
    def __init__(self, proxy, value, **kwargs):
        super(ProcessGroupVariable, self).__init__(**kwargs)
        self.proxy = proxy
        self.value = value

    def as_proxy(self):
        return self.proxy

    def __str__(self):
        return f"ProcessGroupVariable({type(self.value)})"

    def python_type(self):
        return type(self.value)

    def as_python_constant(self):
        return self.value
