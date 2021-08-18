SELECTIVE_INTERN_LIST = {
    "foo",
}

if "__torch_package__" in dir():
    def __getattr__(name):
        if name not in SELECTIVE_INTERN_LIST:
            return getattr(_extern_torch, name)
        raise AttributeError("testing")
