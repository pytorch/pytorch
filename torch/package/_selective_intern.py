SELECTIVE_INTERN_LIST = [
    $interned_modules
]

if "__torch_package__" in dir():
    def __getattr__(name):
        print(dir())
        print(locals())
        print(globals())
        print(f"SHIM: {name}")
        obj = dir().get(name, None)
        obj = obj if obj else locals().get(name, None)
        obj = obj if obj else globals().get(name, None)

        import types

        if isinstance(obj, types.ModuleType) and name not in SELECTIVE_INTERN_LIST:
            return getattr(_extern_copy, name)

        return globals().get(name)
