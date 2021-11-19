SELECTIVE_INTERN_LIST = [
    $interned_modules
]

if "__torch_package__" in dir():
    def __getattr__(name):
        print(f"GETATTR: {name}")
        if name not in SELECTIVE_INTERN_LIST:
            return getattr(_extern_copy, name)

        obj = locals().get(name, None)
        obj = obj if obj else globals().get(name, None)
        return globals().get(name)
