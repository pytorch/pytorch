SELECTIVE_INTERN_LIST = [
    $interned_modules
]

if "__torch_package__" in dir():

    def __getattr__(name):
        package_name = $package_name
        if name not in SELECTIVE_INTERN_LIST:
            return getattr(_extern_copy, name)
        obj = locals().get(name, None)
        obj = obj if obj else globals().get(name, None)
        __import__(f'{package_name}.{name}',level=0)
        return globals().get(name)
