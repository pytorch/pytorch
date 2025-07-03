class Backend(Enum, str):  # noqa: SLOT000
    UNDEFINED = "undefined"
    GLOO = "gloo"
    NCCL = "nccl"
    UCC = "ucc"
    MPI = "mpi"
    XCCL = "xccl"

    _BackendPlugin = namedtuple("_BackendPlugin", ["creator_fn", "extended_api"])

    _plugins: ClassVar[dict[str, _BackendPlugin]] = {}

    _registered_backends: ClassVar[set[str]] = set()

    # 3rd-party devices can register the default backend support here
    default_device_backend_map: ClassVar[dict[str, str]] = {
        "cpu": "gloo",
        "cuda": "nccl",
        "xpu": "xccl",
        "mps": "gloo",
    }

    backend_capability: ClassVar[dict[str, list[str]]] = {
        "gloo": ["cpu", "cuda"],
        "nccl": ["cuda"],
        "xccl": ["xpu"],
        "ucc": ["cpu", "cuda"],
        "mpi": ["cpu", "cuda"],
    }

    backend_type_map: ClassVar[dict[str, ProcessGroup.BackendType]] = {
        "undefined": ProcessGroup.BackendType.UNDEFINED,
        "gloo": ProcessGroup.BackendType.GLOO,
        "nccl": ProcessGroup.BackendType.NCCL,
        "xccl": ProcessGroup.BackendType.XCCL,
        "ucc": ProcessGroup.BackendType.UCC,
        "mpi": ProcessGroup.BackendType.MPI,
    }

    def __str__(self) -> str:
        return self.value

    def __new__(cls, value: str):
        """Create and return a new instance of the class."""
        if isinstance(value, str):
            for member in cls:
                if member.value.lower() == value.lower():
                    return member
            if value.lower() in cls._registered_backends:
                obj = str.__new__(cls, value.lower())
                obj._value_ = value.lower()
                return obj

        obj = str.__new__(cls, value)
        obj._value_ = value
        return obj

    @classmethod
    def register_backend(
        cls,
        name,
        func,
        extended_api=False,
        devices: Optional[Union[str, list[str]]] = None,
    ) -> None:
        if name.lower() in cls._registered_backends:
            raise ValueError(f"Backend {name} already registered")

        if devices is not None:
            device_list = [devices] if isinstance(devices, str) else devices
            for device in device_list:
                if device not in cls.default_device_backend_map:
                    cls.default_device_backend_map[device] = name.lower()
        cls._registered_backends.add(name.lower())

        # Update device capability matrix in Backend class
        if devices is None:
            # This is more of a backward support for groups like `threaded`:
            # assume default devices "cpu" and "cuda", but warn
            warnings.warn(
                f"Device capability of {name} unspecified, assuming `cpu` and "
                "`cuda`. Please specify it via the `devices` argument of "
                "`register_backend`."
            )
            cls.backend_capability[name.lower()] = ["cpu", "cuda"]
        elif isinstance(devices, str):
            # Single device string specified. Simply convert to list.
            cls.backend_capability[name.lower()] = [devices]
        else:
            cls.backend_capability[name.lower()] = devices

        cls.backend_type_map[name.lower()] = cls.ProcessGroup.BackendType.CUSTOM
        cls._plugins[name.upper()] = cls._BackendPlugin(func, extended_api)
