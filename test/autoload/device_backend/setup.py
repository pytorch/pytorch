from setuptools import find_packages, setup

setup(
    name="device_backend",
    version="0.1",
    packages=find_packages(),
    entry_points={
        "torch.backends": [
            "device_backend = backend_pkg:_autoload",
        ],
    },
)
