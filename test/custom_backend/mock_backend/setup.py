from setuptools import setup


setup(
    name="torch_mock_backend",
    packages=["torch_mock_backend"],
    entry_points={
        "torch.backends": [
            "device_backend = torch_mock_backend:_autoload",
        ],
    },
)
