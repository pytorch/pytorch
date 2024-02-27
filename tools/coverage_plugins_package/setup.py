import setuptools  # type: ignore[import]

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="coverage-plugins",
    version="0.0.1",
    author="PyTorch Team",
    author_email="packages@pytorch.org",
    description="plug-in to coverage for PyTorch JIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pytorch/pytorch",
    project_urls={
        "Bug Tracker": "https://github.com/pytorch/pytorch/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
