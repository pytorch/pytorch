import os
import subprocess
import sys
from pathlib import Path

# Enable parallel compilation
os.environ.setdefault("MAX_JOBS", str(os.cpu_count() or 4))

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension


# Collect all C++ sources
def get_sources():
    src_dirs = ["csrc/vulkan", "csrc/backend", "csrc/ops", "csrc/autocast"]
    sources = []
    root = Path(__file__).parent
    for d in src_dirs:
        dir_path = root / d
        if dir_path.exists():
            sources.extend(
                str(p.relative_to(root)) for p in dir_path.glob("*.cpp")
            )
    sources.append("csrc/init.cpp")
    return sources


root_dir = Path(__file__).parent.resolve()
vma_dir = str(root_dir / "third_party/VulkanMemoryAllocator/include")


# Custom build extension that compiles shaders before C++
class ShaderBuildExtension(BuildExtension):
    def build_extensions(self):
        root = Path(__file__).parent
        gen_dir = root / "csrc" / "generated"
        gen_dir.mkdir(parents=True, exist_ok=True)
        header = gen_dir / "shaders.h"

        # Try to compile shaders before building C++
        try:
            print("Compiling Slang shaders...")
            subprocess.run(
                [sys.executable, "tools/compile_shaders.py"],
                check=True,
                cwd=str(root),
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Warning: Shader compilation failed ({e}).")
            if not header.exists():
                print("Generating stub shaders.h for compilation...")
                subprocess.run(
                    [sys.executable, "tools/generate_stub_shaders.py"],
                    check=True,
                    cwd=str(root),
                )
        super().build_extensions()


setup(
    name="torch_vulkan",
    version="0.1.0",
    description="PyTorch Vulkan backend with Slang shaders for full training support",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=[
        CppExtension(
            name="torch_vulkan._C",
            sources=get_sources(),
            include_dirs=[
                str(root_dir / "csrc"),
                str(root_dir / "csrc/generated"),
                vma_dir,
            ],
            libraries=["vulkan"],
            define_macros=[
                ("VMA_STATIC_VULKAN_FUNCTIONS", "0"),
                ("VMA_DYNAMIC_VULKAN_FUNCTIONS", "1"),
            ],
            extra_compile_args=["-std=c++17"],
        ),
    ],
    cmdclass={"build_ext": ShaderBuildExtension},
    python_requires=">=3.9",
    install_requires=["torch>=2.1"],
    entry_points={
        "torch.backends": ["vulkan = torch_vulkan:_register"],
    },
)
