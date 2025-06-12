#!/usr/bin/env python3
import os
import sys
import shutil
import subprocess
import urllib.request
import ssl

def main():
    # Set variables
    MAGMA_VERSION = "2.5.4"

    # Get environment variables
    CUDA_VERSION = os.environ.get('CUDA_VERSION', '')
    CONFIG = os.environ.get('CONFIG', '')
    NUMBER_OF_PROCESSORS = os.environ.get('NUMBER_OF_PROCESSORS', '1')

    if not CUDA_VERSION or not CONFIG:
        print("Error: CUDA_VERSION and CONFIG environment variables must be set")
        sys.exit(1)

    CUVER_NODOT = CUDA_VERSION
    CUVER = f"{CUVER_NODOT[:-1]}.{CUVER_NODOT[-1]}"

    # Convert config to lowercase
    CONFIG_LOWERCASE = CONFIG.replace('D', 'd').replace('R', 'r').replace('M', 'm')

    print(f"Building for configuration: {CONFIG_LOWERCASE}, {CUVER}")

    # Download Ninja
    print("Downloading Ninja...")
    ninja_url = "https://s3.amazonaws.com/ossci-windows/ninja_1.8.2.exe"
    ninja_path = r"C:\Tools\ninja.exe"

    # Create Tools directory if it doesn't exist
    os.makedirs(r"C:\Tools", exist_ok=True)

    # Download with SSL verification disabled (equivalent to curl -k)
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    try:
        with urllib.request.urlopen(ninja_url, context=ssl_context) as response:
            with open(ninja_path, 'wb') as out_file:
                out_file.write(response.read())
    except Exception as e:
        print(f"Error downloading Ninja: {e}")
        sys.exit(1)

    # Set environment variables
    cuda_base = rf"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v{CUVER}"
    os.environ['PATH'] = rf"C:\Tools;{cuda_base}\bin;{cuda_base}\libnvvp;" + os.environ['PATH']
    os.environ['CUDA_PATH'] = cuda_base
    os.environ['NVTOOLSEXT_PATH'] = r"C:\Program Files\NVIDIA Corporation\NvToolsExt"

    # Create and navigate to directory
    magma_dir = f"magma_cuda{CUVER_NODOT}"
    os.makedirs(magma_dir, exist_ok=True)
    os.chdir(magma_dir)

    # Clone or clean magma repository
    if not os.path.exists("magma"):
        print("Cloning magma repository...")
        result = subprocess.run(
            ["git", "clone", "https://github.com/ptrblck/magma_win.git", "magma"],
            capture_output=True
        )
        if result.returncode != 0:
            print(f"Error cloning repository: {result.stderr.decode()}")
            sys.exit(1)
    else:
        print("Cleaning existing build directories...")
        shutil.rmtree("magma/build", ignore_errors=True)
        shutil.rmtree("magma/install", ignore_errors=True)

    # Navigate to build directory
    os.chdir("magma")
    os.makedirs("build", exist_ok=True)
    os.chdir("build")

    # Set GPU target and CUDA architecture list
    GPU_TARGET = "All"
    CUDA_ARCH_LIST = ""

    if CUVER_NODOT == "128":
        CUDA_ARCH_LIST = "-gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_100,code=sm_100 -gencode arch=compute_120,code=sm_120"
    elif CUVER_NODOT[:2] == "12" and CUVER_NODOT != "128":
        CUDA_ARCH_LIST = "-gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90"
    elif CUVER_NODOT == "118":
        CUDA_ARCH_LIST = "-gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90"

    # Set compiler environment variables
    os.environ['CC'] = "cl.exe"
    os.environ['CXX'] = "cl.exe"

    # Run cmake configure
    print("Configuring with cmake...")
    cmake_cmd = [
        "cmake", "..",
        f"-DGPU_TARGET={GPU_TARGET}",
        "-DUSE_FORTRAN=0",
        "-DCMAKE_CXX_FLAGS=/FS /Zf",
        f"-DCMAKE_BUILD_TYPE={CONFIG}",
        "-DCMAKE_GENERATOR=Ninja",
        "-DCMAKE_INSTALL_PREFIX=..\\install\\",
        f"-DCUDA_ARCH_LIST={CUDA_ARCH_LIST}",
        "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
    ]

    result = subprocess.run(cmake_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"CMake configure failed: {result.stderr}")
        sys.exit(1)

    # Build and install
    print("Building and installing...")
    build_cmd = [
        "cmake", "--build", ".", "--target", "install",
        "--config", CONFIG, "--", f"-j{NUMBER_OF_PROCESSORS}"
    ]

    result = subprocess.run(build_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Build failed: {result.stderr}")
        sys.exit(1)

    # Navigate back to original directory
    os.chdir("../../..")

    # Create archive
    print("Creating archive...")
    archive_name = f"magma_{MAGMA_VERSION}_cuda{CUVER_NODOT}_{CONFIG_LOWERCASE}.7z"
    source_path = os.path.join(os.getcwd(), f"magma_cuda{CUVER_NODOT}", "magma", "install", "*")

    archive_cmd = ["7z", "a", archive_name, source_path]
    subprocess.run(archive_cmd)

    # Clean up
    print("Cleaning up...")
    shutil.rmtree(f"magma_cuda{CUVER_NODOT}", ignore_errors=True)

    print("Build completed successfully!")

if __name__ == "__main__":
    main()
