# Step by step guide on using PyTorch's DevContainer

Using PyTorch's DevContainer environment involves a series of steps that will help you set up a development environment that is isolated and replicable. Below, we'll guide you through each step to make this process as smooth as possible:

## Step 1: Install VSCode

1. Navigate to the [Visual Studio Code website](https://code.visualstudio.com/).
2. Download the appropriate installer for your operating system (Windows, Linux, or macOS).
3. Run the installer and follow the on-screen instructions to install VSCode on your system.
4. After installation, launch VSCode.

## Step 2: Install DevContainer Extension

1. In VSCode, go to the Extensions view by clicking on the Extensions icon in the Activity Bar on the side of the window.
2. Search for "Dev Containers" in the Extensions view search bar.
3. Find the "Dev Containers" extension in the search results and click on the install button to install it.

You can also go to the extension's [homepage](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) and [documentation page](https://code.visualstudio.com/docs/devcontainers/containers) to find more details.

## Step 3: Install Docker and Add Current Login User to Docker Group

1. Follow the [official guide](https://docs.docker.com/get-docker/) to install Docker. Don't forget the [post installation steps](https://docs.docker.com/engine/install/linux-postinstall/).

If you are using [Visual Studio Code Remote - SSH](https://code.visualstudio.com/docs/remote/ssh), then you only need to install Docker in the remote host, not your local computer. And the following steps should be run in the remote host.

## Step 4 (Optional): Install NVIDIA Container Toolkit for GPU Usage

1. If you intend to use GPU resources, first ensure you have NVIDIA drivers installed on your system. Check if `nvidia-smi` works to verify your GPU setup.
2. Follow the [official guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#docker) to install the NVIDIA Container Toolkit.
3. After installation, verify that the toolkit is installed correctly by running:
   ```
   docker run --rm --runtime=nvidia --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi
   ```

## Step 5: Clone PyTorch

1. Open a terminal or command prompt.
2. Use the following command to clone the PyTorch repository:
   ```
   git clone https://github.com/pytorch/pytorch
   ```
3. Navigate to the cloned directory:
   ```
   cd pytorch
   ```

## Step 6: Open in DevContainer

1. In VSCode, use the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P` on macOS) to run the "Dev Containers: Open Folder in Container..." command.
2. You will be prompted with two options: CPU dev container or CUDA dev container. Choose the one you want to run.

## Step 7: Wait for Building the Environment

1. After opening the folder in a DevContainer, VSCode will start building the container. This process can take some time as it involves downloading necessary images and setting up the environment.
2. You can monitor the progress in the VSCode terminal.
3. Once the build process completes, you'll have a fully configured PyTorch development environment in a container.
4. The next time you open the same dev container, it will be much faster, as it does not require building the image again.

You are now all set to start developing with PyTorch in a DevContainer environment. This setup ensures you have a consistent and isolated development environment for your PyTorch projects.

## Step 8: Build PyTorch

To build pytorch from source, simply run:
   ```
   python setup.py develop
   ```

The process involves compiling thousands of files, and would take a long time. Fortunately, the compiled objects can be useful for your next build. When you modify some files, you only need to compile the changed files the next time.

Note that only contents in the `pytorch` directory are saved to disk. This directory is mounted to the docker image, while other contents in the docker image are all temporary, and will be lost if docker restarts the image or the server reboots.

For an in-depth understanding of Dev Container and its caveats, please refer to [the full documentation](https://code.visualstudio.com/docs/devcontainers/containers).
