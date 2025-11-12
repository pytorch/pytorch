#!/usr/bin/env python3

# This script is for building  AARCH64 wheels using AWS EC2 instances.
# To generate binaries for the release follow these steps:
# 1. Update mappings for each of the Domain Libraries by adding new row to a table like this:
#         "v1.11.0": ("0.11.0", "rc1"),
# 2. Run script with following arguments for each of the supported python versions and required tag, for example:
# build_aarch64_wheel.py --key-name <YourPemKey> --use-docker --python 3.8 --branch v1.11.0-rc3


import os
import subprocess
import sys
import time
from typing import Optional, Union

import boto3


# AMI images for us-east-1, change the following based on your ~/.aws/config
os_amis = {
    "ubuntu20_04": "ami-052eac90edaa9d08f",  # login_name: ubuntu
    "ubuntu22_04": "ami-0c6c29c5125214c77",  # login_name: ubuntu
    "redhat8": "ami-0698b90665a2ddcf1",  # login_name: ec2-user
}

ubuntu20_04_ami = os_amis["ubuntu20_04"]


def compute_keyfile_path(key_name: Optional[str] = None) -> tuple[str, str]:
    if key_name is None:
        key_name = os.getenv("AWS_KEY_NAME")
        if key_name is None:
            return os.getenv("SSH_KEY_PATH", ""), ""

    homedir_path = os.path.expanduser("~")
    default_path = os.path.join(homedir_path, ".ssh", f"{key_name}.pem")
    return os.getenv("SSH_KEY_PATH", default_path), key_name


ec2 = boto3.resource("ec2")


def ec2_get_instances(filter_name, filter_value):
    return ec2.instances.filter(
        Filters=[{"Name": filter_name, "Values": [filter_value]}]
    )


def ec2_instances_of_type(instance_type="t4g.2xlarge"):
    return ec2_get_instances("instance-type", instance_type)


def ec2_instances_by_id(instance_id):
    rc = list(ec2_get_instances("instance-id", instance_id))
    return rc[0] if len(rc) > 0 else None


def start_instance(
    key_name, ami=ubuntu20_04_ami, instance_type="t4g.2xlarge", ebs_size: int = 50
):
    inst = ec2.create_instances(
        ImageId=ami,
        InstanceType=instance_type,
        SecurityGroups=["ssh-allworld"],
        KeyName=key_name,
        MinCount=1,
        MaxCount=1,
        BlockDeviceMappings=[
            {
                "DeviceName": "/dev/sda1",
                "Ebs": {
                    "DeleteOnTermination": True,
                    "VolumeSize": ebs_size,
                    "VolumeType": "standard",
                },
            }
        ],
    )[0]
    print(f"Create instance {inst.id}")
    inst.wait_until_running()
    running_inst = ec2_instances_by_id(inst.id)
    print(f"Instance started at {running_inst.public_dns_name}")
    return running_inst


class RemoteHost:
    addr: str
    keyfile_path: str
    login_name: str
    container_id: Optional[str] = None
    ami: Optional[str] = None

    def __init__(self, addr: str, keyfile_path: str, login_name: str = "ubuntu"):
        self.addr = addr
        self.keyfile_path = keyfile_path
        self.login_name = login_name

    def _gen_ssh_prefix(self) -> list[str]:
        return [
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            "-i",
            self.keyfile_path,
            f"{self.login_name}@{self.addr}",
            "--",
        ]

    @staticmethod
    def _split_cmd(args: Union[str, list[str]]) -> list[str]:
        return args.split() if isinstance(args, str) else args

    def run_ssh_cmd(self, args: Union[str, list[str]]) -> None:
        subprocess.check_call(self._gen_ssh_prefix() + self._split_cmd(args))

    def check_ssh_output(self, args: Union[str, list[str]]) -> str:
        return subprocess.check_output(
            self._gen_ssh_prefix() + self._split_cmd(args)
        ).decode("utf-8")

    def scp_upload_file(self, local_file: str, remote_file: str) -> None:
        subprocess.check_call(
            [
                "scp",
                "-i",
                self.keyfile_path,
                local_file,
                f"{self.login_name}@{self.addr}:{remote_file}",
            ]
        )

    def scp_download_file(
        self, remote_file: str, local_file: Optional[str] = None
    ) -> None:
        if local_file is None:
            local_file = "."
        subprocess.check_call(
            [
                "scp",
                "-i",
                self.keyfile_path,
                f"{self.login_name}@{self.addr}:{remote_file}",
                local_file,
            ]
        )

    def start_docker(self, image="quay.io/pypa/manylinux2014_aarch64:latest") -> None:
        self.run_ssh_cmd("sudo apt-get install -y docker.io")
        self.run_ssh_cmd(f"sudo usermod -a -G docker {self.login_name}")
        self.run_ssh_cmd("sudo service docker start")
        self.run_ssh_cmd(f"docker pull {image}")
        self.container_id = self.check_ssh_output(
            f"docker run -t -d -w /root {image}"
        ).strip()

    def using_docker(self) -> bool:
        return self.container_id is not None

    def run_cmd(self, args: Union[str, list[str]]) -> None:
        if not self.using_docker():
            return self.run_ssh_cmd(args)
        assert self.container_id is not None
        docker_cmd = self._gen_ssh_prefix() + [
            "docker",
            "exec",
            "-i",
            self.container_id,
            "bash",
        ]
        p = subprocess.Popen(docker_cmd, stdin=subprocess.PIPE)
        p.communicate(
            input=" ".join(["source .bashrc && "] + self._split_cmd(args)).encode(
                "utf-8"
            )
        )
        rc = p.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, docker_cmd)

    def check_output(self, args: Union[str, list[str]]) -> str:
        if not self.using_docker():
            return self.check_ssh_output(args)
        assert self.container_id is not None
        docker_cmd = self._gen_ssh_prefix() + [
            "docker",
            "exec",
            "-i",
            self.container_id,
            "bash",
        ]
        p = subprocess.Popen(docker_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        (out, err) = p.communicate(
            input=" ".join(["source .bashrc && "] + self._split_cmd(args)).encode(
                "utf-8"
            )
        )
        rc = p.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, docker_cmd, output=out, stderr=err)
        return out.decode("utf-8")

    def upload_file(self, local_file: str, remote_file: str) -> None:
        if not self.using_docker():
            return self.scp_upload_file(local_file, remote_file)
        tmp_file = os.path.join("/tmp", os.path.basename(local_file))
        self.scp_upload_file(local_file, tmp_file)
        self.run_ssh_cmd(
            ["docker", "cp", tmp_file, f"{self.container_id}:/root/{remote_file}"]
        )
        self.run_ssh_cmd(["rm", tmp_file])

    def download_file(self, remote_file: str, local_file: Optional[str] = None) -> None:
        if not self.using_docker():
            return self.scp_download_file(remote_file, local_file)
        tmp_file = os.path.join("/tmp", os.path.basename(remote_file))
        self.run_ssh_cmd(
            ["docker", "cp", f"{self.container_id}:/root/{remote_file}", tmp_file]
        )
        self.scp_download_file(tmp_file, local_file)
        self.run_ssh_cmd(["rm", tmp_file])

    def download_wheel(
        self, remote_file: str, local_file: Optional[str] = None
    ) -> None:
        if self.using_docker() and local_file is None:
            basename = os.path.basename(remote_file)
            local_file = basename.replace(
                "-linux_aarch64.whl", "-manylinux2014_aarch64.whl"
            )
        self.download_file(remote_file, local_file)

    def list_dir(self, path: str) -> list[str]:
        return self.check_output(["ls", "-1", path]).split("\n")


def wait_for_connection(addr, port, timeout=15, attempt_cnt=5):
    import socket

    for i in range(attempt_cnt):
        try:
            with socket.create_connection((addr, port), timeout=timeout):
                return
        except (ConnectionRefusedError, TimeoutError):  # noqa: PERF203
            if i == attempt_cnt - 1:
                raise
            time.sleep(timeout)


def update_apt_repo(host: RemoteHost) -> None:
    time.sleep(5)
    host.run_cmd("sudo systemctl stop apt-daily.service || true")
    host.run_cmd("sudo systemctl stop unattended-upgrades.service || true")
    host.run_cmd(
        "while systemctl is-active --quiet apt-daily.service; do sleep 1; done"
    )
    host.run_cmd(
        "while systemctl is-active --quiet unattended-upgrades.service; do sleep 1; done"
    )
    host.run_cmd("sudo apt-get update")
    time.sleep(3)
    host.run_cmd("sudo apt-get update")


def install_condaforge(
    host: RemoteHost, suffix: str = "latest/download/Miniforge3-Linux-aarch64.sh"
) -> None:
    print("Install conda-forge")
    host.run_cmd(f"curl -OL https://github.com/conda-forge/miniforge/releases/{suffix}")
    host.run_cmd(f"sh -f {os.path.basename(suffix)} -b")
    host.run_cmd(f"rm -f {os.path.basename(suffix)}")
    if host.using_docker():
        host.run_cmd("echo 'PATH=$HOME/miniforge3/bin:$PATH'>>.bashrc")
    else:
        host.run_cmd(
            [
                "sed",
                "-i",
                "'/^# If not running interactively.*/i PATH=$HOME/miniforge3/bin:$PATH'",
                ".bashrc",
            ]
        )


def install_condaforge_python(host: RemoteHost, python_version="3.8") -> None:
    if python_version == "3.6":
        # Python-3.6 EOLed and not compatible with conda-4.11
        install_condaforge(
            host, suffix="download/4.10.3-10/Miniforge3-4.10.3-10-Linux-aarch64.sh"
        )
        host.run_cmd(f"conda install -y python={python_version} numpy pyyaml")
    else:
        install_condaforge(
            host, suffix="download/4.11.0-4/Miniforge3-4.11.0-4-Linux-aarch64.sh"
        )
        # Pytorch-1.10 or older are not compatible with setuptools=59.6 or newer
        host.run_cmd(
            f"conda install -y python={python_version} numpy pyyaml setuptools>=59.5.0"
        )


def embed_libgomp(host: RemoteHost, use_conda, wheel_name) -> None:
    host.run_cmd("pip3 install auditwheel")
    host.run_cmd(
        "conda install -y patchelf" if use_conda else "sudo apt-get install -y patchelf"
    )
    from tempfile import NamedTemporaryFile

    with NamedTemporaryFile() as tmp:
        tmp.write(embed_library_script.encode("utf-8"))
        tmp.flush()
        host.upload_file(tmp.name, "embed_library.py")

    print("Embedding libgomp into wheel")
    if host.using_docker():
        host.run_cmd(f"python3 embed_library.py {wheel_name} --update-tag")
    else:
        host.run_cmd(f"python3 embed_library.py {wheel_name}")


def checkout_repo(
    host: RemoteHost,
    *,
    branch: str = "main",
    url: str,
    git_clone_flags: str,
    mapping: dict[str, tuple[str, str]],
) -> Optional[str]:
    for prefix in mapping:
        if not branch.startswith(prefix):
            continue
        tag = f"v{mapping[prefix][0]}-{mapping[prefix][1]}"
        host.run_cmd(f"git clone {url} -b {tag} {git_clone_flags}")
        return mapping[prefix][0]

    host.run_cmd(f"git clone {url} -b {branch} {git_clone_flags}")
    return None


def build_torchvision(
    host: RemoteHost,
    *,
    branch: str = "main",
    use_conda: bool = True,
    git_clone_flags: str,
    run_smoke_tests: bool = True,
) -> str:
    print("Checking out TorchVision repo")
    build_version = checkout_repo(
        host,
        branch=branch,
        url="https://github.com/pytorch/vision",
        git_clone_flags=git_clone_flags,
        mapping={
            "v1.7.1": ("0.8.2", "rc2"),
            "v1.8.0": ("0.9.0", "rc3"),
            "v1.8.1": ("0.9.1", "rc1"),
            "v1.9.0": ("0.10.0", "rc1"),
            "v1.10.0": ("0.11.1", "rc1"),
            "v1.10.1": ("0.11.2", "rc1"),
            "v1.10.2": ("0.11.3", "rc1"),
            "v1.11.0": ("0.12.0", "rc1"),
            "v1.12.0": ("0.13.0", "rc4"),
            "v1.12.1": ("0.13.1", "rc6"),
            "v1.13.0": ("0.14.0", "rc4"),
            "v1.13.1": ("0.14.1", "rc2"),
            "v2.0.0": ("0.15.1", "rc2"),
            "v2.0.1": ("0.15.2", "rc2"),
        },
    )
    print("Building TorchVision wheel")

    # Please note libnpg and jpeg are required to build image.so extension
    if use_conda:
        host.run_cmd("conda install -y libpng jpeg")
        # Remove .so files to force static linking
        host.run_cmd(
            "rm miniforge3/lib/libpng.so miniforge3/lib/libpng16.so miniforge3/lib/libjpeg.so"
        )
        # And patch setup.py to include libz dependency for libpng
        host.run_cmd(
            [
                'sed -i -e \'s/image_link_flags\\.append("png")/image_link_flags += ["png", "z"]/\' vision/setup.py'
            ]
        )

    build_vars = ""
    if branch == "nightly":
        version = host.check_output(
            ["if [ -f vision/version.txt ]; then cat vision/version.txt; fi"]
        ).strip()
        if len(version) == 0:
            # In older revisions, version was embedded in setup.py
            version = (
                host.check_output(["grep", '"version = \'"', "vision/setup.py"])
                .strip()
                .split("'")[1][:-2]
            )
        build_date = (
            host.check_output("cd vision && git log --pretty=format:%s -1")
            .strip()
            .split()[0]
            .replace("-", "")
        )
        build_vars += f"BUILD_VERSION={version}.dev{build_date}"
    elif build_version is not None:
        build_vars += f"BUILD_VERSION={build_version} PYTORCH_VERSION={branch[1:].split('-', maxsplit=1)[0]}"
    if host.using_docker():
        build_vars += " CMAKE_SHARED_LINKER_FLAGS=-Wl,-z,max-page-size=0x10000"

    host.run_cmd(f"cd vision && {build_vars} python3 -m build --wheel --no-isolation")
    vision_wheel_name = host.list_dir("vision/dist")[0]
    embed_libgomp(host, use_conda, os.path.join("vision", "dist", vision_wheel_name))

    print("Copying TorchVision wheel")
    host.download_wheel(os.path.join("vision", "dist", vision_wheel_name))
    if run_smoke_tests:
        host.run_cmd(
            f"pip3 install {os.path.join('vision', 'dist', vision_wheel_name)}"
        )
        host.run_cmd("python3 vision/test/smoke_test.py")
    print("Delete vision checkout")
    host.run_cmd("rm -rf vision")

    return vision_wheel_name


def build_torchdata(
    host: RemoteHost,
    *,
    branch: str = "main",
    use_conda: bool = True,
    git_clone_flags: str = "",
) -> str:
    print("Checking out TorchData repo")
    git_clone_flags += " --recurse-submodules"
    build_version = checkout_repo(
        host,
        branch=branch,
        url="https://github.com/pytorch/data",
        git_clone_flags=git_clone_flags,
        mapping={
            "v1.13.1": ("0.5.1", ""),
            "v2.0.0": ("0.6.0", "rc5"),
            "v2.0.1": ("0.6.1", "rc1"),
        },
    )
    print("Building TorchData wheel")
    build_vars = ""
    if branch == "nightly":
        version = host.check_output(
            ["if [ -f data/version.txt ]; then cat data/version.txt; fi"]
        ).strip()
        build_date = (
            host.check_output("cd data && git log --pretty=format:%s -1")
            .strip()
            .split()[0]
            .replace("-", "")
        )
        build_vars += f"BUILD_VERSION={version}.dev{build_date}"
    elif build_version is not None:
        build_vars += f"BUILD_VERSION={build_version} PYTORCH_VERSION={branch[1:].split('-', maxsplit=1)[0]}"
    if host.using_docker():
        build_vars += " CMAKE_SHARED_LINKER_FLAGS=-Wl,-z,max-page-size=0x10000"

    host.run_cmd(f"cd data && {build_vars} python3 -m build --wheel --no-isolation")
    wheel_name = host.list_dir("data/dist")[0]
    embed_libgomp(host, use_conda, os.path.join("data", "dist", wheel_name))

    print("Copying TorchData wheel")
    host.download_wheel(os.path.join("data", "dist", wheel_name))

    return wheel_name


def build_torchtext(
    host: RemoteHost,
    *,
    branch: str = "main",
    use_conda: bool = True,
    git_clone_flags: str = "",
) -> str:
    print("Checking out TorchText repo")
    git_clone_flags += " --recurse-submodules"
    build_version = checkout_repo(
        host,
        branch=branch,
        url="https://github.com/pytorch/text",
        git_clone_flags=git_clone_flags,
        mapping={
            "v1.9.0": ("0.10.0", "rc1"),
            "v1.10.0": ("0.11.0", "rc2"),
            "v1.10.1": ("0.11.1", "rc1"),
            "v1.10.2": ("0.11.2", "rc1"),
            "v1.11.0": ("0.12.0", "rc1"),
            "v1.12.0": ("0.13.0", "rc2"),
            "v1.12.1": ("0.13.1", "rc5"),
            "v1.13.0": ("0.14.0", "rc3"),
            "v1.13.1": ("0.14.1", "rc1"),
            "v2.0.0": ("0.15.1", "rc2"),
            "v2.0.1": ("0.15.2", "rc2"),
        },
    )
    print("Building TorchText wheel")
    build_vars = ""
    if branch == "nightly":
        version = host.check_output(
            ["if [ -f text/version.txt ]; then cat text/version.txt; fi"]
        ).strip()
        build_date = (
            host.check_output("cd text && git log --pretty=format:%s -1")
            .strip()
            .split()[0]
            .replace("-", "")
        )
        build_vars += f"BUILD_VERSION={version}.dev{build_date}"
    elif build_version is not None:
        build_vars += f"BUILD_VERSION={build_version} PYTORCH_VERSION={branch[1:].split('-', maxsplit=1)[0]}"
    if host.using_docker():
        build_vars += " CMAKE_SHARED_LINKER_FLAGS=-Wl,-z,max-page-size=0x10000"

    host.run_cmd(f"cd text && {build_vars} python3 -m build --wheel --no-isolation")
    wheel_name = host.list_dir("text/dist")[0]
    embed_libgomp(host, use_conda, os.path.join("text", "dist", wheel_name))

    print("Copying TorchText wheel")
    host.download_wheel(os.path.join("text", "dist", wheel_name))

    return wheel_name


def build_torchaudio(
    host: RemoteHost,
    *,
    branch: str = "main",
    use_conda: bool = True,
    git_clone_flags: str = "",
) -> str:
    print("Checking out TorchAudio repo")
    git_clone_flags += " --recurse-submodules"
    build_version = checkout_repo(
        host,
        branch=branch,
        url="https://github.com/pytorch/audio",
        git_clone_flags=git_clone_flags,
        mapping={
            "v1.9.0": ("0.9.0", "rc2"),
            "v1.10.0": ("0.10.0", "rc5"),
            "v1.10.1": ("0.10.1", "rc1"),
            "v1.10.2": ("0.10.2", "rc1"),
            "v1.11.0": ("0.11.0", "rc1"),
            "v1.12.0": ("0.12.0", "rc3"),
            "v1.12.1": ("0.12.1", "rc5"),
            "v1.13.0": ("0.13.0", "rc4"),
            "v1.13.1": ("0.13.1", "rc2"),
            "v2.0.0": ("2.0.1", "rc3"),
            "v2.0.1": ("2.0.2", "rc2"),
        },
    )
    print("Building TorchAudio wheel")
    build_vars = ""
    if branch == "nightly":
        version = (
            host.check_output(["grep", '"version = \'"', "audio/setup.py"])
            .strip()
            .split("'")[1][:-2]
        )
        build_date = (
            host.check_output("cd audio && git log --pretty=format:%s -1")
            .strip()
            .split()[0]
            .replace("-", "")
        )
        build_vars += f"BUILD_VERSION={version}.dev{build_date}"
    elif build_version is not None:
        build_vars += f"BUILD_VERSION={build_version} PYTORCH_VERSION={branch[1:].split('-', maxsplit=1)[0]}"
    if host.using_docker():
        build_vars += " CMAKE_SHARED_LINKER_FLAGS=-Wl,-z,max-page-size=0x10000"

    host.run_cmd(
        f"cd audio && export FFMPEG_ROOT=$(pwd)/third_party/ffmpeg && export USE_FFMPEG=1 \
        && ./packaging/ffmpeg/build.sh \
        && {build_vars} python3 -m build --wheel --no-isolation"
    )

    wheel_name = host.list_dir("audio/dist")[0]
    embed_libgomp(host, use_conda, os.path.join("audio", "dist", wheel_name))

    print("Copying TorchAudio wheel")
    host.download_wheel(os.path.join("audio", "dist", wheel_name))

    return wheel_name


def configure_system(
    host: RemoteHost,
    *,
    compiler: str = "gcc-8",
    use_conda: bool = True,
    python_version: str = "3.8",
) -> None:
    if use_conda:
        install_condaforge_python(host, python_version)

    print("Configuring the system")
    if not host.using_docker():
        update_apt_repo(host)
        host.run_cmd("sudo apt-get install -y ninja-build g++ git cmake gfortran unzip")
    else:
        host.run_cmd("yum install -y sudo")
        host.run_cmd("conda install -y ninja scons")

    if not use_conda:
        host.run_cmd(
            "sudo apt-get install -y python3-dev python3-yaml python3-setuptools python3-wheel python3-pip"
        )
    host.run_cmd("pip3 install dataclasses typing-extensions")
    if not use_conda:
        print("Installing Cython + numpy from PyPy")
        host.run_cmd("sudo pip3 install Cython")
        host.run_cmd("sudo pip3 install numpy")


def build_domains(
    host: RemoteHost,
    *,
    branch: str = "main",
    use_conda: bool = True,
    git_clone_flags: str = "",
) -> tuple[str, str, str, str]:
    vision_wheel_name = build_torchvision(
        host, branch=branch, use_conda=use_conda, git_clone_flags=git_clone_flags
    )
    audio_wheel_name = build_torchaudio(
        host, branch=branch, use_conda=use_conda, git_clone_flags=git_clone_flags
    )
    data_wheel_name = build_torchdata(
        host, branch=branch, use_conda=use_conda, git_clone_flags=git_clone_flags
    )
    text_wheel_name = build_torchtext(
        host, branch=branch, use_conda=use_conda, git_clone_flags=git_clone_flags
    )
    return (vision_wheel_name, audio_wheel_name, data_wheel_name, text_wheel_name)


def start_build(
    host: RemoteHost,
    *,
    branch: str = "main",
    compiler: str = "gcc-8",
    use_conda: bool = True,
    python_version: str = "3.8",
    pytorch_only: bool = False,
    pytorch_build_number: Optional[str] = None,
    shallow_clone: bool = True,
    enable_mkldnn: bool = False,
) -> tuple[str, str, str, str, str]:
    git_clone_flags = " --depth 1 --shallow-submodules" if shallow_clone else ""
    if host.using_docker() and not use_conda:
        print("Auto-selecting conda option for docker images")
        use_conda = True
    if not host.using_docker():
        print("Disable mkldnn for host builds")
        enable_mkldnn = False

    configure_system(
        host, compiler=compiler, use_conda=use_conda, python_version=python_version
    )

    if host.using_docker():
        print("Move libgfortant.a into a standard location")
        # HACK: pypa gforntran.a is compiled without PIC, which leads to the following error
        # libgfortran.a(error.o)(.text._gfortrani_st_printf+0x34): unresolvable R_AARCH64_ADR_PREL_PG_HI21 relocation against symbol `__stack_chk_guard@@GLIBC_2.17'  # noqa: E501, B950
        # Workaround by copying gfortran library from the host
        host.run_ssh_cmd("sudo apt-get install -y gfortran-8")
        host.run_cmd("mkdir -p /usr/lib/gcc/aarch64-linux-gnu/8")
        host.run_ssh_cmd(
            [
                "docker",
                "cp",
                "/usr/lib/gcc/aarch64-linux-gnu/8/libgfortran.a",
                f"{host.container_id}:/opt/rh/devtoolset-10/root/usr/lib/gcc/aarch64-redhat-linux/10/",
            ]
        )

    print("Checking out PyTorch repo")
    host.run_cmd(
        f"git clone --recurse-submodules -b {branch} https://github.com/pytorch/pytorch {git_clone_flags}"
    )

    host.run_cmd("pytorch/.ci/docker/common/install_openblas.sh")

    print("Building PyTorch wheel")
    build_opts = ""
    if pytorch_build_number is not None:
        build_opts += f" -C--build-option=--build-number={pytorch_build_number}"
    # Breakpad build fails on aarch64
    build_vars = "USE_BREAKPAD=0 "
    if branch == "nightly":
        build_date = (
            host.check_output("cd pytorch && git log --pretty=format:%s -1")
            .strip()
            .split()[0]
            .replace("-", "")
        )
        version = host.check_output("cat pytorch/version.txt").strip()[:-2]
        build_vars += f"BUILD_TEST=0 PYTORCH_BUILD_VERSION={version}.dev{build_date} PYTORCH_BUILD_NUMBER=1"
    if branch.startswith(("v1.", "v2.")):
        build_vars += f"BUILD_TEST=0 PYTORCH_BUILD_VERSION={branch[1 : branch.find('-')]} PYTORCH_BUILD_NUMBER=1"
    if host.using_docker():
        build_vars += " CMAKE_SHARED_LINKER_FLAGS=-Wl,-z,max-page-size=0x10000"
    if enable_mkldnn:
        host.run_cmd("pytorch/.ci/docker/common/install_acl.sh")
        print("build pytorch with mkldnn+acl backend")
        build_vars += " USE_MKLDNN=ON USE_MKLDNN_ACL=ON"
        build_vars += " BLAS=OpenBLAS"
        build_vars += " OpenBLAS_HOME=/opt/OpenBLAS"
        build_vars += " ACL_ROOT_DIR=/acl"
        host.run_cmd(
            f"cd $HOME/pytorch && {build_vars} python3 -m build --wheel --no-isolation{build_opts}"
        )
        print("Repair the wheel")
        pytorch_wheel_name = host.list_dir("pytorch/dist")[0]
        ld_library_path = "/acl/build:$HOME/pytorch/build/lib"
        host.run_cmd(
            f"export LD_LIBRARY_PATH={ld_library_path} && auditwheel repair $HOME/pytorch/dist/{pytorch_wheel_name}"
        )
        print("replace the original wheel with the repaired one")
        pytorch_repaired_wheel_name = host.list_dir("wheelhouse")[0]
        host.run_cmd(
            f"cp $HOME/wheelhouse/{pytorch_repaired_wheel_name} $HOME/pytorch/dist/{pytorch_wheel_name}"
        )
    else:
        print("build pytorch without mkldnn backend")
        host.run_cmd(
            f"cd pytorch && {build_vars} python3 -m build --wheel --no-isolation{build_opts}"
        )

    print("Deleting build folder")
    host.run_cmd("cd pytorch && rm -rf build")
    pytorch_wheel_name = host.list_dir("pytorch/dist")[0]
    embed_libgomp(host, use_conda, os.path.join("pytorch", "dist", pytorch_wheel_name))
    print("Copying the wheel")
    host.download_wheel(os.path.join("pytorch", "dist", pytorch_wheel_name))

    print("Installing PyTorch wheel")
    host.run_cmd(f"pip3 install pytorch/dist/{pytorch_wheel_name}")

    if pytorch_only:
        return (pytorch_wheel_name, None, None, None, None)
    domain_wheels = build_domains(
        host, branch=branch, use_conda=use_conda, git_clone_flags=git_clone_flags
    )

    return (pytorch_wheel_name, *domain_wheels)


embed_library_script = """
#!/usr/bin/env python3

from auditwheel.patcher import Patchelf
from auditwheel.wheeltools import InWheelCtx
from auditwheel.elfutils import elf_file_filter
from auditwheel.repair import copylib
from auditwheel.lddtree import lddtree
from subprocess import check_call
import os
import shutil
import sys
from tempfile import TemporaryDirectory


def replace_tag(filename):
   with open(filename, 'r') as f:
     lines = f.read().split("\\n")
   for i,line in enumerate(lines):
       if not line.startswith("Tag: "):
           continue
       lines[i] = line.replace("-linux_", "-manylinux2014_")
       print(f'Updated tag from {line} to {lines[i]}')

   with open(filename, 'w') as f:
       f.write("\\n".join(lines))


class AlignedPatchelf(Patchelf):
    def set_soname(self, file_name: str, new_soname: str) -> None:
        check_call(['patchelf', '--page-size', '65536', '--set-soname', new_soname, file_name])

    def replace_needed(self, file_name: str, soname: str, new_soname: str) -> None:
        check_call(['patchelf', '--page-size', '65536', '--replace-needed', soname, new_soname, file_name])


def embed_library(whl_path, lib_soname, update_tag=False):
    patcher = AlignedPatchelf()
    out_dir = TemporaryDirectory()
    whl_name = os.path.basename(whl_path)
    tmp_whl_name = os.path.join(out_dir.name, whl_name)
    with InWheelCtx(whl_path) as ctx:
        torchlib_path = os.path.join(ctx._tmpdir.name, 'torch', 'lib')
        ctx.out_wheel=tmp_whl_name
        new_lib_path, new_lib_soname = None, None
        for filename, elf in elf_file_filter(ctx.iter_files()):
            if not filename.startswith('torch/lib'):
                continue
            libtree = lddtree(filename)
            if lib_soname not in libtree['needed']:
                continue
            lib_path = libtree['libs'][lib_soname]['path']
            if lib_path is None:
                print(f"Can't embed {lib_soname} as it could not be found")
                break
            if lib_path.startswith(torchlib_path):
                continue

            if new_lib_path is None:
                new_lib_soname, new_lib_path = copylib(lib_path, torchlib_path, patcher)
            patcher.replace_needed(filename, lib_soname, new_lib_soname)
            print(f'Replacing {lib_soname} with {new_lib_soname} for {filename}')
        if update_tag:
            # Add manylinux2014 tag
            for filename in ctx.iter_files():
                if os.path.basename(filename) != 'WHEEL':
                    continue
                replace_tag(filename)
    shutil.move(tmp_whl_name, whl_path)


if __name__ == '__main__':
    embed_library(sys.argv[1], 'libgomp.so.1', len(sys.argv) > 2 and sys.argv[2] == '--update-tag')
"""


def run_tests(host: RemoteHost, whl: str, branch="main") -> None:
    print("Configuring the system")
    update_apt_repo(host)
    host.run_cmd("sudo apt-get install -y python3-pip git")
    host.run_cmd("sudo pip3 install Cython")
    host.run_cmd("sudo pip3 install numpy")
    host.upload_file(whl, ".")
    host.run_cmd(f"sudo pip3 install {whl}")
    host.run_cmd("python3 -c 'import torch;print(torch.rand((3,3))'")
    host.run_cmd(f"git clone -b {branch} https://github.com/pytorch/pytorch")
    host.run_cmd("cd pytorch/test; python3 test_torch.py -v")


def get_instance_name(instance) -> Optional[str]:
    if instance.tags is None:
        return None
    for tag in instance.tags:
        if tag["Key"] == "Name":
            return tag["Value"]
    return None


def list_instances(instance_type: str) -> None:
    print(f"All instances of type {instance_type}")
    for instance in ec2_instances_of_type(instance_type):
        ifaces = instance.network_interfaces
        az = ifaces[0].subnet.availability_zone if len(ifaces) > 0 else None
        print(
            f"{instance.id} {get_instance_name(instance)} {instance.public_dns_name} {instance.state['Name']} {az}"
        )


def terminate_instances(instance_type: str) -> None:
    print(f"Terminating all instances of type {instance_type}")
    instances = list(ec2_instances_of_type(instance_type))
    for instance in instances:
        print(f"Terminating {instance.id}")
        instance.terminate()
    print("Waiting for termination to complete")
    for instance in instances:
        instance.wait_until_terminated()


def parse_arguments():
    from argparse import ArgumentParser

    parser = ArgumentParser("Build and test AARCH64 wheels using EC2")
    parser.add_argument("--key-name", type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--test-only", type=str)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--os", type=str, choices=list(os_amis.keys()))
    group.add_argument("--ami", type=str)
    parser.add_argument(
        "--python-version",
        type=str,
        choices=[f"3.{d}" for d in range(6, 12)],
        default=None,
    )
    parser.add_argument("--alloc-instance", action="store_true")
    parser.add_argument("--list-instances", action="store_true")
    parser.add_argument("--pytorch-only", action="store_true")
    parser.add_argument("--keep-running", action="store_true")
    parser.add_argument("--terminate-instances", action="store_true")
    parser.add_argument("--instance-type", type=str, default="t4g.2xlarge")
    parser.add_argument("--ebs-size", type=int, default=50)
    parser.add_argument("--branch", type=str, default="main")
    parser.add_argument("--use-docker", action="store_true")
    parser.add_argument(
        "--compiler",
        type=str,
        choices=["gcc-7", "gcc-8", "gcc-9", "clang"],
        default="gcc-8",
    )
    parser.add_argument("--use-torch-from-pypi", action="store_true")
    parser.add_argument("--pytorch-build-number", type=str, default=None)
    parser.add_argument("--disable-mkldnn", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    ami = (
        args.ami
        if args.ami is not None
        else os_amis[args.os]
        if args.os is not None
        else ubuntu20_04_ami
    )
    keyfile_path, key_name = compute_keyfile_path(args.key_name)

    if args.list_instances:
        list_instances(args.instance_type)
        sys.exit(0)

    if args.terminate_instances:
        terminate_instances(args.instance_type)
        sys.exit(0)

    if len(key_name) == 0:
        raise RuntimeError("""
            Cannot start build without key_name, please specify
            --key-name argument or AWS_KEY_NAME environment variable.""")
    if len(keyfile_path) == 0 or not os.path.exists(keyfile_path):
        raise RuntimeError(f"""
            Cannot find keyfile with name: [{key_name}] in path: [{keyfile_path}], please
            check `~/.ssh/` folder or manually set SSH_KEY_PATH environment variable.""")

    # Starting the instance
    inst = start_instance(
        key_name, ami=ami, instance_type=args.instance_type, ebs_size=args.ebs_size
    )
    instance_name = f"{args.key_name}-{args.os}"
    if args.python_version is not None:
        instance_name += f"-py{args.python_version}"
    inst.create_tags(
        DryRun=False,
        Tags=[
            {
                "Key": "Name",
                "Value": instance_name,
            }
        ],
    )
    addr = inst.public_dns_name
    wait_for_connection(addr, 22)
    host = RemoteHost(addr, keyfile_path)
    host.ami = ami
    if args.use_docker:
        update_apt_repo(host)
        host.start_docker()

    if args.test_only:
        run_tests(host, args.test_only)
        sys.exit(0)

    if args.alloc_instance:
        if args.python_version is None:
            sys.exit(0)
        install_condaforge_python(host, args.python_version)
        sys.exit(0)

    python_version = args.python_version if args.python_version is not None else "3.10"

    if args.use_torch_from_pypi:
        configure_system(host, compiler=args.compiler, python_version=python_version)
        print("Installing PyTorch wheel")
        host.run_cmd("pip3 install torch")
        build_domains(
            host, branch=args.branch, git_clone_flags=" --depth 1 --shallow-submodules"
        )
    else:
        start_build(
            host,
            branch=args.branch,
            compiler=args.compiler,
            python_version=python_version,
            pytorch_only=args.pytorch_only,
            pytorch_build_number=args.pytorch_build_number,
            enable_mkldnn=not args.disable_mkldnn,
        )
    if not args.keep_running:
        print(f"Waiting for instance {inst.id} to terminate")
        inst.terminate()
        inst.wait_until_terminated()
