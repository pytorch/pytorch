import glob
import os
import shlex
import shutil
import subprocess
import tempfile
import sys
from pathlib import Path

from lib.utils import clone_vllm, get_post_build_pinned_commit, read_yaml_file, run
import os

class VllmTestRunner:
    def __init__(self, file_path="") -> None:
        self.test_configs = self._fetch_configs(file_path)

    def run(self, test_names):
        valid_tests = []
        for test_name in test_names:
            if test_name not in self.test_configs:
                print(
                    f"[warning] cannot detect test name {test_name}, please input valid test name "
                )
                continue
            config = self.test_configs.get(test_name)
            valid_tests.append(config)
        os.chdir("vllm")
        for config in valid_tests:
            self.test(config)
        os.chdir("..")

    def test(self, config={}):
        testid = config["id"]
        steps = config["steps"]
        sub_path = config.get("path", ".")
        print(f"running test config: {testid}")
        for step in steps:
            run(step, cwd=sub_path, logging=True)

    def _fetch_configs(self, path=""):
        base_dir = os.path.dirname(__file__)
        file_path = path if path else os.path.join(base_dir, "test_config.yaml")
        res = read_yaml_file(file_path)
        config_map = {}
        for item in res:
            if "id" in item:
                config_map[item["id"]] = item
            else:
                raise ValueError(f"Missing 'id' in config: {item}")

        print(f"config_map: {config_map}")
        return config_map

    def prepare_test_env(self):
        """
        prepare vllm test env
        this includes:
          - clone vllm repo
          - install test necessary dependencies
          - install whls from previous job
        """
        clone_vllm(get_post_build_pinned_commit("vllm"))
        os.chdir("vllm")
        os.environ["UV_INDEX_STRATEGY"] = "unsafe-best-match"
        self.install_test_base()
        os.chdir("..")
        self.install_local_whls()

        os.chdir("vllm")
        self.generated_test_txt()
        install_packages("test.txt")
        run("cat test.txt")
        install('--no-build-isolation "git+https://github.com/state-spaces/mamba@v2.2.4"')
        install("hf_transfer")
        run("pip freeze | grep -E 'torch|xformers|torchvision|torchaudio|flashinfer'")
        os.chdir("..")

    def install_test_base(self):
        run("cp vllm/collect_env.py .")
        # remove  vllm/vllm
        if os.path.exists("vllm"):
            print("Removing 'vllm' directory...")
            shutil.rmtree("vllm")
        run("python3 use_existing_torch.py")
        run("cat requirements/common.txt")
        run("cat requirements/build.txt")
        install_packages("requirements/common.txt")
        install_packages("requirements/build.txt")
        print("done installing test base")

    def install_local_whls(self):
        torch = "dist/torch-*.whl"
        local_whls = [
            "dist/vision/torchvision*.whl",
            "dist/audio/torchaudio*.whl",
            "wheels/xformers/xformers*.whl",
            "wheels/flashinfer-python/flashinfer*.whl",
        ]
        run("pwd")
        torch_match = glob.glob(torch)[0]
        if not torch_match:
            run("pwd")
            run("ls")
            run("ls -al dist/")
            raise ValueError(f"No match for: {torch}")

        print(f"[INFO] Installing: {torch_match}")
        run(f"python3 -m pip install '{torch_match}[opt-einsum]'")

        for pattern in local_whls:
            matches = glob.glob(pattern)
            if not matches:
                print(f"[WARN] No match for: {pattern}")
                continue
            whl_path = matches[0]
            print(f"[INFO] Installing: {whl_path}")
            run(f"pip install {shlex.quote(whl_path)}")
        print("done installing test base")

    def generated_test_txt(
        self, target_file: str = "requirements/test.in", res_file="test.txt"
    ):
        """
        read directly from vllm's test.in to generate compilable test.txt for pip install.
        clean the torch dependencies, replace with whl locations, then generate the test.txt
        """
        # remove torch dependencies
        clean_torch_dependecies()
        pkgs = ["torch", "torchvision", "torchaudio", "xformers", "flashinfer-python"]

        tmp_head_path = Path(tempfile.mkstemp()[1])
        with tmp_head_path.open("w") as tmp_head:
            for pkg in pkgs:
                try:
                    result = subprocess.run(
                        ["pip", "freeze"], check=True, stdout=subprocess.PIPE, text=True
                    )
                    lines = [
                        line
                        for line in result.stdout.splitlines()
                        if line.startswith(pkg) and "@ file://" in line
                    ]
                    tmp_head.writelines(line + "\n" for line in lines)
                except subprocess.CalledProcessError:
                    print(f"[WARN] Failed to get freeze info for {pkg}")
            tmp_head.write("\n")
            # Append original test.in
            with open(target_file) as tf:
                tmp_head.writelines(tf.readlines())
        shutil.move(str(tmp_head_path), target_file)
        print(f"[INFO] Local wheel requirements prepended to {target_file}")
        uv_pip_compile(target_file, res_file,"--index-strategy", "unsafe-best-match")

def clean_torch_dependecies(requires_files=["requirements/test.in"]):
    # Keywords to match exactly
    keywords_to_remove = ["torch==", "torchaudio==", "torchvision==", "mamba_ssm"]
    for file in requires_files:
        print(f">>> cleaning {file}")
        with open(file) as f:
            lines = f.readlines()
        cleaned_lines = []
        for line in lines:
            line_lower = line.strip().lower()
            if any(line_lower.startswith(kw) for kw in keywords_to_remove):
                print("removed:", line.strip())
            else:
                cleaned_lines.append(line)
        print(f"<<< done cleaning {file}\n")

def ensure_uv():
    uv_path = shutil.which("uv")
    if uv_path:
        print(f"[INFO] Found uv at {uv_path}")
        return uv_path
    else:
        print("[INFO] uv not found, installing uv via pip...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "uv"], check=True)
        uv_path = shutil.which("uv")
        if not uv_path:
            raise RuntimeError("Failed to install uv")
        return uv_path

def in_venv():
    return sys.prefix != sys.base_prefix

def uv_pip_install( args: str =""):
    uv = ensure_uv()
    args_list = shlex.split(args)
    cmd_list = [uv, "pip","install"]+ args_list
    cmd_str = shlex.join(cmd_list)  # 安全打印
    run(cmd_str)  # 你自己的 run(str)
    run(cmd_str)

def uv_pip_compile(input_file: str, output_file: str, additional_args: str = "", k:str=""):
    uv = ensure_uv()
    args = ["pip", "compile", input_file, "-o", output_file, additional_args]
    cmd = shlex.join([uv] + args)
    print(f"[INFO] Running: {cmd}")
    run(cmd)

def compile(input_file: str, output_file: str, additional_args: str = ""):
    uv = ensure_uv()
    os.system(f"{uv} pip compile --system {input_file} -o {output_file} {additional_args}")

def install_packages(package: str):
    uv = ensure_uv()
    os.system(f"{uv} pip install --system -r {package}")

def install(package: str):
    uv = ensure_uv()
    os.system(f"{uv} pip install --system {package}")
