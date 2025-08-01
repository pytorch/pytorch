from cement import Controller, ex
from lib.utils import generate_dataclass_help

from .vllm_build import build_vllm, VllmBuildConfig


class VllmBuildController(Controller):
    class Meta:
        label = "vllm"
        stacked_on = "base"
        stacked_type = "nested"
        help = (
            "Build external libraries. to run the build, do cli.py external --target vllm run, to see the details of help: \
        cli.py external --target vllm help "
        )
        arguments = [
            (
                ["--torch-whl-dir"],
                {
                    "help": "Path to a local folder where torch wheel is located",
                    "dest": "torch_whl_dir",
                    "type": str,
                    "required": False,
                },
            ),
            (
                ["--artifact-dir"],
                {
                    "help": "Path to a local folder where artifacts from external builds will be stored",
                    "dest": "artifact_dir",
                    "default": "./results",
                    "type": str,
                    "required": False,
                },
            ),
            (
                ["--base-image"],
                {
                    "help": "Path to a local folder where artifacts from external builds will be stored",
                    "dest": "base_image",
                    "default": "",
                    "type": str,
                    "required": False,
                },
            ),
        ]

    @ex(help="Build vllm")
    def run(self):
        pargs = self.app.pargs
        base_image = pargs.base_image
        torch_whl_dir = pargs.torch_whl_dir
        artifact_dir = self.app.pargs.artifact_dir
        build_vllm(artifact_dir, torch_whl_dir, base_image)

    @ex(help="Show detailed help for vllm")
    def help(self):
        print("[HELP] Extended help for target `vllm`")
        print()
        print("These environment variables are used in the build with default value:")
        print(generate_dataclass_help(VllmBuildConfig))
        print()
        print("Example usage:")
        print("    TAG=nightly-vllm CUDA_VERSION=12.8.0 cli.py external --target vllm")
