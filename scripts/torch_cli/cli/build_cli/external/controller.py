from cement import Controller, ex
from cli.lib.utils import generate_dataclass_help
from cli.build_cli.external.vllm_build import build_vllm, VllmBuildConfig


class ExternalBuildController(Controller):
    class Meta:
        label = "build_external"
        aliases = ["external"]
        stacked_on = "build"
        stacked_type = "nested"
        help = (
            "Build external libraries with pytorch ci"
        )

    @ex(
        help=f"Build vLLM external library with docker image, environment variables:{generate_dataclass_help(VllmBuildConfig)}",
        arguments=[
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
                    "help": "Directory where build artifacts will be saved",
                    "dest": "artifact_dir",
                    "default": "./results",
                    "type": str,
                    "required": False,
                },
            ),
            (
                ["--base-image"],
                {
                    "help": "Base image used for building",
                    "dest": "base_image",
                    "default": "",
                    "type": str,
                    "required": False,
                },
            ),
        ],
    )
    def vllm(self):
        pargs = self.app.pargs
        build_vllm(
            artifact_dir=pargs.artifact_dir,
            torch_whl_dir=pargs.torch_whl_dir,
            base_image=pargs.base_image
        )
