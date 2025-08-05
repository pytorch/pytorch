from cement import Controller, ex
from cli.lib.utils import generate_dataclass_help, read_yaml_file
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
        help=f"Build external vllm library with docker build command, environment variables:{generate_dataclass_help(VllmBuildConfig)}",
        arguments=[
            (
                ["--config"],
                {
                    "help": "Path to a config file to define the build parameters",
                    "dest": "config",
                    "type": str,
                    "required": False,
                }
            ),
        ],
    )
    def vllm(self):
        pargs = self.app.pargs
        config_dir = pargs.config
        config_map = read_yaml_file(config_dir)
        build_vllm(config = config_map)
