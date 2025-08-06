from cement import Controller, ex
from cli.lib.vllm_utils import build_vllm
from cli.lib.utils import generate_dataclass_help, read_yaml_file


class ExternalBuildController(Controller):
    class Meta:
        label = "build_external"
        aliases = ["external"]
        stacked_on = "build"
        stacked_type = "nested"
        help = "Build external libraries with pytorch ci"

    @ex(
        help="Build vllm",
    )
    def vllm(self):
        config = self.app.pargs.config
        config_map = {}
        if config:
            print("please input config file, otherwise use default config")
            config_map = read_yaml_file(config)
        build_vllm(config=config_map)
