from cement import Controller, ex
from cli.lib.utils import read_yaml_file


class ExternalBuildController(Controller):
    class Meta:
        label = "build_external"  # must be unique among all controllers
        aliases = ["external"]  # alias for the controller
        stacked_on = "build"
        stacked_type = "nested"
        help = "Build external libraries with pytorch ci"

    @ex(
        help='Build vllm,  Example: python3 -m cli.run --config".github/ci_configs/CONFIG_TEMPLATE.yaml" build external vllm',
    )
    def vllm(self):
        config = self.app.pargs.config
        config_map = {}
        if config:
            self.app.info("use config file user provided")
            self.app.info("Reading config yaml file ...")
            config_map = read_yaml_file(config, self.app)
            self.app.info(f"config_map: {config_map}")
        else:
            self.app.info("please input config file, otherwise use default config")
        self.app.info("implement vllm build")
