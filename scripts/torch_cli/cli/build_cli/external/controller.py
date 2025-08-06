from cement import Controller, ex

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
        if config:
            self.app.info("use config filer user provided")
        else:
            self.app.info("please input config file, otherwise use default config")
        # TODO: implement vllm build
        self.app.info("implement vllm build")
