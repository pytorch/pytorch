from cli.build_cli.external.controller import ExternalBuildController
from cli.build_cli.main import BuildController

# All the build controllers that can be registered in cement app
BUILD_CONTROLLERS = [BuildController, ExternalBuildController]
