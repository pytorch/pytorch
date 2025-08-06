from cement import App, Controller, ex
from cli.build_cli import BUILD_CONTROLLERS


class MainController(Controller):
    class Meta:
        label = "base"
        description = "Main entry point"
        arguments = [
            (
                ["--config"],
                {
                    "help": "Path to config file for ci build and test",
                    "dest": "config",
                    "default": "",
                    "required": False,
                },
            ),
        ]

class TorchCli(App):
    class Meta:
        label = "cli"
        base_controller = "base"
        log_handler = 'logging'
        log_level = 'INFO'
        handlers = [MainController] + BUILD_CONTROLLERS

    def info(self, msg):
        self.log.info(msg)

    def debug(self, msg):
        self.log.debug(msg)

    def warn(self, msg):
        self.log.warning(msg)

    def error(self, msg):
        self.log.error(msg)

def main():
    with TorchCli() as app:
        app.run()


if __name__ == "__main__":
    main()
