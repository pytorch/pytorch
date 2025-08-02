from cement import App, Controller, ex
from cli.build import BUILD_CONTROLLERS
from cli.test import TEST_CONTROLLERS


class MainController(Controller):
    class Meta:
        label = "base"
        description = "Main entry point"


class TorchCli(App):
    class Meta:
        label = "cli"
        base_controller = "base"
        handlers = [MainController] + TEST_CONTROLLERS + BUILD_CONTROLLERS


def main():
    with TorchCli() as app:
        app.run()


if __name__ == "__main__":
    main()
