from cement import App, Controller
from build_cli.external.controller import ExternalBuildController


class MainController(Controller):
    class Meta:
        label = "base"
        help = "base for build CLI"

class BuildApp(App):
    class Meta:
        label = "build"
        base_controller = "base"
        handlers = [MainController] + ExternalBuildController

def main():
    with BuildApp() as app:
        app.run()

if __name__ == "__main__":
    main()
