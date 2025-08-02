from cement import App, Controller, ex
from test_cli.external.controller import ExternalTestController


class MainController(Controller):
    class Meta:
        label = "base"
        help = "base for build CLI"

    @ex(help="Show")
    def hello(self):
        print("hello world")

class BuildApp(App):
    class Meta:
        label = "build"
        base_controller = "base"
        handlers = [MainController, ExternalTestController]

def main():
    with BuildApp() as app:
        app.run()

if __name__ == "__main__":
    main()
