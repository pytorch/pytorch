from cement import App, Controller, ex
from test_cli.controllers import ALL_CONTROLLERS


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
        handlers = [MainController] + ALL_CONTROLLERS

def main():
    with BuildApp() as app:
        app.run()

if __name__ == "__main__":
    main()
