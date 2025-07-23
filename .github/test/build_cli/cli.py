from cement import App, Controller
from controllers import ALL_CONTROLLERS

class MainController(Controller):
    class Meta:
        label = "base"
        help = "pt2-bm-cli: PyTorch Benchmark CLI"

class BuildApp(App):
    class Meta:
        label = 'build'
        base_controller = 'base'
        handlers = [MainController] + ALL_CONTROLLERS

def main():
    with BuildApp() as app:
        app.run()

if __name__ == '__main__':
    main()
