from cement import App, Controller, ex


class TestController(Controller):
    class Meta:
        label = "test"
        stacked_on = "base"
        stacked_type = "nested"
        description = "Build CLI group"
