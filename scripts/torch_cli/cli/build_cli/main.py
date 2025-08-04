from cement import Controller

class BuildController(Controller):
    class Meta:
        label = "build"
        stacked_on = "base"
        stacked_type = "nested"
        description = "Build CLI group"
