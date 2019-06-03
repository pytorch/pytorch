from widget_module import Widget


class DerivedWidget(Widget):
    def __init__(self, message):
        super(DerivedWidget, self).__init__(message)

    def the_answer(self):
        return 42
