r"""
Define exceptions used in test_exception.py. We define them in a
separate file on purpose to make sure the fully qualified exception class name
is captured correctly in suce cases.
"""
class MyKeyError(KeyError):
    def __init__(self, msg):
        super(KeyError, self).__init__(msg)
