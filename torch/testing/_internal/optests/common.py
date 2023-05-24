# Exception class for if a test fails because of a problem with the test
# (and not a problem with the function it is testing).
class TestFrameworkError(Exception):
    pass
