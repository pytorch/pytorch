# Only used for PyTorch open source BUCK build

def expect(condition, message = None):
    if not condition:
        fail(message)
