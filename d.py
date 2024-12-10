import contextlib
import torch

@contextlib.contextmanager
def transaction():
    # print('begin')
    try:
        yield from do_it()
    except:
        # print('rollback')
        raise
    else:
        pass
        # print('commit')

def do_it():
    # print('Refactored initial setup')
    yield # Body of with-statement is executed here
    # print('Refactored finalization of successful transaction')

def gene():
    for i in range(2):
        with transaction():
            yield i
            # return
            raise StopIteration  # This is wrong
        # print('Should not be reached')

@torch.compile(backend='eager', fullgraph=True)
def fn(t):
    try:
        for i in gene():
            t += i
    except StopIteration:
        assert False
        # print('main: i =', i)

t = torch.randn(2)
fn(t)