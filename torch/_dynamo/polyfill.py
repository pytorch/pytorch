"""
Python polyfills for common builtins.
"""


def all(iterator):
    for elem in iterator:
        if not elem:
            return False
    return True


def index(iterator, item, start=0, end=-1):
    for i, elem in enumerate(list(iterator))[start:end]:
        if item == elem:
            return i
    # This will not run in dynamo
    raise ValueError(f"{item} is not in {type(iterator)}")


def repeat(item, count):
    for i in range(count):
        yield item


def zip(*iterators):
    iterators = [iter(i) for i in iterators]
    while True:
        # This should not yield anything if any next
        # raises `StopIteration`
        nexts = tuple(next(i) for i in iterators)
        yield nexts


def next_p(iterable):
    return iterable.__next__()
