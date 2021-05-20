class GoodPickle:
    def __repr__(self):
        return "GoodPickle"

class BadPickle:
    def __getstate__(self):
        raise TypeError("I can't be pickled!")

    def __repr__(self):
        return "BadPickle"
