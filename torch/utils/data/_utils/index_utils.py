from math import sqrt, floor


class Range:
    size: int

    # for slicing
    start: int
    stop: int
    step: int

    # for iterating
    index: int

    def __init__(self, size: int, start: int = None, stop: int = None, step: int = None):
        self.size = size
        self.start = start if start is not None else 0
        self.stop = stop if stop is not None else size
        self.step = step if step is not None else 1
        if not (0 <= self.start <= self.stop):
            raise ValueError('not (0 <= start <= stop)')
        if self.stop > self.size:
            pass  # Allow wrap-around
        if self.step <= 0:
            raise ValueError('step <= 0')
        if (size == 0) != (len(self) == 0):
            raise ValueError('if size==0 then stop-start must also be 0')

    def __len__(self):
        if self.start == self.stop:
            return 0
        return (self.stop - self.start - 1) // self.step + 1

    def __getitem__(self, index):
        if isinstance(index, slice):
            sl = slice(index.start if index.start is not None else 0,
                       index.stop if index.stop is not None else len(self),
                       index.step if index.step is not None else 1)
            if not (0 <= sl.start and sl.start <= sl.stop and 0 < sl.step):
                raise IndexError
            return self._slice(sl)

        if not 0 <= index < len(self):
            raise IndexError

        abs_index = (self.start + index * self.step) % self.size

        return self._get(abs_index)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index == len(self):
            raise StopIteration
        next = self[self.index]
        self.index = self.index + 1
        return next

    def _get(self, index):
        return index

    def _slice(self, sl):
        return Range(self.size,
                     self.start + sl.start * self.step,
                     self.start + sl.stop * self.step,
                     self.step * sl.step)


# We can use a faster primehood test, but there is no readily available implementation.
# But even this brute-force test is reasonably fast, O(sqrt(n))
def _is_prime(n):
    if n == 2:
        return True
    if n == 1 or n % 2 == 0:
        return False

    for d in range(3, floor(sqrt(n)) + 1, 2):  # can use isqrt in Python 3.8
        if n % d == 0:
            return False

    return True


class Permutation(Range):
    """
    Generates a random permutation of integers from 0 up to size.
    Inspired by https://preshing.com/20121224/how-to-generate-a-sequence-of-unique-random-integers/
    """

    prime: int
    seed: int

    def __init__(self, size: int, seed: int, start: int = None, stop: int = None, step: int = None, _prime: int = None):
        super().__init__(size, start, stop, step)
        self.prime = self._get_prime(size) if _prime is None else _prime
        self.seed = seed % self.prime

    def _get(self, index):
        x = self._map(index)

        while x >= self.size:
            # If we map to a number greater than size, then the cycle of successive mappings must eventually result
            # in a number less than size. Proof: The cycle of successive mappings traces a path
            # that either always stays in the set n>=size or it enters and leaves it,
            # else the 1:1 mapping would be violated (two numbers would map to the same number).
            # Moreover, `set(range(size)) - set(map(n) for n in range(size) if map(n) < size)`
            # equals the `set(map(n) for n in range(size, prime) if map(n) < size)`
            # because the total mapping is exhaustive.
            # Which means we'll arrive at a number that wasn't mapped to by any other valid index.
            # This will take at most `prime-size` steps, and `prime-size` is on the order of log(size), so fast.
            # But usually we just need to remap once.
            x = self._map(x)

        return x

    def _slice(self, sl):
        return Permutation(self.size,
                           self.seed,
                           self.start + sl.start * self.step,
                           self.start + sl.stop * self.step,
                           self.step * sl.step,
                           self.prime)

    @staticmethod
    def _get_prime(size):
        """
        Returns the prime number >= size which has the form (4n-1)
        """
        n = size + (3 - size % 4)
        while not _is_prime(n):
            # We expect to find a prime after O(log(size)) iterations
            # Using a brute-force primehood test, total complexity is O(log(size)*sqrt(size)), which is pretty good.
            n = n + 4
        return n

    def _map(self, index):
        a = self._permute_qpr(index)
        b = (a + self.seed) % self.prime
        c = self._permute_qpr(b)
        return c

    def _permute_qpr(self, x):
        residue = pow(x, 2, self.prime)

        if x * 2 < self.prime:
            return residue
        else:
            return self.prime - residue
