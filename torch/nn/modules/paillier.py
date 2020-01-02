from .module import Module
from torch import rand_like, pow as tpow, fmod


class Paillier(Module):
    # from phe.util import invert, powmod, getprimeover, isqrt
    def __init__(self, p, q):
        super(Paillier, self).__init__()
        if q < p:  # ensure that p < q.
            self.p = q
            self.q = p
        else:
            self.p = p
            self.q = q
        self._derive_public_key()
        self._derive_private_key()

        self.nsquare = self.n * self.n
        # Wonder what this does
        self.max_int = self.n // 3 - 1

    def forward(self, input):
        self._encrypt(input)

    def backward(self, input):
        return self._decrypt(input)

    def _decrypt(self, ciphertext):
        decrypt_to_p = self._l_function(self._powmod_tensor(ciphertext, self.p - 1, self.psquare), self.p) * self.hp % self.p
        decrypt_to_q = self._l_function(self._powmod_tensor(ciphertext, self.q - 1, self.qsquare), self.q) * self.hq % self.q
        return self._crt(decrypt_to_p, decrypt_to_q)

    def _encrypt(self, plaintext, r_value=None):
        if self.n - self.max_int <= plaintext < self.n:
        # Very large plaintext, take a sneaky shortcut using inverses
            neg_plaintext = self.n - plaintext
            neg_ciphertext = (self.n * neg_plaintext + 1) % self.nsquare
            nude_ciphertext = self._invert(neg_ciphertext, self.nsquare)
        else:
        #     # we chose g = n + 1, so that we can exploit the fact that
        #     # (n+1)^plaintext = n*plaintext + 1 mod n^2
            nude_ciphertext = (self.n * plaintext + 1) % self.nsquare

        r = r_value or rand_like(plaintext) * self.n
        obfuscator = self._powmod_tensor(r, self.n, self.nsquare)

        return (nude_ciphertext * obfuscator) % self.nsquare

    def _derive_public_key(self):
        self.n = self.p * self.q
        self.g = self.n + 1

    def _derive_private_key(self):
        self.qsquare = self.q * self.q
        self.psquare = self.p * self.p
        self.p_inverse = self._invert(self.p, self.q)
        self.hp = self._h_function(self.p, self.psquare)
        self.hq = self._h_function(self.q, self.qsquare)

    @staticmethod
    def _l_function(x, p):
        return (x - 1) // p

    def _h_function(self, x, xsquare):
        return self._invert(self._l_function(self._powmod(self.g, x - 1, xsquare), x), x)

    def _invert(self, a, b):
        r, s, _ = self._extended_euclidean_algorithm(a, b)
        if r != 1:
            raise ZeroDivisionError('invert() no inverse exists')
        return s % b

    def _crt(self, mp, mq):
        u = (mq - mp) * self.p_inverse % self.q
        return mp + (u * self.p)

    @staticmethod
    def _extended_euclidean_algorithm(a, b):
        r0, r1 = a, b
        s0, s1 = 1, 0
        t0, t1 = 0, 1
        while r1 != 0:
            q = r0 // r1
            r0, r1 = r1, r0 - q * r1
            s0, s1 = s1, s0 - q * s1
            t0, t1 = t1, t0 - q * t1
        return r0, s0, t0

    @staticmethod
    def _powmod(a, b, c):
        if a == 1:
            return 1
        return pow(a, b, c)


    @staticmethod
    def _powmod_tensor(a, b, c):
        return fmod(tpow(a, b), c)
