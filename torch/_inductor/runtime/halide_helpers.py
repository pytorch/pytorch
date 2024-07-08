# mypy: allow-untyped-defs
import halide as hl  # type: ignore[import-untyped, import-not-found]

PHILOX_N_ROUNDS_DEFAULT = 10  # Default number of rounds for philox
PHILOX_KEY_A_U32 = hl.u32(0x9E3779B9)
PHILOX_KEY_B_U32 = hl.u32(0xBB67AE85)
PHILOX_ROUND_A_U32 = hl.u32(0xD2511F53)
PHILOX_ROUND_B_U32 = hl.u32(0xCD9E8D57)
PHILOX_KEY_A_U64 = hl.u64(0x9E3779B97F4A7C15)
PHILOX_KEY_B_U64 = hl.u64(0xBB67AE8584CAA73B)
PHILOX_ROUND_A_U64 = hl.u64(0xD2E7470EE14C6C93)
PHILOX_ROUND_B_U64 = hl.u64(0xCA5A826395121157)


def _pair_uniform_to_normal(u1, u2):
    """Box-Muller transform"""
    u1 = hl.max(1.0e-7, u1)
    th = 6.283185307179586 * u2
    r = hl.sqrt(-2.0 * hl.log(u1))
    return r * hl.cos(th), r * hl.sin(th)


def _uint_to_uniform_float(x):
    """
    Numerically stable function to convert a random uint into a random float uniformly sampled in [0, 1).
    """

    # TODO:
    # conditions can be simplified
    # scale is ((2**23 - 1) / 2**23) * 2**(N_BITS - 1)
    if x.type() == hl.UInt(32) or x.type() == hl.Int(32):
        # maximum value such that `MAX_INT * scale < 1.0` (with float rounding)
        x = hl.cast(hl.Int(32), x)
        scale = hl.f64(4.6566127342e-10)
    elif x.type() in (hl.UInt(64), hl.Int(64)):
        x = hl.cast(hl.Int(64), x)
        scale = hl.f64(1.0842020432385337e-19)
    else:
        raise TypeError(f"Unsupported type {x.type()} in philox")
    x = hl.select(x < 0, -x - 1, x)
    return x * scale


def philox_impl(c0, c1, c2, c3, k0, k1, n_rounds):
    def umulhi(a, b):
        a = hl.cast(hl.UInt(64), a)
        b = hl.cast(hl.UInt(64), b)
        return hl.cast(hl.UInt(32), ((a * b) >> 32) & hl.u64(0xFFFFFFFF))

    if c0.type() == hl.UInt(32):
        PHILOX_KEY_A, PHILOX_KEY_B, PHILOX_ROUND_A, PHILOX_ROUND_B = (
            PHILOX_KEY_A_U32,
            PHILOX_KEY_B_U32,
            PHILOX_ROUND_A_U32,
            PHILOX_ROUND_B_U32,
        )
    else:
        PHILOX_KEY_A, PHILOX_KEY_B, PHILOX_ROUND_A, PHILOX_ROUND_B = (
            PHILOX_KEY_A_U64,
            PHILOX_KEY_B_U64,
            PHILOX_ROUND_A_U64,
            PHILOX_ROUND_B_U64,
        )

    for _ in range(n_rounds):
        _c0, _c2 = c0, c2

        c0 = umulhi(PHILOX_ROUND_B, _c2) ^ c1 ^ k0
        c2 = umulhi(PHILOX_ROUND_A, _c0) ^ c3 ^ k1
        c1 = PHILOX_ROUND_B * _c2
        c3 = PHILOX_ROUND_A * _c0
        # raise key
        k0 = k0 + PHILOX_KEY_A
        k1 = k1 + PHILOX_KEY_B

    return c0, c1, c2, c3


def halide_philox(seed, c0, c1, c2, c3, n_rounds):
    seed = hl.cast(hl.UInt(64), seed)

    if c0.type().bits() == 32:
        seed_hi = hl.cast(hl.UInt(32), (seed >> 32) & hl.u64(0xFFFFFFFF))
        seed_lo = hl.cast(hl.UInt(32), seed & hl.u64(0xFFFFFFFF))
    elif c0.type().bits() == 64:
        seed_hi = hl.u64(1)
        seed_lo = seed
    else:
        raise TypeError(f"Unsupported type {c0.type()} in philox")

    return philox_impl(c0, c1, c2, c3, seed_lo, seed_hi, n_rounds)


def randint4x(seed: hl.Expr, offsets: hl.Expr, n_rounds):
    offsets = hl.cast(hl.UInt(32), offsets)
    _0 = offsets * hl.u32(0)
    return halide_philox(seed, offsets, _0, _0, _0, n_rounds)


def rand4x(seed, offsets, n_rounds=PHILOX_N_ROUNDS_DEFAULT):
    i1, i2, i3, i4 = randint4x(seed, offsets, n_rounds)
    u1 = _uint_to_uniform_float(i1)
    u2 = _uint_to_uniform_float(i2)
    u3 = _uint_to_uniform_float(i3)
    u4 = _uint_to_uniform_float(i4)
    return u1, u2, u3, u4


def randint(seed, offset, n_rounds=PHILOX_N_ROUNDS_DEFAULT):
    ret, _, _, _ = randint4x(seed, offset, n_rounds)
    return ret


def rand(seed, offset, n_rounds=PHILOX_N_ROUNDS_DEFAULT):
    source = randint(seed, offset, n_rounds)
    return _uint_to_uniform_float(source)


def randn(seed, offset):
    i1, i2, _, _ = randint4x(seed, offset, PHILOX_N_ROUNDS_DEFAULT)
    u1 = _uint_to_uniform_float(i1)
    u2 = _uint_to_uniform_float(i2)
    n1, _ = _pair_uniform_to_normal(u1, u2)
    return n1


def randint64(seed: hl.Expr, offsets: hl.Expr, low, high):
    r0, r1, r2, r3 = randint4x(seed, offsets, PHILOX_N_ROUNDS_DEFAULT)
    r0 = hl.cast(hl.UInt(64), r0)
    r1 = hl.cast(hl.UInt(64), r1)

    result = r0 | (r1 << 32)
    size = high - low
    result = result % hl.cast(hl.UInt(64), size)
    result = hl.cast(hl.UInt(64), result) + low
    return result


def main(seed: int, offsets: list[int]):
    import numpy as np

    seed_var = hl.Var("seed")
    offset_var = hl.Var("offset")

    seed_input = hl.Buffer(np.array([seed], dtype=np.uint32))
    offset_input = hl.Buffer(np.array(offsets, dtype=np.uint32))

    seed_input = seed_input[seed_var]
    offset_input = offset_input[offset_var]

    randints = hl.Func("randints")

    y = randint4x(seed_input, offset_input, PHILOX_N_ROUNDS_DEFAULT)
    # y = rand4x(seed_input, offset_input, PHILOX_N_ROUNDS_DEFAULT) ok
    # y = randint64(seed_input, offset_input, 0, 1000) ok
    # y = randint(seed_input, offset_input, n_rounds)
    # y = rand(seed_input, offset_input, n_rounds)
    # y = randn(seed_input, offset_input)

    randints[seed_var, offset_var] = y

    out = randints.realize([1, len(offsets)])
    return np.asanyarray(out)


if __name__ == "__main__":
    out = main(5123, [0, 1, 2, 100])
    print(out)
