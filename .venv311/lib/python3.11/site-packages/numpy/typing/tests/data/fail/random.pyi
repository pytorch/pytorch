import numpy as np
import numpy.typing as npt

SEED_FLOAT: float = 457.3
SEED_ARR_FLOAT: npt.NDArray[np.float64] = np.array([1.0, 2, 3, 4])
SEED_ARRLIKE_FLOAT: list[float] = [1.0, 2.0, 3.0, 4.0]
SEED_SEED_SEQ: np.random.SeedSequence = np.random.SeedSequence(0)
SEED_STR: str = "String seeding not allowed"

# default rng
np.random.default_rng(SEED_FLOAT)  # type: ignore[arg-type]
np.random.default_rng(SEED_ARR_FLOAT)  # type: ignore[arg-type]
np.random.default_rng(SEED_ARRLIKE_FLOAT)  # type: ignore[arg-type]
np.random.default_rng(SEED_STR)  # type: ignore[arg-type]

# Seed Sequence
np.random.SeedSequence(SEED_FLOAT)  # type: ignore[arg-type]
np.random.SeedSequence(SEED_ARR_FLOAT)  # type: ignore[arg-type]
np.random.SeedSequence(SEED_ARRLIKE_FLOAT)  # type: ignore[arg-type]
np.random.SeedSequence(SEED_SEED_SEQ)  # type: ignore[arg-type]
np.random.SeedSequence(SEED_STR)  # type: ignore[arg-type]

seed_seq: np.random.bit_generator.SeedSequence = np.random.SeedSequence()
seed_seq.spawn(11.5)  # type: ignore[arg-type]
seed_seq.generate_state(3.14)  # type: ignore[arg-type]
seed_seq.generate_state(3, np.uint8)  # type: ignore[arg-type]
seed_seq.generate_state(3, "uint8")  # type: ignore[arg-type]
seed_seq.generate_state(3, "u1")  # type: ignore[arg-type]
seed_seq.generate_state(3, np.uint16)  # type: ignore[arg-type]
seed_seq.generate_state(3, "uint16")  # type: ignore[arg-type]
seed_seq.generate_state(3, "u2")  # type: ignore[arg-type]
seed_seq.generate_state(3, np.int32)  # type: ignore[arg-type]
seed_seq.generate_state(3, "int32")  # type: ignore[arg-type]
seed_seq.generate_state(3, "i4")  # type: ignore[arg-type]

# Bit Generators
np.random.MT19937(SEED_FLOAT)  # type: ignore[arg-type]
np.random.MT19937(SEED_ARR_FLOAT)  # type: ignore[arg-type]
np.random.MT19937(SEED_ARRLIKE_FLOAT)  # type: ignore[arg-type]
np.random.MT19937(SEED_STR)  # type: ignore[arg-type]

np.random.PCG64(SEED_FLOAT)  # type: ignore[arg-type]
np.random.PCG64(SEED_ARR_FLOAT)  # type: ignore[arg-type]
np.random.PCG64(SEED_ARRLIKE_FLOAT)  # type: ignore[arg-type]
np.random.PCG64(SEED_STR)  # type: ignore[arg-type]

np.random.Philox(SEED_FLOAT)  # type: ignore[arg-type]
np.random.Philox(SEED_ARR_FLOAT)  # type: ignore[arg-type]
np.random.Philox(SEED_ARRLIKE_FLOAT)  # type: ignore[arg-type]
np.random.Philox(SEED_STR)  # type: ignore[arg-type]

np.random.SFC64(SEED_FLOAT)  # type: ignore[arg-type]
np.random.SFC64(SEED_ARR_FLOAT)  # type: ignore[arg-type]
np.random.SFC64(SEED_ARRLIKE_FLOAT)  # type: ignore[arg-type]
np.random.SFC64(SEED_STR)  # type: ignore[arg-type]

# Generator
np.random.Generator(None)  # type: ignore[arg-type]
np.random.Generator(12333283902830213)  # type: ignore[arg-type]
np.random.Generator("OxFEEDF00D")  # type: ignore[arg-type]
np.random.Generator([123, 234])  # type: ignore[arg-type]
np.random.Generator(np.array([123, 234], dtype="u4"))  # type: ignore[arg-type]
