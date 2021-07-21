import jax
import jax.numpy as jnp
import numpy as np

dtype=jnp.bfloat16
# 0+ -> 0-
# 0- -> 0+
# 0 -> 1
# 0 -> -1
# 1 -> 2
# 1 -> 0
# -1 -> -2
# -1 -> 0
# nan -> 0
# 0 -> nan
# nan -> nan
# nan -> inf
# inf -> nan
# inf -> -inf
# -inf -> inf
# inf -> 0
# 0 -> inf
# -inf -> 0
# 0 -> -inf
nan = float('nan')
inf = float('inf')
cases = ((0, -0),
         (-0, 0),
         (0, 1),
         (0, -1),
         (1, -2),
         (1, 0),
         (1, 2),
         (-1, -2),
         (-1, 0),
         (2, -1),
         (2, 1),
         (20, 3000),
         (20, -3000),
         (3000, -20),
        (-3000, 20),
         (nan, 0),
         (0, nan),
         (nan, nan),
         (nan, inf),
         (inf, nan),
         (inf, -inf),
         (-inf, inf),
         (inf, 0),
         (0, inf),
         (-inf, 0),
         (0, -inf),
         )

print("(")
for from_v, to_v in cases:
    from_a = jnp.array([from_v], dtype=dtype)
    to_a = jnp.array([to_v], dtype=dtype)
    result = (from_v, to_v, jnp.nextafter(from_a, to_a).item())
    print(result,",")
print(")")