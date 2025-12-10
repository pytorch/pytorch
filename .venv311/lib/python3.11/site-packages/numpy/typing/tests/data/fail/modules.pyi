import numpy as np

np.testing.bob  # type: ignore[attr-defined]
np.bob  # type: ignore[attr-defined]

# Stdlib modules in the namespace by accident
np.warnings  # type: ignore[attr-defined]
np.sys  # type: ignore[attr-defined]
np.os  # type: ignore[attr-defined]
np.math  # type: ignore[attr-defined]

# Public sub-modules that are not imported to their parent module by default;
# e.g. one must first execute `import numpy.lib.recfunctions`
np.lib.recfunctions  # type: ignore[attr-defined]

np.__deprecated_attrs__  # type: ignore[attr-defined]
np.__expired_functions__  # type: ignore[attr-defined]
