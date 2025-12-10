from typing import Any, TypeAlias

import numpy as np

# NOTE: `_StrLike_co` and `_BytesLike_co` are pointless, as `np.str_` and
# `np.bytes_` are already subclasses of their builtin counterpart
_CharLike_co: TypeAlias = str | bytes

# The `<X>Like_co` type-aliases below represent all scalars that can be
# coerced into `<X>` (with the casting rule `same_kind`)
_BoolLike_co: TypeAlias = bool | np.bool
_UIntLike_co: TypeAlias = bool | np.unsignedinteger | np.bool
_IntLike_co: TypeAlias = int | np.integer | np.bool
_FloatLike_co: TypeAlias = float | np.floating | np.integer | np.bool
_ComplexLike_co: TypeAlias = complex | np.number | np.bool
_NumberLike_co: TypeAlias = _ComplexLike_co
_TD64Like_co: TypeAlias = int | np.timedelta64 | np.integer | np.bool
# `_VoidLike_co` is technically not a scalar, but it's close enough
_VoidLike_co: TypeAlias = tuple[Any, ...] | np.void
_ScalarLike_co: TypeAlias = complex | str | bytes | np.generic
