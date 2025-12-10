"""Private counterpart of ``numpy.typing``."""

from ._array_like import ArrayLike as ArrayLike
from ._array_like import NDArray as NDArray
from ._array_like import _ArrayLike as _ArrayLike
from ._array_like import _ArrayLikeAnyString_co as _ArrayLikeAnyString_co
from ._array_like import _ArrayLikeBool_co as _ArrayLikeBool_co
from ._array_like import _ArrayLikeBytes_co as _ArrayLikeBytes_co
from ._array_like import _ArrayLikeComplex128_co as _ArrayLikeComplex128_co
from ._array_like import _ArrayLikeComplex_co as _ArrayLikeComplex_co
from ._array_like import _ArrayLikeDT64_co as _ArrayLikeDT64_co
from ._array_like import _ArrayLikeFloat64_co as _ArrayLikeFloat64_co
from ._array_like import _ArrayLikeFloat_co as _ArrayLikeFloat_co
from ._array_like import _ArrayLikeInt as _ArrayLikeInt
from ._array_like import _ArrayLikeInt_co as _ArrayLikeInt_co
from ._array_like import _ArrayLikeNumber_co as _ArrayLikeNumber_co
from ._array_like import _ArrayLikeObject_co as _ArrayLikeObject_co
from ._array_like import _ArrayLikeStr_co as _ArrayLikeStr_co
from ._array_like import _ArrayLikeString_co as _ArrayLikeString_co
from ._array_like import _ArrayLikeTD64_co as _ArrayLikeTD64_co
from ._array_like import _ArrayLikeUInt_co as _ArrayLikeUInt_co
from ._array_like import _ArrayLikeVoid_co as _ArrayLikeVoid_co
from ._array_like import _FiniteNestedSequence as _FiniteNestedSequence
from ._array_like import _SupportsArray as _SupportsArray
from ._array_like import _SupportsArrayFunc as _SupportsArrayFunc

#
from ._char_codes import _BoolCodes as _BoolCodes
from ._char_codes import _ByteCodes as _ByteCodes
from ._char_codes import _BytesCodes as _BytesCodes
from ._char_codes import _CDoubleCodes as _CDoubleCodes
from ._char_codes import _CharacterCodes as _CharacterCodes
from ._char_codes import _CLongDoubleCodes as _CLongDoubleCodes
from ._char_codes import _Complex64Codes as _Complex64Codes
from ._char_codes import _Complex128Codes as _Complex128Codes
from ._char_codes import _ComplexFloatingCodes as _ComplexFloatingCodes
from ._char_codes import _CSingleCodes as _CSingleCodes
from ._char_codes import _DoubleCodes as _DoubleCodes
from ._char_codes import _DT64Codes as _DT64Codes
from ._char_codes import _FlexibleCodes as _FlexibleCodes
from ._char_codes import _Float16Codes as _Float16Codes
from ._char_codes import _Float32Codes as _Float32Codes
from ._char_codes import _Float64Codes as _Float64Codes
from ._char_codes import _FloatingCodes as _FloatingCodes
from ._char_codes import _GenericCodes as _GenericCodes
from ._char_codes import _HalfCodes as _HalfCodes
from ._char_codes import _InexactCodes as _InexactCodes
from ._char_codes import _Int8Codes as _Int8Codes
from ._char_codes import _Int16Codes as _Int16Codes
from ._char_codes import _Int32Codes as _Int32Codes
from ._char_codes import _Int64Codes as _Int64Codes
from ._char_codes import _IntCCodes as _IntCCodes
from ._char_codes import _IntCodes as _IntCodes
from ._char_codes import _IntegerCodes as _IntegerCodes
from ._char_codes import _IntPCodes as _IntPCodes
from ._char_codes import _LongCodes as _LongCodes
from ._char_codes import _LongDoubleCodes as _LongDoubleCodes
from ._char_codes import _LongLongCodes as _LongLongCodes
from ._char_codes import _NumberCodes as _NumberCodes
from ._char_codes import _ObjectCodes as _ObjectCodes
from ._char_codes import _ShortCodes as _ShortCodes
from ._char_codes import _SignedIntegerCodes as _SignedIntegerCodes
from ._char_codes import _SingleCodes as _SingleCodes
from ._char_codes import _StrCodes as _StrCodes
from ._char_codes import _StringCodes as _StringCodes
from ._char_codes import _TD64Codes as _TD64Codes
from ._char_codes import _UByteCodes as _UByteCodes
from ._char_codes import _UInt8Codes as _UInt8Codes
from ._char_codes import _UInt16Codes as _UInt16Codes
from ._char_codes import _UInt32Codes as _UInt32Codes
from ._char_codes import _UInt64Codes as _UInt64Codes
from ._char_codes import _UIntCCodes as _UIntCCodes
from ._char_codes import _UIntCodes as _UIntCodes
from ._char_codes import _UIntPCodes as _UIntPCodes
from ._char_codes import _ULongCodes as _ULongCodes
from ._char_codes import _ULongLongCodes as _ULongLongCodes
from ._char_codes import _UnsignedIntegerCodes as _UnsignedIntegerCodes
from ._char_codes import _UShortCodes as _UShortCodes
from ._char_codes import _VoidCodes as _VoidCodes

#
from ._dtype_like import DTypeLike as DTypeLike
from ._dtype_like import _DTypeLike as _DTypeLike
from ._dtype_like import _DTypeLikeBool as _DTypeLikeBool
from ._dtype_like import _DTypeLikeBytes as _DTypeLikeBytes
from ._dtype_like import _DTypeLikeComplex as _DTypeLikeComplex
from ._dtype_like import _DTypeLikeComplex_co as _DTypeLikeComplex_co
from ._dtype_like import _DTypeLikeDT64 as _DTypeLikeDT64
from ._dtype_like import _DTypeLikeFloat as _DTypeLikeFloat
from ._dtype_like import _DTypeLikeInt as _DTypeLikeInt
from ._dtype_like import _DTypeLikeObject as _DTypeLikeObject
from ._dtype_like import _DTypeLikeStr as _DTypeLikeStr
from ._dtype_like import _DTypeLikeTD64 as _DTypeLikeTD64
from ._dtype_like import _DTypeLikeUInt as _DTypeLikeUInt
from ._dtype_like import _DTypeLikeVoid as _DTypeLikeVoid
from ._dtype_like import _SupportsDType as _SupportsDType
from ._dtype_like import _VoidDTypeLike as _VoidDTypeLike

#
from ._nbit import _NBitByte as _NBitByte
from ._nbit import _NBitDouble as _NBitDouble
from ._nbit import _NBitHalf as _NBitHalf
from ._nbit import _NBitInt as _NBitInt
from ._nbit import _NBitIntC as _NBitIntC
from ._nbit import _NBitIntP as _NBitIntP
from ._nbit import _NBitLong as _NBitLong
from ._nbit import _NBitLongDouble as _NBitLongDouble
from ._nbit import _NBitLongLong as _NBitLongLong
from ._nbit import _NBitShort as _NBitShort
from ._nbit import _NBitSingle as _NBitSingle

#
from ._nbit_base import (
    NBitBase as NBitBase,  # type: ignore[deprecated]  # pyright: ignore[reportDeprecated]
)
from ._nbit_base import _8Bit as _8Bit
from ._nbit_base import _16Bit as _16Bit
from ._nbit_base import _32Bit as _32Bit
from ._nbit_base import _64Bit as _64Bit
from ._nbit_base import _96Bit as _96Bit
from ._nbit_base import _128Bit as _128Bit

#
from ._nested_sequence import _NestedSequence as _NestedSequence

#
from ._scalars import _BoolLike_co as _BoolLike_co
from ._scalars import _CharLike_co as _CharLike_co
from ._scalars import _ComplexLike_co as _ComplexLike_co
from ._scalars import _FloatLike_co as _FloatLike_co
from ._scalars import _IntLike_co as _IntLike_co
from ._scalars import _NumberLike_co as _NumberLike_co
from ._scalars import _ScalarLike_co as _ScalarLike_co
from ._scalars import _TD64Like_co as _TD64Like_co
from ._scalars import _UIntLike_co as _UIntLike_co
from ._scalars import _VoidLike_co as _VoidLike_co

#
from ._shape import _AnyShape as _AnyShape
from ._shape import _Shape as _Shape
from ._shape import _ShapeLike as _ShapeLike

#
from ._ufunc import _GUFunc_Nin2_Nout1 as _GUFunc_Nin2_Nout1
from ._ufunc import _UFunc_Nin1_Nout1 as _UFunc_Nin1_Nout1
from ._ufunc import _UFunc_Nin1_Nout2 as _UFunc_Nin1_Nout2
from ._ufunc import _UFunc_Nin2_Nout1 as _UFunc_Nin2_Nout1
from ._ufunc import _UFunc_Nin2_Nout2 as _UFunc_Nin2_Nout2
