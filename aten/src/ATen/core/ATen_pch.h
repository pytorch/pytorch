// This global header must not depend on native_functions.yaml or
// incremental builds will be next to useless
#pragma push_macro("TORCH_ASSERT_NO_OPERATORS")
#define TORCH_ASSERT_NO_OPERATORS

// This macro doesn't work if defined after the first time inttypes.h
// is included, so won't work anywhere if not defined here.
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif
#include <cinttypes>

// This list of headers was generated using a script that finds
// high-impact headers and then manually tweaked to remove OS specific
// or duplicate headers (e.g. <cassert> and <assert.h>) and to remove
// "impl" headers (e.g BFloat16-inl.h or complex_math.h in c10).

// To generate the initial list:
// 1. Build pytorch from scratch with all build caching disabled
// 2. Generate a build trace with ninjatracing (https://github.com/nico/ninjatracing)
//    $ ninjatracing /path/to/pytorch/build/.ninja_log > trace_all.json
// 3. Run pch_gen.py from https://github.com/peterbell10/build_analysis/
//    $ python pch_gen.py --threshold .80 --target torch_cpu --build_dir /path/to/pytorch/build --trace trace_all.json
//    Where the threshold can be tweaked until c10 and some of ATen
//    core are included but TORCH_ASSERT_NO_OPERATORS still passes.

#include <cassert>
#include <cctype>
#include <cerrno>
#include <climits>
#include <clocale>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cwchar>
#include <cwctype>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <complex>
#include <deque>
#include <exception>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <iomanip>
#include <ios>
#include <iosfwd>
#include <istream>
#include <iterator>
#include <limits>
#include <locale>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <numeric>
#include <ostream>
#include <ratio>
#include <set>
#include <sstream>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <system_error>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <c10/core/Allocator.h>
#include <c10/core/AutogradState.h>
#include <c10/core/Backend.h>
#include <c10/core/CopyBytes.h>
#include <c10/core/DefaultDtype.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/DeviceType.h>
#include <c10/core/DispatchKey.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/core/GeneratorImpl.h>
#include <c10/core/GradMode.h>
#include <c10/core/InferenceMode.h>
#include <c10/core/Layout.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/QScheme.h>
#include <c10/core/ScalarType.h>
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <c10/core/Storage.h>
#include <c10/core/StorageImpl.h>
#include <c10/core/Stream.h>
#include <c10/core/SymInt.h>
#include <c10/core/SymIntArrayRef.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/TensorOptions.h>
#include <c10/core/UndefinedTensorImpl.h>
#include <c10/core/WrapDimMinimal.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/core/impl/InlineDeviceGuard.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/core/impl/PyInterpreter.h>
#include <c10/core/impl/SizesAndStrides.h>
#include <c10/core/impl/VirtualGuardImpl.h>

#include <c10/macros/Export.h>
#include <c10/macros/Macros.h>

#include <c10/util/AlignOf.h>
#include <c10/util/Array.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Backtrace.h>
#include <c10/util/C++17.h>
#include <c10/util/ConstexprCrc.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Exception.h>
#include <c10/util/ExclusivelyOwned.h>
#include <c10/util/Flags.h>
#include <c10/util/Half.h>
#include <c10/util/IdWrapper.h>
#include <c10/util/Logging.h>
#include <c10/util/MaybeOwned.h>
#include <c10/util/Metaprogramming.h>
#include <c10/util/Optional.h>
#include <c10/util/Registry.h>
#include <c10/util/SmallVector.h>
#include <c10/util/StringUtil.h>
#include <c10/util/ThreadLocalDebugInfo.h>
#include <c10/util/Type.h>
#include <c10/util/TypeCast.h>
#include <c10/util/TypeIndex.h>
#include <c10/util/TypeList.h>
#include <c10/util/TypeSafeSignMath.h>
#include <c10/util/TypeTraits.h>
#include <c10/util/UniqueVoidPtr.h>
#include <c10/util/accumulate.h>
#include <c10/util/complex.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/in_place.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/irange.h>
#include <c10/util/llvmMathExtras.h>
#include <c10/util/python_stub.h>
#include <c10/util/qint32.h>
#include <c10/util/qint8.h>
#include <c10/util/quint2x4.h>
#include <c10/util/quint4x2.h>
#include <c10/util/quint8.h>
#include <c10/util/reverse_iterator.h>
#include <c10/util/safe_numerics.h>
#include <c10/util/string_utils.h>
#include <c10/util/string_view.h>
#include <c10/util/typeid.h>

#include <ATen/core/DeprecatedTypeProperties.h>
#include <ATen/core/DeprecatedTypePropertiesRegistry.h>
#include <ATen/core/DimVector.h>
#include <ATen/core/Dimname.h>
#include <ATen/core/Generator.h>
#include <ATen/core/NamedTensor.h>
#include <ATen/core/QuantizerBase.h>
#include <ATen/core/TensorAccessor.h>
#include <ATen/core/TensorBase.h>
#include <ATen/core/symbol.h>

#pragma pop_macro("TORCH_ASSERT_NO_OPERATORS")
