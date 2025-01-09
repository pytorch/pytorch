#pragma once
#include <c10/core/QScheme.h>

// Forward declarations of core ATen types used in dispatch functions
namespace c10 {

template <typename T>
class List;
template <typename T>
class IListRef;
class Stream;
class Scalar;
class SymInt;
class SymIntList;
struct Storage;
struct TensorOptions;
template <typename T>
class ArrayRef;
template <typename T>
class OptionalArrayRef;

} // namespace c10

namespace at {

class Tensor;
class OptionalTensorRef;
struct Dimname;
struct Generator;
using TensorList = c10::ArrayRef<Tensor>;
using ITensorListRef = c10::IListRef<Tensor>;
using IOptTensorListRef = c10::IListRef<OptionalTensorRef>;
using DimnameList = c10::ArrayRef<Dimname>;
using IntArrayRef = c10::ArrayRef<int64_t>;
using OptionalIntArrayRef = c10::OptionalArrayRef<int64_t>;
using OptionalSymIntArrayRef = c10::OptionalArrayRef<c10::SymInt>;

using c10::QScheme;
using c10::Scalar;
using c10::Storage;
using c10::Stream;
using c10::SymInt;
using c10::SymIntList;
using c10::TensorOptions;

} // namespace at
