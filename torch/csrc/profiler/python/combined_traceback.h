#pragma once

#include <torch/csrc/profiler/combined_traceback.h>

#include <pybind11/pybind11.h>
#include <torch/csrc/utils/pybind.h>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace nlohmann {
inline namespace json_abi_v3_12_0 {

template <typename T, typename SFINAE>
struct adl_serializer;

template <
    template <typename U, typename V, typename... Args> class ObjectType,
    template <typename U, typename... Args> class ArrayType,
    class StringType,
    class BooleanType,
    class NumberIntegerType,
    class NumberUnsignedType,
    class NumberFloatType,
    template <typename U> class AllocatorType,
    template <typename T, typename SFINAE> class JSONSerializer,
    class BinaryType,
    class CustomBaseClass>
class basic_json;

// Keep this in sync with nlohmann/json_fwd.hpp from the vendored nlohmann:
// https://github.com/nlohmann/json/blob/55f93686c01528224f448c19128836e7df245f72/include/nlohmann/json_fwd.hpp
// Do not include nlohmann headers here: this public header is installed without
// vendored nlohmann.
using json = basic_json<
    std::map,
    std::vector,
    std::string,
    bool,
    std::int64_t,
    std::uint64_t,
    double,
    std::allocator,
    adl_serializer,
    std::vector<std::uint8_t>,
    void>;

} // namespace json_abi_v3_12_0
} // namespace nlohmann

namespace torch {

// symbolize combined traceback objects, converting them into lists of
// dictionaries that are easily consumed in python.

// returns std::vector because one use is to call it with a batch of
// tracebacks that come from a larger datastructure (e.g. a memory snapshot)
// and then have more c++ code to put those objects in the right place.
TORCH_API std::vector<pybind11::object> py_symbolize(
    std::vector<CapturedTraceback*>& to_symbolize);

// Return the callback in json format so that it can be used within cpp
TORCH_API std::vector<nlohmann::json> json_symbolize(
    std::vector<CapturedTraceback*>& to_symbolize);

// requires GIL to be held, frees any pending free frames
TORCH_PYTHON_API void freeDeadCapturedTracebackFrames();

TORCH_PYTHON_API void installCapturedTracebackPython();

} // namespace torch
