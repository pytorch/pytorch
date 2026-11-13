#pragma once

#include <torch/csrc/Export.h>
#include <cstddef>
#include <cstdint>

namespace torch::jit {

// Copyright (c) 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
// 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020 Python Software
// Foundation; All Rights Reserved
//
// Stolen (with appropriate modifications) by @agolynski
// (https://github.com/pytorch/pytorch/pull/33019) from cpython repo
// Objects/sliceobject.c with comment: this is harder to get right than you
// might think
//
// This adjusts indexes according to python list semantics and returns number
// of elements in the resulting list.
TORCH_API int64_t slice_indices_adjust(
    int64_t length,
    int64_t* start,
    int64_t* stop,
    int64_t step);

} // namespace torch::jit
