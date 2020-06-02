#pragma once

/*
 * We split Tensor.h into TensorBody.h and TensorMethods.h because we want
 * all TensorMethods to be inlined, but they depend on the Dispatcher,
 * which in turn depends on many other things, which then depend back on Tensor.
 *
 * We can break this dependency chain by having the dispatcher only depend on
 * TensorBody.h and not TensorMethods.h.
 */
#include <ATen/core/TensorBody.h>
#include <ATen/core/TensorMethods.h>
