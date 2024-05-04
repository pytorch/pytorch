#pragma once

#include <cmath>
// Copyright 2005, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// The Google C++ Testing and Mocking Framework (Google Test)
//
// This header file declares functions and macros used internally by
// Google Test.  They are subject to change without notice.

using Bits = uint32_t;

// this avoids the "dereferencing type-punned pointer
// will break strict-aliasing rules" error
union Float {
  float float_;
  Bits bits_;
};

// # of bits in a number.
static const size_t kBitCount = 8 * sizeof(Bits);
// The mask for the sign bit.
static const Bits kSignBitMask = static_cast<Bits>(1) << (kBitCount - 1);

// GOOGLETEST_CM0001 DO NOT DELETE

// Converts an integer from the sign-and-magnitude representation to
// the biased representation.  More precisely, let N be 2 to the
// power of (kBitCount - 1), an integer x is represented by the
// unsigned number x + N.
//
// For instance,
//
//   -N + 1 (the most negative number representable using
//          sign-and-magnitude) is represented by 1;
//   0      is represented by N; and
//   N - 1  (the biggest number representable using
//          sign-and-magnitude) is represented by 2N - 1.
//
// Read http://en.wikipedia.org/wiki/Signed_number_representations
// for more details on signed number representations.
static Bits SignAndMagnitudeToBiased(const Bits& sam) {
  if (kSignBitMask & sam) {
    // sam represents a negative number.
    return ~sam + 1;
  } else {
    // sam represents a positive number.
    return kSignBitMask | sam;
  }
}

// Given two numbers in the sign-and-magnitude representation,
// returns the distance between them as an unsigned number.
static Bits DistanceBetweenSignAndMagnitudeNumbers(
    const Bits& sam1,
    const Bits& sam2) {
  const Bits biased1 = SignAndMagnitudeToBiased(sam1);
  const Bits biased2 = SignAndMagnitudeToBiased(sam2);
  return (biased1 >= biased2) ? (biased1 - biased2) : (biased2 - biased1);
}

// How many ULP's (Units in the Last Place) we want to tolerate when
// comparing two numbers.  The larger the value, the more error we
// allow.  A 0 value means that two numbers must be exactly the same
// to be considered equal.
//
// The maximum error of a single floating-point operation is 0.5
// units in the last place.  On Intel CPU's, all floating-point
// calculations are done with 80-bit precision, while double has 64
// bits.  Therefore, 4 should be enough for ordinary use.
//
// See the following article for more details on ULP:
// http://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
static const size_t kMaxUlps = 4;

// Returns true if and only if this number is at most kMaxUlps ULP's away
// from rhs.  In particular, this function:
//
//   - returns false if either number is (or both are) NAN.
//   - treats really large numbers as almost equal to infinity.
//   - thinks +0.0 and -0.0 are 0 DLP's apart.
inline bool AlmostEquals(float lhs, float rhs) {
  // The IEEE standard says that any comparison operation involving
  // a NAN must return false.
  if (std::isnan(lhs) || std::isnan(rhs))
    return false;

  Float l = {lhs};
  Float r = {rhs};

  return DistanceBetweenSignAndMagnitudeNumbers(l.bits_, r.bits_) <= kMaxUlps;
}
