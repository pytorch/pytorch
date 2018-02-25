/*
 * Copyright 2014 Google Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MATRIX_FUNCS_H_
#define MATRIX_FUNCS_H_

#include <stdlib.h>
#include <math.h>
#include <algorithm>

#ifdef DOUBLE_PRECISION
#define MTYPE double
#else
#define MTYPE float
#endif

#define MYRAND ((double)rand() / ((double)RAND_MAX + 1))

inline MTYPE _zero(MTYPE x) {
    return 0;
}

inline MTYPE _one(MTYPE x) {
    return 1;
}

inline MTYPE _abs(MTYPE x) {
    return x > 0 ? x : -x;
}

inline MTYPE _square(MTYPE x) {
    return x * x;
}

inline MTYPE _sigma1(MTYPE x) {
    return (tanh(x / 2) + 1) / 2;
}

inline MTYPE _sigma2(MTYPE x) {
    return 1 / (1 + exp(-x));
}

inline MTYPE _recip(MTYPE x) {
    return 1 / x;
}

inline MTYPE _exp(MTYPE x) {
    return exp(x);
}

inline MTYPE _log(MTYPE x) {
    return log(x);
}

inline MTYPE _tanh(MTYPE x) {
    return tanh(x);
}

inline MTYPE _sign(MTYPE x) {
    return x > 0 ? 1 : -1;
}

inline MTYPE _rand(MTYPE x) {
    return MYRAND;
}

inline MTYPE _divide(MTYPE x, MTYPE y) {
    return x / y;
}

inline MTYPE _mult(MTYPE x, MTYPE y) {
    return x * y;
}

inline MTYPE _add(MTYPE x, MTYPE y) {
    return x + y;
}

inline MTYPE _addSquare(MTYPE x, MTYPE y) {
    return x*x + y;
}

inline MTYPE _addWithScale(MTYPE x, MTYPE y, MTYPE scale) {
    return x + scale*y;
}

inline MTYPE _addWithScale2(MTYPE x, MTYPE y, MTYPE scaleThis, MTYPE scaleM) {
    return scaleThis * x + scaleM * y;
}

inline MTYPE _max(MTYPE x, MTYPE y) {
    return std::max(x, y);
}

inline MTYPE _min(MTYPE x, MTYPE y) {
    return std::min(x, y);
}

inline MTYPE _bigger(MTYPE x, MTYPE y) {
    return x > y;
}

inline MTYPE _smaller(MTYPE x, MTYPE y) {
    return x < y;
}

inline MTYPE _equal(MTYPE x, MTYPE y) {
    return x == y;
}

inline MTYPE _notEqual(MTYPE x, MTYPE y) {
    return x != y;
}

#endif /* MATRIX_FUNCS_H_ */
