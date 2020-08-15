// Taken from
// https://github.com/JuliaLang/julia/blob/v1.1.0/src/support/strtod.c

#include <ATen/core/Macros.h>
#include <locale.h>
#include <stdlib.h>

#if defined(__APPLE__) || defined(__FreeBSD__)
#include <xlocale.h>
#endif

// The following code is derived from the Python function _PyOS_ascii_strtod
// see http://hg.python.org/cpython/file/default/Python/pystrtod.c
//
// Copyright © 2001-2014 Python Software Foundation; All Rights Reserved
//
// The following modifications have been made:
// - Leading spaces are ignored
// - Parsing of hex floats is supported in the derived version
// - Python functions for tolower, isdigit and malloc have been replaced by the
// respective
//   C stdlib functions

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <string.h>
#include <locale>

#define D_PNAN ((double)+NAN)
#define D_PINF ((double)+INFINITY)

namespace {
#ifdef __ANDROID__
int case_insensitive_match(const char* s, const char* t) {
  while (*t && tolower(*s) == *t) {
    s++;
    t++;
  }
  return *t ? 0 : 1;
}

double parse_inf_or_nan(const char* p, char** endptr) {
  double retval;
  const char* s;
  int negate = 0;

  s = p;
  if (*s == '-') {
    negate = 1;
    s++;
  } else if (*s == '+') {
    s++;
  }
  if (case_insensitive_match(s, "inf")) {
    s += 3;
    if (case_insensitive_match(s, "inity"))
      s += 5;
    retval = negate ? -D_PINF : D_PINF;
  } else if (case_insensitive_match(s, "nan")) {
    s += 3;
    retval = negate ? -D_PNAN : D_PNAN;
  } else {
    s = p;
    retval = -1.0;
  }
  *endptr = (char*)s;
  return retval;
}
#endif

} // namespace

namespace torch {
namespace jit {

#ifdef _MSC_VER
C10_EXPORT double strtod_c(const char* nptr, char** endptr) {
  static _locale_t loc = _create_locale(LC_ALL, "C");
  return _strtod_l(nptr, endptr, loc);
}
#elif defined(__ANDROID__)
C10_EXPORT double strtod_c(const char* nptr, char** endptr) {
  char* fail_pos;
  double val;
  const char *p, *decimal_point_pos;
  const char* end = NULL; /* Silence gcc */
  const char* digits_pos = NULL;
  int negate = 0;

  fail_pos = NULL;

  char decimal_point =
      std::use_facet<std::numpunct<char>>(std::locale()).decimal_point();

  decimal_point_pos = NULL;

  /* Parse infinities and nans */
  val = parse_inf_or_nan(nptr, endptr);
  if (*endptr != nptr)
    return val;

  /* Set errno to zero, so that we can distinguish zero results
     and underflows */
  errno = 0;

  /* We process the optional sign manually, then pass the remainder to
     the system strtod.  This ensures that the result of an underflow
     has the correct sign.  */
  p = nptr;

  /* parse leading spaces */
  while (isspace((unsigned char)*p)) {
    p++;
  }

  /* Process leading sign, if present */
  if (*p == '-') {
    negate = 1;
    p++;
  } else if (*p == '+') {
    p++;
  }

  /* This code path is used for hex floats */
  if (*p == '0' && (*(p + 1) == 'x' || *(p + 1) == 'X')) {
    digits_pos = p;
    p += 2;
    /* Check that what's left begins with a digit or decimal point */
    if (!isxdigit(*p) && *p != '.')
      goto invalid_string;

    if (decimal_point != '.') {
      /* Look for a '.' in the input; if present, it'll need to be
         swapped for the current locale's decimal point before we
         call strtod.  On the other hand, if we find the current
         locale's decimal point then the input is invalid. */
      while (isxdigit(*p))
        p++;

      if (*p == '.') {
        decimal_point_pos = p++;

        /* locate end of number */
        while (isxdigit(*p))
          p++;

        if (*p == 'p' || *p == 'P')
          p++;
        if (*p == '+' || *p == '-')
          p++;
        while (isdigit(*p))
          p++;
        end = p;
      } else if (*p == decimal_point)
        goto invalid_string;
      /* For the other cases, we need not convert the decimal point */
    }
  } else {
    /* Check that what's left begins with a digit or decimal point */
    if (!isdigit(*p) && *p != '.')
      goto invalid_string;

    digits_pos = p;
    if (decimal_point != '.') {
      /* Look for a '.' in the input; if present, it'll need to be
         swapped for the current locale's decimal point before we
         call strtod.  On the other hand, if we find the current
         locale's decimal point then the input is invalid. */
      while (isdigit(*p))
        p++;

      if (*p == '.') {
        decimal_point_pos = p++;

        /* locate end of number */
        while (isdigit(*p))
          p++;

        if (*p == 'e' || *p == 'E')
          p++;
        if (*p == '+' || *p == '-')
          p++;
        while (isdigit(*p))
          p++;
        end = p;
      } else if (*p == decimal_point)
        goto invalid_string;
      /* For the other cases, we need not convert the decimal point */
    }
  }

  if (decimal_point_pos) {
    char *copy, *c;
    /* Create a copy of the input, with the '.' converted to the
       locale-specific decimal point */
    copy = (char*)malloc(end - digits_pos + 2);
    if (copy == NULL) {
      *endptr = (char*)nptr;
      errno = ENOMEM;
      return val;
    }

    c = copy;
    memcpy(c, digits_pos, decimal_point_pos - digits_pos);
    c += decimal_point_pos - digits_pos;
    memcpy(c, &decimal_point, 1);
    c += 1;
    memcpy(c, decimal_point_pos + 1, end - (decimal_point_pos + 1));
    c += end - (decimal_point_pos + 1);
    *c = 0;

    val = strtod(copy, &fail_pos);

    if (fail_pos) {
      fail_pos = (char*)digits_pos + (fail_pos - copy);
    }

    free(copy);
  } else {
    val = strtod(digits_pos, &fail_pos);
  }

  if (fail_pos == digits_pos)
    goto invalid_string;

  if (negate && fail_pos != nptr)
    val = -val;
  *endptr = fail_pos;

  return val;

invalid_string:
  *endptr = (char*)nptr;
  errno = EINVAL;
  return -1.0;
}
#else
C10_EXPORT double strtod_c(const char* nptr, char** endptr) {
  /// NOLINTNEXTLINE(hicpp-signed-bitwise)
  static locale_t loc = newlocale(LC_ALL_MASK, "C", nullptr);
  return strtod_l(nptr, endptr, loc);
}
#endif

C10_EXPORT float strtof_c(const char* nptr, char** endptr) {
  return (float)strtod_c(nptr, endptr);
}

} // namespace jit
} // namespace torch
