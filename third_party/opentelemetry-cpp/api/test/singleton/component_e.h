// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#if defined(_MSC_VER)
// component_e is a DLL

#  ifdef BUILD_COMPONENT_E
__declspec(dllexport)
#  else
__declspec(dllimport)
#  endif

#endif

    void do_something_in_e();
