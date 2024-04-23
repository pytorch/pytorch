// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

/*
  TODO:
  Once singleton are supported for windows,
  expand this test to use ::LoadLibrary, ::GetProcAddress, ::FreeLibrary
*/
#ifndef _WIN32
#  include <dlfcn.h>
#endif

#include <iostream>

#include "component_a.h"
#include "component_b.h"
#include "component_c.h"
#include "component_d.h"
#include "component_e.h"
#include "component_f.h"

#include "opentelemetry/trace/default_span.h"
#include "opentelemetry/trace/provider.h"
#include "opentelemetry/trace/span.h"
#include "opentelemetry/trace/span_context_kv_iterable.h"
#include "opentelemetry/trace/span_startoptions.h"
#include "opentelemetry/trace/tracer_provider.h"

using namespace opentelemetry;

void do_something()
{
  do_something_in_a();
  do_something_in_b();
  do_something_in_c();
  do_something_in_d();
  do_something_in_e();
  do_something_in_f();

  /*
    See https://github.com/bazelbuild/bazel/issues/4218

    There is no way to set LD_LIBRARY_PATH in bazel,
    for dlopen() to find the library.

    Verified manually that dlopen("/full/path/to/libcomponent_g.so") works,
    and that the test passes in this case.
  */

#ifndef BAZEL_BUILD
  /* Call do_something_in_g() */

  void *component_g = dlopen("libcomponent_g.so", RTLD_NOW);
  EXPECT_NE(component_g, nullptr);

  auto *func_g = (void (*)())dlsym(component_g, "do_something_in_g");
  EXPECT_NE(func_g, nullptr);

  (*func_g)();

  dlclose(component_g);

  /* Call do_something_in_h() */

  void *component_h = dlopen("libcomponent_h.so", RTLD_NOW);
  EXPECT_NE(component_h, nullptr);

  auto *func_h = (void (*)())dlsym(component_h, "do_something_in_h");
  EXPECT_NE(func_h, nullptr);

  (*func_h)();

  dlclose(component_h);
#endif
}

int span_a_lib_count   = 0;
int span_a_f1_count    = 0;
int span_a_f2_count    = 0;
int span_b_lib_count   = 0;
int span_b_f1_count    = 0;
int span_b_f2_count    = 0;
int span_c_lib_count   = 0;
int span_c_f1_count    = 0;
int span_c_f2_count    = 0;
int span_d_lib_count   = 0;
int span_d_f1_count    = 0;
int span_d_f2_count    = 0;
int span_e_lib_count   = 0;
int span_e_f1_count    = 0;
int span_e_f2_count    = 0;
int span_f_lib_count   = 0;
int span_f_f1_count    = 0;
int span_f_f2_count    = 0;
int span_g_lib_count   = 0;
int span_g_f1_count    = 0;
int span_g_f2_count    = 0;
int span_h_lib_count   = 0;
int span_h_f1_count    = 0;
int span_h_f2_count    = 0;
int unknown_span_count = 0;

void reset_counts()
{
  span_a_lib_count   = 0;
  span_a_f1_count    = 0;
  span_a_f2_count    = 0;
  span_b_lib_count   = 0;
  span_b_f1_count    = 0;
  span_b_f2_count    = 0;
  span_c_lib_count   = 0;
  span_c_f1_count    = 0;
  span_c_f2_count    = 0;
  span_d_lib_count   = 0;
  span_d_f1_count    = 0;
  span_d_f2_count    = 0;
  span_e_lib_count   = 0;
  span_e_f1_count    = 0;
  span_e_f2_count    = 0;
  span_f_lib_count   = 0;
  span_f_f1_count    = 0;
  span_f_f2_count    = 0;
  span_g_lib_count   = 0;
  span_g_f1_count    = 0;
  span_g_f2_count    = 0;
  span_h_lib_count   = 0;
  span_h_f1_count    = 0;
  span_h_f2_count    = 0;
  unknown_span_count = 0;
}

class MyTracer : public trace::Tracer
{
public:
  nostd::shared_ptr<trace::Span> StartSpan(
      nostd::string_view name,
      const common::KeyValueIterable & /* attributes */,
      const trace::SpanContextKeyValueIterable & /* links */,
      const trace::StartSpanOptions & /* options */) noexcept override
  {
    nostd::shared_ptr<trace::Span> result(new trace::DefaultSpan(trace::SpanContext::GetInvalid()));

    /*
      Unit test code, no need to be fancy.
    */

    if (name == "A::library")
    {
      span_a_lib_count++;
    }
    else if (name == "A::f1")
    {
      span_a_f1_count++;
    }
    else if (name == "A::f2")
    {
      span_a_f2_count++;
    }
    else if (name == "B::library")
    {
      span_b_lib_count++;
    }
    else if (name == "B::f1")
    {
      span_b_f1_count++;
    }
    else if (name == "B::f2")
    {
      span_b_f2_count++;
    }
    else if (name == "C::library")
    {
      span_c_lib_count++;
    }
    else if (name == "C::f1")
    {
      span_c_f1_count++;
    }
    else if (name == "C::f2")
    {
      span_c_f2_count++;
    }
    else if (name == "D::library")
    {
      span_d_lib_count++;
    }
    else if (name == "D::f1")
    {
      span_d_f1_count++;
    }
    else if (name == "D::f2")
    {
      span_d_f2_count++;
    }
    else if (name == "E::library")
    {
      span_e_lib_count++;
    }
    else if (name == "E::f1")
    {
      span_e_f1_count++;
    }
    else if (name == "E::f2")
    {
      span_e_f2_count++;
    }
    else if (name == "F::library")
    {
      span_f_lib_count++;
    }
    else if (name == "F::f1")
    {
      span_f_f1_count++;
    }
    else if (name == "F::f2")
    {
      span_f_f2_count++;
    }
    else if (name == "G::library")
    {
      span_g_lib_count++;
    }
    else if (name == "G::f1")
    {
      span_g_f1_count++;
    }
    else if (name == "G::f2")
    {
      span_g_f2_count++;
    }
    else if (name == "H::library")
    {
      span_h_lib_count++;
    }
    else if (name == "H::f1")
    {
      span_h_f1_count++;
    }
    else if (name == "H::f2")
    {
      span_h_f2_count++;
    }
    else
    {
      unknown_span_count++;
    }

    return result;
  }

  void ForceFlushWithMicroseconds(uint64_t /* timeout */) noexcept override {}

  void CloseWithMicroseconds(uint64_t /* timeout */) noexcept override {}
};

class MyTracerProvider : public trace::TracerProvider
{
public:
  static std::shared_ptr<trace::TracerProvider> Create()
  {
    std::shared_ptr<trace::TracerProvider> result(new MyTracerProvider());
    return result;
  }

#if OPENTELEMETRY_ABI_VERSION_NO >= 2
  nostd::shared_ptr<trace::Tracer> GetTracer(
      nostd::string_view /* name */,
      nostd::string_view /* version */,
      nostd::string_view /* schema_url */,
      const common::KeyValueIterable * /* attributes */) noexcept override
  {
    nostd::shared_ptr<trace::Tracer> result(new MyTracer());
    return result;
  }
#else
  nostd::shared_ptr<trace::Tracer> GetTracer(nostd::string_view /* name */,
                                             nostd::string_view /* version */,
                                             nostd::string_view /* schema_url */) noexcept override
  {
    nostd::shared_ptr<trace::Tracer> result(new MyTracer());
    return result;
  }
#endif
};

void setup_otel()
{
  std::shared_ptr<opentelemetry::trace::TracerProvider> provider = MyTracerProvider::Create();

  // The whole point of this test is to make sure
  // that the API singleton behind SetTracerProvider()
  // works for all components, static and dynamic.

  // Set the global tracer provider
  trace_api::Provider::SetTracerProvider(provider);
}

void cleanup_otel()
{
  std::shared_ptr<opentelemetry::trace::TracerProvider> provider(
      new opentelemetry::trace::NoopTracerProvider());

  // Set the global tracer provider
  trace_api::Provider::SetTracerProvider(provider);
}

TEST(SingletonTest, Uniqueness)
{
  do_something();

  EXPECT_EQ(span_a_lib_count, 0);
  EXPECT_EQ(span_a_f1_count, 0);
  EXPECT_EQ(span_a_f2_count, 0);
  EXPECT_EQ(span_b_lib_count, 0);
  EXPECT_EQ(span_b_f1_count, 0);
  EXPECT_EQ(span_b_f2_count, 0);
  EXPECT_EQ(span_c_lib_count, 0);
  EXPECT_EQ(span_c_f1_count, 0);
  EXPECT_EQ(span_c_f2_count, 0);
  EXPECT_EQ(span_d_lib_count, 0);
  EXPECT_EQ(span_d_f1_count, 0);
  EXPECT_EQ(span_d_f2_count, 0);
  EXPECT_EQ(span_e_lib_count, 0);
  EXPECT_EQ(span_e_f1_count, 0);
  EXPECT_EQ(span_e_f2_count, 0);
  EXPECT_EQ(span_f_lib_count, 0);
  EXPECT_EQ(span_f_f1_count, 0);
  EXPECT_EQ(span_f_f2_count, 0);
  EXPECT_EQ(span_g_lib_count, 0);
  EXPECT_EQ(span_g_f1_count, 0);
  EXPECT_EQ(span_g_f2_count, 0);
  EXPECT_EQ(span_h_lib_count, 0);
  EXPECT_EQ(span_h_f1_count, 0);
  EXPECT_EQ(span_h_f2_count, 0);
  EXPECT_EQ(unknown_span_count, 0);

  reset_counts();
  setup_otel();

  do_something();

  EXPECT_EQ(span_a_lib_count, 1);
  EXPECT_EQ(span_a_f1_count, 2);
  EXPECT_EQ(span_a_f2_count, 1);
  EXPECT_EQ(span_b_lib_count, 1);
  EXPECT_EQ(span_b_f1_count, 2);
  EXPECT_EQ(span_b_f2_count, 1);
  EXPECT_EQ(span_c_lib_count, 1);
  EXPECT_EQ(span_c_f1_count, 2);
  EXPECT_EQ(span_c_f2_count, 1);
  EXPECT_EQ(span_d_lib_count, 1);
  EXPECT_EQ(span_d_f1_count, 2);
  EXPECT_EQ(span_d_f2_count, 1);
  EXPECT_EQ(span_e_lib_count, 1);
  EXPECT_EQ(span_e_f1_count, 2);
  EXPECT_EQ(span_e_f2_count, 1);
  EXPECT_EQ(span_f_lib_count, 1);
  EXPECT_EQ(span_f_f1_count, 2);
  EXPECT_EQ(span_f_f2_count, 1);

#ifndef BAZEL_BUILD
  EXPECT_EQ(span_g_lib_count, 1);
  EXPECT_EQ(span_g_f1_count, 2);
  EXPECT_EQ(span_g_f2_count, 1);
  EXPECT_EQ(span_h_lib_count, 1);
  EXPECT_EQ(span_h_f1_count, 2);
  EXPECT_EQ(span_h_f2_count, 1);
#endif

  EXPECT_EQ(unknown_span_count, 0);

  reset_counts();
  cleanup_otel();

  do_something();

  EXPECT_EQ(span_a_lib_count, 0);
  EXPECT_EQ(span_a_f1_count, 0);
  EXPECT_EQ(span_a_f2_count, 0);
  EXPECT_EQ(span_b_lib_count, 0);
  EXPECT_EQ(span_b_f1_count, 0);
  EXPECT_EQ(span_b_f2_count, 0);
  EXPECT_EQ(span_c_lib_count, 0);
  EXPECT_EQ(span_c_f1_count, 0);
  EXPECT_EQ(span_c_f2_count, 0);
  EXPECT_EQ(span_d_lib_count, 0);
  EXPECT_EQ(span_d_f1_count, 0);
  EXPECT_EQ(span_d_f2_count, 0);
  EXPECT_EQ(span_e_lib_count, 0);
  EXPECT_EQ(span_e_f1_count, 0);
  EXPECT_EQ(span_e_f2_count, 0);
  EXPECT_EQ(span_f_lib_count, 0);
  EXPECT_EQ(span_f_f1_count, 0);
  EXPECT_EQ(span_f_f2_count, 0);
  EXPECT_EQ(span_g_lib_count, 0);
  EXPECT_EQ(span_g_f1_count, 0);
  EXPECT_EQ(span_g_f2_count, 0);
  EXPECT_EQ(span_h_lib_count, 0);
  EXPECT_EQ(span_h_f1_count, 0);
  EXPECT_EQ(span_h_f2_count, 0);
  EXPECT_EQ(unknown_span_count, 0);
}
