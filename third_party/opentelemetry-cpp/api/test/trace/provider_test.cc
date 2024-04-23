// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#include "opentelemetry/trace/provider.h"
#include "opentelemetry/nostd/shared_ptr.h"
#include "opentelemetry/trace/tracer_provider.h"

#include <gtest/gtest.h>

using opentelemetry::trace::Provider;
using opentelemetry::trace::Tracer;
using opentelemetry::trace::TracerProvider;

namespace nostd = opentelemetry::nostd;

class TestProvider : public TracerProvider
{

#if OPENTELEMETRY_ABI_VERSION_NO >= 2
  nostd::shared_ptr<Tracer> GetTracer(
      nostd::string_view /* name */,
      nostd::string_view /* version */,
      nostd::string_view /* schema_url */,
      const opentelemetry::common::KeyValueIterable * /* attributes */) noexcept override
  {
    return nostd::shared_ptr<Tracer>(nullptr);
  }
#else
  nostd::shared_ptr<Tracer> GetTracer(nostd::string_view /* name */,
                                      nostd::string_view /* version */,
                                      nostd::string_view /* schema_url */) noexcept override
  {
    return nostd::shared_ptr<Tracer>(nullptr);
  }
#endif
};

TEST(Provider, GetTracerProviderDefault)
{
  auto tf = Provider::GetTracerProvider();
  EXPECT_NE(nullptr, tf);
}

TEST(Provider, SetTracerProvider)
{
  auto tf = nostd::shared_ptr<TracerProvider>(new TestProvider());
  Provider::SetTracerProvider(tf);
  ASSERT_EQ(tf, Provider::GetTracerProvider());
}
