// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#include "opentelemetry/context/runtime_context.h"
#include "opentelemetry/context/context.h"

#include <algorithm>
#include <map>

#include <gtest/gtest.h>

using namespace opentelemetry;

// Tests that GetCurrent returns the current context
TEST(RuntimeContextTest, GetCurrent)
{
  std::map<std::string, context::ContextValue> map_test = {{"test_key", (int64_t)123}};
  context::Context test_context                         = context::Context(map_test);
  auto old_context = context::RuntimeContext::Attach(test_context);
  EXPECT_EQ(context::RuntimeContext::GetCurrent(), test_context);
}

// Tests that detach resets the context to the previous context
TEST(RuntimeContextTest, Detach)
{
  std::map<std::string, context::ContextValue> map_test = {{"test_key", (int64_t)123}};
  context::Context test_context                         = context::Context(map_test);
  context::Context foo_context                          = context::Context(map_test);

  auto test_context_token = context::RuntimeContext::Attach(test_context);
  auto foo_context_token  = context::RuntimeContext::Attach(foo_context);

  foo_context_token.reset();
  EXPECT_EQ(context::RuntimeContext::GetCurrent(), test_context);
  test_context_token.reset();
}

// Tests that detach returns false when the wrong context is provided
TEST(RuntimeContextTest, DetachWrongContext)
{
  std::map<std::string, context::ContextValue> map_test = {{"test_key", (int64_t)123}};
  context::Context test_context                         = context::Context(map_test);
  auto test_context_token = context::RuntimeContext::Attach(test_context);
  EXPECT_TRUE(context::RuntimeContext::Detach(*test_context_token));
  EXPECT_FALSE(context::RuntimeContext::Detach(*test_context_token));
}

// Tests that the ThreadLocalContext can handle three attached contexts
TEST(RuntimeContextTest, ThreeAttachDetach)
{
  std::map<std::string, context::ContextValue> map_test = {{"test_key", (int64_t)123}};
  context::Context test_context                         = context::Context(map_test);
  context::Context foo_context                          = context::Context(map_test);
  context::Context other_context                        = context::Context(map_test);
  auto test_context_token  = context::RuntimeContext::Attach(test_context);
  auto foo_context_token   = context::RuntimeContext::Attach(foo_context);
  auto other_context_token = context::RuntimeContext::Attach(other_context);

  EXPECT_TRUE(context::RuntimeContext::Detach(*other_context_token));
  EXPECT_TRUE(context::RuntimeContext::Detach(*foo_context_token));
  EXPECT_TRUE(context::RuntimeContext::Detach(*test_context_token));
}

// Tests that SetValue returns a context with the passed in data and the
// RuntimeContext data when a context is not passed into the
// RuntimeContext::SetValue method.
TEST(RuntimeContextTest, SetValueRuntimeContext)
{
  context::Context foo_context  = context::Context("foo_key", (int64_t)596);
  auto old_context_token        = context::RuntimeContext::Attach(foo_context);
  context::Context test_context = context::RuntimeContext::SetValue("test_key", (int64_t)123);
  EXPECT_EQ(nostd::get<int64_t>(test_context.GetValue("test_key")), 123);
  EXPECT_EQ(nostd::get<int64_t>(test_context.GetValue("foo_key")), 596);
}

// Tests that SetValue returns a context with the passed in data and the
// passed in context data when a context* is passed into the
// RuntimeContext::SetValue method.
TEST(RuntimeContextTest, SetValueOtherContext)
{
  context::Context foo_context = context::Context("foo_key", (int64_t)596);
  context::Context test_context =
      context::RuntimeContext::SetValue("test_key", (int64_t)123, &foo_context);
  EXPECT_EQ(nostd::get<int64_t>(test_context.GetValue("test_key")), 123);
  EXPECT_EQ(nostd::get<int64_t>(test_context.GetValue("foo_key")), 596);
}

// Tests that SetValue returns the ContextValue associated with the
// passed in string and the current Runtime Context
TEST(RuntimeContextTest, GetValueRuntimeContext)
{
  context::Context foo_context = context::Context("foo_key", (int64_t)596);
  auto old_context_token       = context::RuntimeContext::Attach(foo_context);
  EXPECT_EQ(nostd::get<int64_t>(context::RuntimeContext::GetValue("foo_key")), 596);
}

// Tests that SetValue returns the ContextValue associated with the
// passed in string and the passed in context
TEST(RuntimeContextTest, GetValueOtherContext)
{
  context::Context foo_context = context::Context("foo_key", (int64_t)596);
  EXPECT_EQ(nostd::get<int64_t>(context::RuntimeContext::GetValue("foo_key", &foo_context)), 596);
}

// Test that any possible order of context detaching doesn't mess up the stack.
TEST(RuntimeContextTest, DetachOutOfOrder)
{
  std::vector<size_t> indices;
  indices.push_back(0);
  indices.push_back(1);
  indices.push_back(2);
  indices.push_back(3);

  std::vector<context::Context> contexts;
  for (auto i : indices)
  {
    contexts.push_back(context::Context("index", (int64_t)i));
  }

  do
  {
    std::vector<nostd::unique_ptr<context::Token>> tokens;

    for (auto &c : contexts)
    {
      tokens.push_back(context::RuntimeContext::Attach(c));
    }

    for (size_t i : indices)
    {
      auto token = std::move(tokens.at(i));
      context::RuntimeContext::Detach(*token);
    }

    EXPECT_EQ(context::RuntimeContext::GetCurrent(), context::Context());

  } while (std::next_permutation(indices.begin(), indices.end()));
}
