// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "opentelemetry/common/macros.h"
#include "opentelemetry/context/context.h"
#include "opentelemetry/nostd/shared_ptr.h"
#include "opentelemetry/nostd/string_view.h"
#include "opentelemetry/nostd/unique_ptr.h"
#include "opentelemetry/version.h"

OPENTELEMETRY_BEGIN_NAMESPACE
namespace context
{
// The Token object provides is returned when attaching objects to the
// RuntimeContext object and is associated with a context object, and
// can be provided to the RuntimeContext Detach method to remove the
// associated context from the RuntimeContext.
class Token
{
public:
  bool operator==(const Context &other) const noexcept { return context_ == other; }

  ~Token() noexcept;

private:
  friend class RuntimeContextStorage;

  // A constructor that sets the token's Context object to the
  // one that was passed in.
  Token(const Context &context) : context_(context) {}

  const Context context_;
};

/**
 * RuntimeContextStorage is used by RuntimeContext to store Context frames.
 *
 * Custom context management strategies can be implemented by deriving from
 * this class and passing an initialized RuntimeContextStorage object to
 * RuntimeContext::SetRuntimeContextStorage.
 */
class OPENTELEMETRY_EXPORT RuntimeContextStorage
{
public:
  /**
   * Return the current context.
   * @return the current context
   */
  virtual Context GetCurrent() noexcept = 0;

  /**
   * Set the current context.
   * @param the new current context
   * @return a token for the new current context. This never returns a nullptr.
   */
  virtual nostd::unique_ptr<Token> Attach(const Context &context) noexcept = 0;

  /**
   * Detach the context related to the given token.
   * @param token a token related to a context
   * @return true if the context could be detached
   */
  virtual bool Detach(Token &token) noexcept = 0;

  virtual ~RuntimeContextStorage() {}

protected:
  nostd::unique_ptr<Token> CreateToken(const Context &context) noexcept
  {
    return nostd::unique_ptr<Token>(new Token(context));
  }
};

/**
 * Construct and return the default RuntimeContextStorage
 * @return a ThreadLocalContextStorage
 */
static RuntimeContextStorage *GetDefaultStorage() noexcept;

// Provides a wrapper for propagating the context object globally.
//
// By default, a thread-local runtime context storage is used.
class OPENTELEMETRY_EXPORT RuntimeContext
{
public:
  // Return the current context.
  static Context GetCurrent() noexcept { return GetRuntimeContextStorage()->GetCurrent(); }

  // Sets the current 'Context' object. Returns a token
  // that can be used to reset to the previous Context.
  static nostd::unique_ptr<Token> Attach(const Context &context) noexcept
  {
    return GetRuntimeContextStorage()->Attach(context);
  }

  // Resets the context to a previous value stored in the
  // passed in token. Returns true if successful, false otherwise
  static bool Detach(Token &token) noexcept { return GetRuntimeContextStorage()->Detach(token); }

  // Sets the Key and Value into the passed in context or if a context is not
  // passed in, the RuntimeContext.
  // Should be used to SetValues to the current RuntimeContext, is essentially
  // equivalent to RuntimeContext::GetCurrent().SetValue(key,value). Keep in
  // mind that the current RuntimeContext will not be changed, and the new
  // context will be returned.
  static Context SetValue(nostd::string_view key,
                          const ContextValue &value,
                          Context *context = nullptr) noexcept
  {
    Context temp_context;
    if (context == nullptr)
    {
      temp_context = GetCurrent();
    }
    else
    {
      temp_context = *context;
    }
    return temp_context.SetValue(key, value);
  }

  // Returns the value associated with the passed in key and either the
  // passed in context* or the runtime context if a context is not passed in.
  // Should be used to get values from the current RuntimeContext, is
  // essentially equivalent to RuntimeContext::GetCurrent().GetValue(key).
  static ContextValue GetValue(nostd::string_view key, Context *context = nullptr) noexcept
  {
    Context temp_context;
    if (context == nullptr)
    {
      temp_context = GetCurrent();
    }
    else
    {
      temp_context = *context;
    }
    return temp_context.GetValue(key);
  }

  /**
   * Provide a custom runtime context storage.
   *
   * This provides a possibility to override the default thread-local runtime
   * context storage. This has to be set before any spans are created by the
   * application, otherwise the behavior is undefined.
   *
   * @param storage a custom runtime context storage
   */
  static void SetRuntimeContextStorage(nostd::shared_ptr<RuntimeContextStorage> storage) noexcept
  {
    GetStorage() = storage;
  }

  /**
   * Provide a pointer to const runtime context storage.
   *
   * The returned pointer can only be used for extending the lifetime of the runtime context
   * storage.
   *
   */
  static nostd::shared_ptr<const RuntimeContextStorage> GetConstRuntimeContextStorage() noexcept
  {
    return GetRuntimeContextStorage();
  }

private:
  static nostd::shared_ptr<RuntimeContextStorage> GetRuntimeContextStorage() noexcept
  {
    return GetStorage();
  }

  OPENTELEMETRY_API_SINGLETON static nostd::shared_ptr<RuntimeContextStorage> &GetStorage() noexcept
  {
    static nostd::shared_ptr<RuntimeContextStorage> context(GetDefaultStorage());
    return context;
  }
};

inline Token::~Token() noexcept
{
  context::RuntimeContext::Detach(*this);
}

// The ThreadLocalContextStorage class is a derived class from
// RuntimeContextStorage and provides a wrapper for propagating context through
// cpp thread locally. This file must be included to use the RuntimeContext
// class if another implementation has not been registered.
class ThreadLocalContextStorage : public RuntimeContextStorage
{
public:
  ThreadLocalContextStorage() noexcept = default;

  // Return the current context.
  Context GetCurrent() noexcept override { return GetStack().Top(); }

  // Resets the context to the value previous to the passed in token. This will
  // also detach all child contexts of the passed in token.
  // Returns true if successful, false otherwise.
  bool Detach(Token &token) noexcept override
  {
    // In most cases, the context to be detached is on the top of the stack.
    if (token == GetStack().Top())
    {
      GetStack().Pop();
      return true;
    }

    if (!GetStack().Contains(token))
    {
      return false;
    }

    while (!(token == GetStack().Top()))
    {
      GetStack().Pop();
    }

    GetStack().Pop();

    return true;
  }

  // Sets the current 'Context' object. Returns a token
  // that can be used to reset to the previous Context.
  nostd::unique_ptr<Token> Attach(const Context &context) noexcept override
  {
    GetStack().Push(context);
    return CreateToken(context);
  }

private:
  // A nested class to store the attached contexts in a stack.
  class Stack
  {
    friend class ThreadLocalContextStorage;

    Stack() noexcept : size_(0), capacity_(0), base_(nullptr) {}

    // Pops the top Context off the stack.
    void Pop() noexcept
    {
      if (size_ == 0)
      {
        return;
      }
      // Store empty Context before decrementing `size`, to ensure
      // the shared_ptr object (if stored in prev context object ) are released.
      // The stack is not resized, and the unused memory would be reutilised
      // for subsequent context storage.
      base_[size_ - 1] = Context();
      size_ -= 1;
    }

    bool Contains(const Token &token) const noexcept
    {
      for (size_t pos = size_; pos > 0; --pos)
      {
        if (token == base_[pos - 1])
        {
          return true;
        }
      }

      return false;
    }

    // Returns the Context at the top of the stack.
    Context Top() const noexcept
    {
      if (size_ == 0)
      {
        return Context();
      }
      return base_[size_ - 1];
    }

    // Pushes the passed in context pointer to the top of the stack
    // and resizes if necessary.
    void Push(const Context &context) noexcept
    {
      size_++;
      if (size_ > capacity_)
      {
        Resize(size_ * 2);
      }
      base_[size_ - 1] = context;
    }

    // Reallocates the storage array to the pass in new capacity size.
    void Resize(size_t new_capacity) noexcept
    {
      size_t old_size = size_ - 1;
      if (new_capacity == 0)
      {
        new_capacity = 2;
      }
      Context *temp = new Context[new_capacity];
      if (base_ != nullptr)
      {
        // vs2015 does not like this construct considering it unsafe:
        // - std::copy(base_, base_ + old_size, temp);
        // Ref.
        // https://stackoverflow.com/questions/12270224/xutility2227-warning-c4996-std-copy-impl
        for (size_t i = 0; i < (std::min)(old_size, new_capacity); i++)
        {
          temp[i] = base_[i];
        }
        delete[] base_;
      }
      base_     = temp;
      capacity_ = new_capacity;
    }

    ~Stack() noexcept { delete[] base_; }

    size_t size_;
    size_t capacity_;
    Context *base_;
  };

  OPENTELEMETRY_API_SINGLETON Stack &GetStack()
  {
    static thread_local Stack stack_ = Stack();
    return stack_;
  }
};

static RuntimeContextStorage *GetDefaultStorage() noexcept
{
  return new ThreadLocalContextStorage();
}
}  // namespace context
OPENTELEMETRY_END_NAMESPACE
