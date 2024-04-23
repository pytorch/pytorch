// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>

#include "opentelemetry/common/kv_properties.h"
#include "opentelemetry/nostd/function_ref.h"
#include "opentelemetry/nostd/shared_ptr.h"
#include "opentelemetry/nostd/span.h"
#include "opentelemetry/nostd/string_view.h"
#include "opentelemetry/nostd/unique_ptr.h"
#include "opentelemetry/version.h"

#if OPENTELEMETRY_HAVE_WORKING_REGEX
#  include <regex>
#endif

OPENTELEMETRY_BEGIN_NAMESPACE
namespace trace
{

/**
 * TraceState carries tracing-system specific context in a list of key-value pairs. TraceState
 * allows different vendors to propagate additional information and inter-operate with their legacy
 * id formats.
 *
 * For more information, see the W3C Trace Context specification:
 * https://www.w3.org/TR/trace-context
 */
class OPENTELEMETRY_EXPORT TraceState
{
public:
  static constexpr int kKeyMaxSize         = 256;
  static constexpr int kValueMaxSize       = 256;
  static constexpr int kMaxKeyValuePairs   = 32;
  static constexpr auto kKeyValueSeparator = '=';
  static constexpr auto kMembersSeparator  = ',';

  OPENTELEMETRY_API_SINGLETON static nostd::shared_ptr<TraceState> GetDefault()
  {
    static nostd::shared_ptr<TraceState> ts{new TraceState()};
    return ts;
  }

  /**
   * Returns shared_ptr to a newly created TraceState parsed from the header provided.
   * @param header Encoding of the tracestate header defined by
   * the W3C Trace Context specification https://www.w3.org/TR/trace-context/
   * @return TraceState A new TraceState instance or DEFAULT
   */
  static nostd::shared_ptr<TraceState> FromHeader(nostd::string_view header) noexcept
  {

    common::KeyValueStringTokenizer kv_str_tokenizer(header);
    size_t cnt = kv_str_tokenizer.NumTokens();  // upper bound on number of kv pairs
    if (cnt > kMaxKeyValuePairs)
    {
      cnt = kMaxKeyValuePairs;
    }

    nostd::shared_ptr<TraceState> ts(new TraceState(cnt));
    bool kv_valid;
    nostd::string_view key, value;
    while (kv_str_tokenizer.next(kv_valid, key, value) && ts->kv_properties_->Size() < cnt)
    {
      if (kv_valid == false)
      {
        return GetDefault();
      }

      if (!IsValidKey(key) || !IsValidValue(value))
      {
        // invalid header. return empty TraceState
        ts->kv_properties_.reset(new common::KeyValueProperties());
        break;
      }

      ts->kv_properties_->AddEntry(key, value);
    }

    return ts;
  }

  /**
   * Creates a w3c tracestate header from TraceState object
   */
  std::string ToHeader() const noexcept
  {
    std::string header_s;
    bool first = true;
    kv_properties_->GetAllEntries(
        [&header_s, &first](nostd::string_view key, nostd::string_view value) noexcept {
          if (!first)
          {
            header_s.append(",");
          }
          else
          {
            first = false;
          }
          header_s.append(std::string(key.data(), key.size()));
          header_s.append(1, kKeyValueSeparator);
          header_s.append(std::string(value.data(), value.size()));
          return true;
        });
    return header_s;
  }

  /**
   *  Returns `value` associated with `key` passed as argument
   *  Returns empty string if key is invalid  or not found
   */
  bool Get(nostd::string_view key, std::string &value) const noexcept
  {
    if (!IsValidKey(key))
    {
      return false;
    }

    return kv_properties_->GetValue(key, value);
  }

  /**
   * Returns shared_ptr of `new` TraceState object with following mutations applied to the existing
   * instance: Update Key value: The updated value must be moved to beginning of List Add : The new
   * key-value pair SHOULD be added to beginning of List
   *
   * If the provided key-value pair is invalid, or results in transtate that violates the
   * tracecontext specification, empty TraceState instance will be returned.
   *
   * If the existing object has maximum list members, it's copy is returned.
   */
  nostd::shared_ptr<TraceState> Set(const nostd::string_view &key,
                                    const nostd::string_view &value) noexcept
  {
    auto curr_size = kv_properties_->Size();
    if (!IsValidKey(key) || !IsValidValue(value))
    {
      // max size reached or invalid key/value. Returning empty TraceState
      return TraceState::GetDefault();
    }
    auto allocate_size = curr_size;
    if (curr_size < kMaxKeyValuePairs)
    {
      allocate_size += 1;
    }
    nostd::shared_ptr<TraceState> ts(new TraceState(allocate_size));
    if (curr_size < kMaxKeyValuePairs)
    {
      // add new field first
      ts->kv_properties_->AddEntry(key, value);
    }
    // add rest of the fields.
    kv_properties_->GetAllEntries([&ts](nostd::string_view key, nostd::string_view value) {
      ts->kv_properties_->AddEntry(key, value);
      return true;
    });
    return ts;
  }

  /**
   * Returns shared_ptr to a `new` TraceState object after removing the attribute with given key (
   * if present )
   * @returns empty TraceState object if key is invalid
   * @returns copy of original TraceState object if key is not present (??)
   */
  nostd::shared_ptr<TraceState> Delete(const nostd::string_view &key) noexcept
  {
    if (!IsValidKey(key))
    {
      return TraceState::GetDefault();
    }
    auto curr_size     = kv_properties_->Size();
    auto allocate_size = curr_size;
    std::string unused;
    if (kv_properties_->GetValue(key, unused))
    {
      allocate_size -= 1;
    }
    nostd::shared_ptr<TraceState> ts(new TraceState(allocate_size));
    kv_properties_->GetAllEntries(
        [&ts, &key](nostd::string_view e_key, nostd::string_view e_value) {
          if (key != e_key)
            ts->kv_properties_->AddEntry(e_key, e_value);
          return true;
        });
    return ts;
  }

  // Returns true if there are no keys, false otherwise.
  bool Empty() const noexcept { return kv_properties_->Size() == 0; }

  // @return all key-values entris by repeatedly invoking the function reference passed as argument
  // for each entry
  bool GetAllEntries(
      nostd::function_ref<bool(nostd::string_view, nostd::string_view)> callback) const noexcept
  {
    return kv_properties_->GetAllEntries(callback);
  }
  /** Returns whether key is a valid key. See https://www.w3.org/TR/trace-context/#key
   * Identifiers MUST begin with a lowercase letter or a digit, and can only contain
   * lowercase letters (a-z), digits (0-9), underscores (_), dashes (-), asterisks (*),
   * and forward slashes (/).
   * For multi-tenant vendor scenarios, an at sign (@) can be used to prefix the vendor name.
   *
   */
  static bool IsValidKey(nostd::string_view key)
  {
#if OPENTELEMETRY_HAVE_WORKING_REGEX
    return IsValidKeyRegEx(key);
#else
    return IsValidKeyNonRegEx(key);
#endif
  }

  /** Returns whether value is a valid value. See https://www.w3.org/TR/trace-context/#value
   * The value is an opaque string containing up to 256 printable ASCII (RFC0020)
   *  characters ((i.e., the range 0x20 to 0x7E) except comma , and equal =)
   */
  static bool IsValidValue(nostd::string_view value)
  {
#if OPENTELEMETRY_HAVE_WORKING_REGEX
    return IsValidValueRegEx(value);
#else
    return IsValidValueNonRegEx(value);
#endif
  }

private:
  TraceState() : kv_properties_(new common::KeyValueProperties()) {}
  TraceState(size_t size) : kv_properties_(new common::KeyValueProperties(size)) {}

  static nostd::string_view TrimString(nostd::string_view str, size_t left, size_t right)
  {
    while (str[static_cast<std::size_t>(right)] == ' ' && left < right)
    {
      right--;
    }
    while (str[static_cast<std::size_t>(left)] == ' ' && left < right)
    {
      left++;
    }
    return str.substr(left, right - left + 1);
  }

#if OPENTELEMETRY_HAVE_WORKING_REGEX
  static bool IsValidKeyRegEx(nostd::string_view key)
  {
    static std::regex reg_key("^[a-z0-9][a-z0-9*_\\-/]{0,255}$");
    static std::regex reg_key_multitenant(
        "^[a-z0-9][a-z0-9*_\\-/]{0,240}(@)[a-z0-9][a-z0-9*_\\-/]{0,13}$");
    std::string key_s(key.data(), key.size());
    if (std::regex_match(key_s, reg_key) || std::regex_match(key_s, reg_key_multitenant))
    {
      return true;
    }
    return false;
  }

  static bool IsValidValueRegEx(nostd::string_view value)
  {
    // Hex 0x20 to 0x2B, 0x2D to 0x3C, 0x3E to 0x7E
    static std::regex reg_value(
        "^[\\x20-\\x2B\\x2D-\\x3C\\x3E-\\x7E]{0,255}[\\x21-\\x2B\\x2D-\\x3C\\x3E-\\x7E]$");
    // Need to benchmark without regex, as a string object is created here.
    return std::regex_match(std::string(value.data(), value.size()), reg_value);
  }
#else
  static bool IsValidKeyNonRegEx(nostd::string_view key)
  {
    if (key.empty() || key.size() > kKeyMaxSize || !IsLowerCaseAlphaOrDigit(key[0]))
    {
      return false;
    }

    int ats = 0;

    for (const char c : key)
    {
      if (!IsLowerCaseAlphaOrDigit(c) && c != '_' && c != '-' && c != '@' && c != '*' && c != '/')
      {
        return false;
      }
      if ((c == '@') && (++ats > 1))
      {
        return false;
      }
    }
    return true;
  }

  static bool IsValidValueNonRegEx(nostd::string_view value)
  {
    if (value.empty() || value.size() > kValueMaxSize)
    {
      return false;
    }

    for (const char c : value)
    {
      if (c < ' ' || c > '~' || c == ',' || c == '=')
      {
        return false;
      }
    }
    return true;
  }
#endif

  static bool IsLowerCaseAlphaOrDigit(char c) { return isdigit(c) || islower(c); }

private:
  // Store entries in a C-style array to avoid using std::array or std::vector.
  nostd::unique_ptr<common::KeyValueProperties> kv_properties_;
};

}  // namespace trace
OPENTELEMETRY_END_NAMESPACE
