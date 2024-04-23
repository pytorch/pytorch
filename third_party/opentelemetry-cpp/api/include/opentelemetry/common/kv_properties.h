// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "opentelemetry/common/key_value_iterable_view.h"
#include "opentelemetry/common/string_util.h"
#include "opentelemetry/nostd/function_ref.h"
#include "opentelemetry/nostd/string_view.h"
#include "opentelemetry/nostd/unique_ptr.h"
#include "opentelemetry/version.h"

#include <cstring>
#include <string>
#include <type_traits>

OPENTELEMETRY_BEGIN_NAMESPACE
namespace common
{

// Constructor parameter for KeyValueStringTokenizer
struct KeyValueStringTokenizerOptions
{
  char member_separator     = ',';
  char key_value_separator  = '=';
  bool ignore_empty_members = true;
};

// Tokenizer for key-value headers
class KeyValueStringTokenizer
{
public:
  KeyValueStringTokenizer(
      nostd::string_view str,
      const KeyValueStringTokenizerOptions &opts = KeyValueStringTokenizerOptions()) noexcept
      : str_(str), opts_(opts), index_(0)
  {}

  static nostd::string_view GetDefaultKeyOrValue()
  {
    static std::string default_str = "";
    return default_str;
  }

  // Returns next key value in the string header
  // @param valid_kv : if the found kv pair is valid or not
  // @param key : key in kv pair
  // @param key : value in kv pair
  // @returns true if next kv pair was found, false otherwise.
  bool next(bool &valid_kv, nostd::string_view &key, nostd::string_view &value) noexcept
  {
    valid_kv = true;
    while (index_ < str_.size())
    {
      bool is_empty_pair = false;
      size_t end         = str_.find(opts_.member_separator, index_);
      if (end == std::string::npos)
      {
        end = str_.size() - 1;
      }
      else if (end == index_)  // empty pair. do not update end
      {
        is_empty_pair = true;
      }
      else
      {
        end--;
      }

      auto list_member = StringUtil::Trim(str_, index_, end);
      if (list_member.size() == 0 || is_empty_pair)
      {
        // empty list member
        index_ = end + 2 - is_empty_pair;
        if (opts_.ignore_empty_members)
        {
          continue;
        }

        valid_kv = true;
        key      = GetDefaultKeyOrValue();
        value    = GetDefaultKeyOrValue();
        return true;
      }

      auto key_end_pos = list_member.find(opts_.key_value_separator);
      if (key_end_pos == std::string::npos)
      {
        // invalid member
        valid_kv = false;
      }
      else
      {
        key   = list_member.substr(0, key_end_pos);
        value = list_member.substr(key_end_pos + 1);
      }

      index_ = end + 2;

      return true;
    }

    // no more entries remaining
    return false;
  }

  // Returns total number of tokens in header string
  size_t NumTokens() const noexcept
  {
    size_t cnt = 0, begin = 0;
    while (begin < str_.size())
    {
      ++cnt;
      size_t end = str_.find(opts_.member_separator, begin);
      if (end == std::string::npos)
      {
        break;
      }

      begin = end + 1;
    }

    return cnt;
  }

  // Resets the iterator
  void reset() noexcept { index_ = 0; }

private:
  nostd::string_view str_;
  KeyValueStringTokenizerOptions opts_;
  size_t index_;
};

// Class to store fixed size array of key-value pairs of string type
class KeyValueProperties
{
  // Class to store key-value pairs of string types
public:
  class Entry
  {
  public:
    Entry() : key_(nullptr), value_(nullptr) {}

    // Copy constructor
    Entry(const Entry &copy)
    {
      key_   = CopyStringToPointer(copy.key_.get());
      value_ = CopyStringToPointer(copy.value_.get());
    }

    // Copy assignment operator
    Entry &operator=(Entry &other)
    {
      key_   = CopyStringToPointer(other.key_.get());
      value_ = CopyStringToPointer(other.value_.get());
      return *this;
    }

    // Move contructor and assignment operator
    Entry(Entry &&other) = default;
    Entry &operator=(Entry &&other) = default;

    // Creates an Entry for a given key-value pair.
    Entry(nostd::string_view key, nostd::string_view value)
    {
      key_   = CopyStringToPointer(key);
      value_ = CopyStringToPointer(value);
    }

    // Gets the key associated with this entry.
    nostd::string_view GetKey() const noexcept { return key_.get(); }

    // Gets the value associated with this entry.
    nostd::string_view GetValue() const noexcept { return value_.get(); }

    // Sets the value for this entry. This overrides the previous value.
    void SetValue(nostd::string_view value) noexcept { value_ = CopyStringToPointer(value); }

  private:
    // Store key and value as raw char pointers to avoid using std::string.
    nostd::unique_ptr<const char[]> key_;
    nostd::unique_ptr<const char[]> value_;

    // Copies string into a buffer and returns a unique_ptr to the buffer.
    // This is a workaround for the fact that memcpy doesn't accept a const destination.
    nostd::unique_ptr<const char[]> CopyStringToPointer(nostd::string_view str)
    {
      char *temp = new char[str.size() + 1];
      memcpy(temp, str.data(), str.size());
      temp[str.size()] = '\0';
      return nostd::unique_ptr<const char[]>(temp);
    }
  };

  // Maintain the number of entries in entries_.
  size_t num_entries_;

  // Max size of allocated array
  size_t max_num_entries_;

  // Store entries in a C-style array to avoid using std::array or std::vector.
  nostd::unique_ptr<Entry[]> entries_;

public:
  // Create Key-value list of given size
  // @param size : Size of list.
  KeyValueProperties(size_t size) noexcept
      : num_entries_(0), max_num_entries_(size), entries_(new Entry[size])
  {}

  // Create Empty Key-Value list
  KeyValueProperties() noexcept : num_entries_(0), max_num_entries_(0), entries_(nullptr) {}

  template <class T, class = typename std::enable_if<detail::is_key_value_iterable<T>::value>::type>
  KeyValueProperties(const T &keys_and_values) noexcept
      : num_entries_(0),
        max_num_entries_(keys_and_values.size()),
        entries_(new Entry[max_num_entries_])
  {
    for (auto &e : keys_and_values)
    {
      Entry entry(e.first, e.second);
      (entries_.get())[num_entries_++] = std::move(entry);
    }
  }

  // Adds new kv pair into kv properties
  void AddEntry(nostd::string_view key, nostd::string_view value) noexcept
  {
    if (num_entries_ < max_num_entries_)
    {
      Entry entry(key, value);
      (entries_.get())[num_entries_++] = std::move(entry);
    }
  }

  // Returns all kv pair entries
  bool GetAllEntries(
      nostd::function_ref<bool(nostd::string_view, nostd::string_view)> callback) const noexcept
  {
    for (size_t i = 0; i < num_entries_; i++)
    {
      auto &entry = (entries_.get())[i];
      if (!callback(entry.GetKey(), entry.GetValue()))
      {
        return false;
      }
    }
    return true;
  }

  // Return value for key if exists, return false otherwise
  bool GetValue(nostd::string_view key, std::string &value) const noexcept
  {
    for (size_t i = 0; i < num_entries_; i++)
    {
      auto &entry = (entries_.get())[i];
      if (entry.GetKey() == key)
      {
        const auto &entry_value = entry.GetValue();
        value                   = std::string(entry_value.data(), entry_value.size());
        return true;
      }
    }
    return false;
  }

  size_t Size() const noexcept { return num_entries_; }
};
}  // namespace common
OPENTELEMETRY_END_NAMESPACE
