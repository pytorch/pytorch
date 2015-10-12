// Protocol Buffers - Google's data interchange format
// Copyright 2008 Google Inc.  All rights reserved.
// https://developers.google.com/protocol-buffers/
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

#include <google/protobuf/util/internal/json_stream_parser.h>

#include <google/protobuf/stubs/logging.h>
#include <google/protobuf/stubs/common.h>
#include <google/protobuf/stubs/time.h>
#include <google/protobuf/util/internal/expecting_objectwriter.h>
#include <google/protobuf/util/internal/object_writer.h>
#include <google/protobuf/stubs/strutil.h>
#include <gtest/gtest.h>
#include <google/protobuf/stubs/status.h>


namespace google {
namespace protobuf {
namespace util {
using util::Status;
namespace error {
using util::error::INVALID_ARGUMENT;
}  // namespace error
namespace converter {

using util::Status;

// Tests for the JSON Stream Parser. These tests are intended to be
// comprehensive and cover the following:
//
// Positive tests:
// - true, false, null
// - empty object or array.
// - negative and positive double and int, unsigned int
// - single and double quoted strings
// - string key, unquoted key, numeric key
// - array containing array, object, value
// - object containing array, object, value
// - unicode handling in strings
// - ascii escaping (\b, \f, \n, \r, \t, \v)
// - trailing commas
//
// Negative tests:
// - illegal literals
// - mismatched quotes failure on strings
// - unterminated string failure
// - unexpected end of string failure
// - mismatched object and array closing
// - Failure to close array or object
// - numbers too large
// - invalid unicode escapes.
// - invalid unicode sequences.
// - numbers as keys
//
// For each test we split the input string on every possible character to ensure
// the parser is able to handle arbitrarily split input for all cases. We also
// do a final test of the entire test case one character at a time.
class JsonStreamParserTest : public ::testing::Test {
 protected:
  JsonStreamParserTest() : mock_(), ow_(&mock_) {}
  virtual ~JsonStreamParserTest() {}

  util::Status RunTest(StringPiece json, int split, bool coerce_utf8 = false) {
    JsonStreamParser parser(&mock_);

    // Special case for split == length, test parsing one character at a time.
    if (split == json.length()) {
      GOOGLE_LOG(INFO) << "Testing split every char: " << json;
      for (int i = 0; i < json.length(); ++i) {
        StringPiece single = json.substr(i, 1);
        util::Status result = parser.Parse(single);
        if (!result.ok()) {
          return result;
        }
      }
      return parser.FinishParse();
    }

    // Normal case, split at the split point and parse two substrings.
    StringPiece first = json.substr(0, split);
    StringPiece rest = json.substr(split);
    GOOGLE_LOG(INFO) << "Testing split: " << first << "><" << rest;
    util::Status result = parser.Parse(first);
    if (result.ok()) {
      result = parser.Parse(rest);
      if (result.ok()) {
        result = parser.FinishParse();
      }
    }
    return result;
  }

  void DoTest(StringPiece json, int split, bool coerce_utf8 = false) {
    util::Status result = RunTest(json, split, coerce_utf8);
    if (!result.ok()) {
      GOOGLE_LOG(WARNING) << result;
    }
    EXPECT_OK(result);
  }

  void DoErrorTest(StringPiece json, int split, StringPiece error_prefix) {
    util::Status result = RunTest(json, split);
    EXPECT_EQ(util::error::INVALID_ARGUMENT, result.error_code());
    StringPiece error_message(result.error_message());
    EXPECT_EQ(error_prefix, error_message.substr(0, error_prefix.size()));
  }


  MockObjectWriter mock_;
  ExpectingObjectWriter ow_;
};


// Positive tests

// - true, false, null
TEST_F(JsonStreamParserTest, SimpleTrue) {
  StringPiece str = "true";
  for (int i = 0; i <= str.length(); ++i) {
    ow_.RenderBool("", true);
    DoTest(str, i);
  }
}

TEST_F(JsonStreamParserTest, SimpleFalse) {
  StringPiece str = "false";
  for (int i = 0; i <= str.length(); ++i) {
    ow_.RenderBool("", false);
    DoTest(str, i);
  }
}

TEST_F(JsonStreamParserTest, SimpleNull) {
  StringPiece str = "null";
  for (int i = 0; i <= str.length(); ++i) {
    ow_.RenderNull("");
    DoTest(str, i);
  }
}

// - empty object and array.
TEST_F(JsonStreamParserTest, EmptyObject) {
  StringPiece str = "{}";
  for (int i = 0; i <= str.length(); ++i) {
    ow_.StartObject("")->EndObject();
    DoTest(str, i);
  }
}

TEST_F(JsonStreamParserTest, EmptyList) {
  StringPiece str = "[]";
  for (int i = 0; i <= str.length(); ++i) {
    ow_.StartList("")->EndList();
    DoTest(str, i);
  }
}

// - negative and positive double and int, unsigned int
TEST_F(JsonStreamParserTest, SimpleDouble) {
  StringPiece str = "42.5";
  for (int i = 0; i <= str.length(); ++i) {
    ow_.RenderDouble("", 42.5);
    DoTest(str, i);
  }
}

TEST_F(JsonStreamParserTest, ScientificDouble) {
  StringPiece str = "1.2345e-10";
  for (int i = 0; i < str.length(); ++i) {
    ow_.RenderDouble("", 1.2345e-10);
    DoTest(str, i);
  }
}

TEST_F(JsonStreamParserTest, SimpleNegativeDouble) {
  StringPiece str = "-1045.235";
  for (int i = 0; i <= str.length(); ++i) {
    ow_.RenderDouble("", -1045.235);
    DoTest(str, i);
  }
}

TEST_F(JsonStreamParserTest, SimpleInt) {
  StringPiece str = "123456";
  for (int i = 0; i <= str.length(); ++i) {
    ow_.RenderUint64("", 123456);
    DoTest(str, i);
  }
}

TEST_F(JsonStreamParserTest, SimpleNegativeInt) {
  StringPiece str = "-79497823553162765";
  for (int i = 0; i <= str.length(); ++i) {
    ow_.RenderInt64("", -79497823553162765LL);
    DoTest(str, i);
  }
}

TEST_F(JsonStreamParserTest, SimpleUnsignedInt) {
  StringPiece str = "11779497823553162765";
  for (int i = 0; i <= str.length(); ++i) {
    ow_.RenderUint64("", 11779497823553162765ULL);
    DoTest(str, i);
  }
}

// - single and double quoted strings
TEST_F(JsonStreamParserTest, EmptyDoubleQuotedString) {
  StringPiece str = "\"\"";
  for (int i = 0; i <= str.length(); ++i) {
    ow_.RenderString("", "");
    DoTest(str, i);
  }
}

TEST_F(JsonStreamParserTest, EmptySingleQuotedString) {
  StringPiece str = "''";
  for (int i = 0; i <= str.length(); ++i) {
    ow_.RenderString("", "");
    DoTest(str, i);
  }
}

TEST_F(JsonStreamParserTest, SimpleDoubleQuotedString) {
  StringPiece str = "\"Some String\"";
  for (int i = 0; i <= str.length(); ++i) {
    ow_.RenderString("", "Some String");
    DoTest(str, i);
  }
}

TEST_F(JsonStreamParserTest, SimpleSingleQuotedString) {
  StringPiece str = "'Another String'";
  for (int i = 0; i <= str.length(); ++i) {
    ow_.RenderString("", "Another String");
    DoTest(str, i);
  }
}

// - string key, unquoted key, numeric key
TEST_F(JsonStreamParserTest, ObjectKeyTypes) {
  StringPiece str =
      "{'s': true, \"d\": false, key: null, snake_key: [], camelKey: {}}";
  for (int i = 0; i <= str.length(); ++i) {
    ow_.StartObject("")
        ->RenderBool("s", true)
        ->RenderBool("d", false)
        ->RenderNull("key")
        ->StartList("snake_key")
        ->EndList()
        ->StartObject("camelKey")
        ->EndObject()
        ->EndObject();
    DoTest(str, i);
  }
}

// - array containing array, object, values (true, false, null, num, string)
TEST_F(JsonStreamParserTest, ArrayValues) {
  StringPiece str =
      "[true, false, null, 'a string', \"another string\", [22, -127, 45.3, "
      "-1056.4, 11779497823553162765], {'key': true}]";
  for (int i = 0; i <= str.length(); ++i) {
    ow_.StartList("")
        ->RenderBool("", true)
        ->RenderBool("", false)
        ->RenderNull("")
        ->RenderString("", "a string")
        ->RenderString("", "another string")
        ->StartList("")
        ->RenderUint64("", 22)
        ->RenderInt64("", -127)
        ->RenderDouble("", 45.3)
        ->RenderDouble("", -1056.4)
        ->RenderUint64("", 11779497823553162765ULL)
        ->EndList()
        ->StartObject("")
        ->RenderBool("key", true)
        ->EndObject()
        ->EndList();
    DoTest(str, i);
  }
}

// - object containing array, object, value (true, false, null, num, string)
TEST_F(JsonStreamParserTest, ObjectValues) {
  StringPiece str =
      "{t: true, f: false, n: null, s: 'a string', d: \"another string\", pi: "
      "22, ni: -127, pd: 45.3, nd: -1056.4, pl: 11779497823553162765, l: [[]], "
      "o: {'key': true}}";
  for (int i = 0; i <= str.length(); ++i) {
    ow_.StartObject("")
        ->RenderBool("t", true)
        ->RenderBool("f", false)
        ->RenderNull("n")
        ->RenderString("s", "a string")
        ->RenderString("d", "another string")
        ->RenderUint64("pi", 22)
        ->RenderInt64("ni", -127)
        ->RenderDouble("pd", 45.3)
        ->RenderDouble("nd", -1056.4)
        ->RenderUint64("pl", 11779497823553162765ULL)
        ->StartList("l")
        ->StartList("")
        ->EndList()
        ->EndList()
        ->StartObject("o")
        ->RenderBool("key", true)
        ->EndObject()
        ->EndObject();
    DoTest(str, i);
  }
}


TEST_F(JsonStreamParserTest, RejectNonUtf8WhenNotCoerced) {
  StringPiece json = "{\"address\":\xFF\"חרושת 23, רעננה, ישראל\"}";
  for (int i = 0; i <= json.length(); ++i) {
    DoErrorTest(json, i, "Encountered non UTF-8 code points.");
  }
  json = "{\"address\": \"חרושת 23,\xFFרעננה, ישראל\"}";
  for (int i = 0; i <= json.length(); ++i) {
    DoErrorTest(json, i, "Encountered non UTF-8 code points.");
  }
}

#ifndef _MSC_VER
// - unicode handling in strings
TEST_F(JsonStreamParserTest, UnicodeEscaping) {
  StringPiece str = "[\"\\u0639\\u0631\\u0628\\u0649\"]";
  for (int i = 0; i <= str.length(); ++i) {
    // TODO(xiaofeng): Figure out what default encoding to use for JSON strings.
    // In protobuf we use UTF-8 for strings, but for JSON we probably should
    // allow different encodings?
    ow_.StartList("")->RenderString("", "\u0639\u0631\u0628\u0649")->EndList();
    DoTest(str, i);
  }
}
#endif

// - ascii escaping (\b, \f, \n, \r, \t, \v)
TEST_F(JsonStreamParserTest, AsciiEscaping) {
  StringPiece str =
      "[\"\\b\", \"\\ning\", \"test\\f\", \"\\r\\t\", \"test\\\\\\ving\"]";
  for (int i = 0; i <= str.length(); ++i) {
    ow_.StartList("")
        ->RenderString("", "\b")
        ->RenderString("", "\ning")
        ->RenderString("", "test\f")
        ->RenderString("", "\r\t")
        ->RenderString("", "test\\\ving")
        ->EndList();
    DoTest(str, i);
  }
}

// - trailing commas, we support a single trailing comma but no internal commas.
TEST_F(JsonStreamParserTest, TrailingCommas) {
  StringPiece str = "[['a',true,], {b: null,},]";
  for (int i = 0; i <= str.length(); ++i) {
    ow_.StartList("")
        ->StartList("")
        ->RenderString("", "a")
        ->RenderBool("", true)
        ->EndList()
        ->StartObject("")
        ->RenderNull("b")
        ->EndObject()
        ->EndList();
    DoTest(str, i);
  }
}

// Negative tests

// illegal literals
TEST_F(JsonStreamParserTest, ExtraTextAfterTrue) {
  StringPiece str = "truee";
  for (int i = 0; i <= str.length(); ++i) {
    ow_.RenderBool("", true);
    DoErrorTest(str, i, "Parsing terminated before end of input.");
  }
}

TEST_F(JsonStreamParserTest, InvalidNumberDashOnly) {
  StringPiece str = "-";
  for (int i = 0; i <= str.length(); ++i) {
    DoErrorTest(str, i, "Unable to parse number.");
  }
}

TEST_F(JsonStreamParserTest, InvalidNumberDashName) {
  StringPiece str = "-foo";
  for (int i = 0; i <= str.length(); ++i) {
    DoErrorTest(str, i, "Unable to parse number.");
  }
}

TEST_F(JsonStreamParserTest, InvalidLiteralInArray) {
  StringPiece str = "[nule]";
  for (int i = 0; i <= str.length(); ++i) {
    ow_.StartList("");
    DoErrorTest(str, i, "Unexpected token.");
  }
}

TEST_F(JsonStreamParserTest, InvalidLiteralInObject) {
  StringPiece str = "{123false}";
  for (int i = 0; i <= str.length(); ++i) {
    ow_.StartObject("");
    DoErrorTest(str, i, "Expected an object key or }.");
  }
}

// mismatched quotes failure on strings
TEST_F(JsonStreamParserTest, MismatchedSingleQuotedLiteral) {
  StringPiece str = "'Some str\"";
  for (int i = 0; i <= str.length(); ++i) {
    DoErrorTest(str, i, "Closing quote expected in string.");
  }
}

TEST_F(JsonStreamParserTest, MismatchedDoubleQuotedLiteral) {
  StringPiece str = "\"Another string that ends poorly!'";
  for (int i = 0; i <= str.length(); ++i) {
    DoErrorTest(str, i, "Closing quote expected in string.");
  }
}

// unterminated strings
TEST_F(JsonStreamParserTest, UnterminatedLiteralString) {
  StringPiece str = "\"Forgot the rest of i";
  for (int i = 0; i <= str.length(); ++i) {
    DoErrorTest(str, i, "Closing quote expected in string.");
  }
}

TEST_F(JsonStreamParserTest, UnterminatedStringEscape) {
  StringPiece str = "\"Forgot the rest of \\";
  for (int i = 0; i <= str.length(); ++i) {
    DoErrorTest(str, i, "Closing quote expected in string.");
  }
}

TEST_F(JsonStreamParserTest, UnterminatedStringInArray) {
  StringPiece str = "[\"Forgot to close the string]";
  for (int i = 0; i <= str.length(); ++i) {
    ow_.StartList("");
    DoErrorTest(str, i, "Closing quote expected in string.");
  }
}

TEST_F(JsonStreamParserTest, UnterminatedStringInObject) {
  StringPiece str = "{f: \"Forgot to close the string}";
  for (int i = 0; i <= str.length(); ++i) {
    ow_.StartObject("");
    DoErrorTest(str, i, "Closing quote expected in string.");
  }
}

TEST_F(JsonStreamParserTest, UnterminatedObject) {
  StringPiece str = "{";
  for (int i = 0; i <= str.length(); ++i) {
    ow_.StartObject("");
    DoErrorTest(str, i, "Unexpected end of string.");
  }
}


// mismatched object and array closing
TEST_F(JsonStreamParserTest, MismatchedCloseObject) {
  StringPiece str = "{'key': true]";
  for (int i = 0; i <= str.length(); ++i) {
    ow_.StartObject("")->RenderBool("key", true);
    DoErrorTest(str, i, "Expected , or } after key:value pair.");
  }
}

TEST_F(JsonStreamParserTest, MismatchedCloseArray) {
  StringPiece str = "[true, null}";
  for (int i = 0; i <= str.length(); ++i) {
    ow_.StartList("")->RenderBool("", true)->RenderNull("");
    DoErrorTest(str, i, "Expected , or ] after array value.");
  }
}

// Invalid object keys.
TEST_F(JsonStreamParserTest, InvalidNumericObjectKey) {
  StringPiece str = "{42: true}";
  for (int i = 0; i <= str.length(); ++i) {
    ow_.StartObject("");
    DoErrorTest(str, i, "Expected an object key or }.");
  }
}

TEST_F(JsonStreamParserTest, InvalidLiteralObjectInObject) {
  StringPiece str = "{{bob: true}}";
  for (int i = 0; i <= str.length(); ++i) {
    ow_.StartObject("");
    DoErrorTest(str, i, "Expected an object key or }.");
  }
}

TEST_F(JsonStreamParserTest, InvalidLiteralArrayInObject) {
  StringPiece str = "{[null]}";
  for (int i = 0; i <= str.length(); ++i) {
    ow_.StartObject("");
    DoErrorTest(str, i, "Expected an object key or }.");
  }
}

TEST_F(JsonStreamParserTest, InvalidLiteralValueInObject) {
  StringPiece str = "{false}";
  for (int i = 0; i <= str.length(); ++i) {
    ow_.StartObject("");
    DoErrorTest(str, i, "Expected an object key or }.");
  }
}

TEST_F(JsonStreamParserTest, MissingColonAfterStringInObject) {
  StringPiece str = "{\"key\"}";
  for (int i = 0; i <= str.length(); ++i) {
    ow_.StartObject("");
    DoErrorTest(str, i, "Expected : between key:value pair.");
  }
}

TEST_F(JsonStreamParserTest, MissingColonAfterKeyInObject) {
  StringPiece str = "{key}";
  for (int i = 0; i <= str.length(); ++i) {
    ow_.StartObject("");
    DoErrorTest(str, i, "Expected : between key:value pair.");
  }
}

TEST_F(JsonStreamParserTest, EndOfTextAfterKeyInObject) {
  StringPiece str = "{key";
  for (int i = 0; i <= str.length(); ++i) {
    ow_.StartObject("");
    DoErrorTest(str, i, "Unexpected end of string.");
  }
}

TEST_F(JsonStreamParserTest, MissingValueAfterColonInObject) {
  StringPiece str = "{key:}";
  for (int i = 0; i <= str.length(); ++i) {
    ow_.StartObject("");
    DoErrorTest(str, i, "Unexpected token.");
  }
}

TEST_F(JsonStreamParserTest, MissingCommaBetweenObjectEntries) {
  StringPiece str = "{key:20 'hello': true}";
  for (int i = 0; i <= str.length(); ++i) {
    ow_.StartObject("")->RenderUint64("key", 20);
    DoErrorTest(str, i, "Expected , or } after key:value pair.");
  }
}

TEST_F(JsonStreamParserTest, InvalidLiteralAsObjectKey) {
  StringPiece str = "{false: 20}";
  for (int i = 0; i <= str.length(); ++i) {
    ow_.StartObject("");
    DoErrorTest(str, i, "Expected an object key or }.");
  }
}

TEST_F(JsonStreamParserTest, ExtraCharactersAfterObject) {
  StringPiece str = "{}}";
  for (int i = 0; i <= str.length(); ++i) {
    ow_.StartObject("")->EndObject();
    DoErrorTest(str, i, "Parsing terminated before end of input.");
  }
}

// numbers too large
TEST_F(JsonStreamParserTest, PositiveNumberTooBig) {
  StringPiece str = "[18446744073709551616]";  // 2^64
  for (int i = 0; i <= str.length(); ++i) {
    ow_.StartList("");
    DoErrorTest(str, i, "Unable to parse number.");
  }
}

TEST_F(JsonStreamParserTest, NegativeNumberTooBig) {
  StringPiece str = "[-18446744073709551616]";
  for (int i = 0; i <= str.length(); ++i) {
    ow_.StartList("");
    DoErrorTest(str, i, "Unable to parse number.");
  }
}

/*
TODO(sven): Fail parsing when parsing a double that is too large.

TEST_F(JsonStreamParserTest, DoubleTooBig) {
  StringPiece str = "[184464073709551232321616.45]";
  for (int i = 0; i <= str.length(); ++i) {
    ow_.StartList("");
    DoErrorTest(str, i, "Unable to parse number");
  }
}
*/

// invalid unicode sequence.
TEST_F(JsonStreamParserTest, UnicodeEscapeCutOff) {
  StringPiece str = "\"\\u12";
  for (int i = 0; i <= str.length(); ++i) {
    DoErrorTest(str, i, "Illegal hex string.");
  }
}

TEST_F(JsonStreamParserTest, UnicodeEscapeInvalidCharacters) {
  StringPiece str = "\"\\u12$4hello";
  for (int i = 0; i <= str.length(); ++i) {
    DoErrorTest(str, i, "Invalid escape sequence.");
  }
}

// Extra commas with an object or array.
TEST_F(JsonStreamParserTest, ExtraCommaInObject) {
  StringPiece str = "{'k1': true,,'k2': false}";
  for (int i = 0; i <= str.length(); ++i) {
    ow_.StartObject("")->RenderBool("k1", true);
    DoErrorTest(str, i, "Expected an object key or }.");
  }
}

TEST_F(JsonStreamParserTest, ExtraCommaInArray) {
  StringPiece str = "[true,,false}";
  for (int i = 0; i <= str.length(); ++i) {
    ow_.StartList("")->RenderBool("", true);
    DoErrorTest(str, i, "Unexpected token.");
  }
}

// Extra text beyond end of value.
TEST_F(JsonStreamParserTest, ExtraTextAfterLiteral) {
  StringPiece str = "'hello', 'world'";
  for (int i = 0; i <= str.length(); ++i) {
    ow_.RenderString("", "hello");
    DoErrorTest(str, i, "Parsing terminated before end of input.");
  }
}

TEST_F(JsonStreamParserTest, ExtraTextAfterObject) {
  StringPiece str = "{'key': true} 'oops'";
  for (int i = 0; i <= str.length(); ++i) {
    ow_.StartObject("")->RenderBool("key", true)->EndObject();
    DoErrorTest(str, i, "Parsing terminated before end of input.");
  }
}

TEST_F(JsonStreamParserTest, ExtraTextAfterArray) {
  StringPiece str = "[null] 'oops'";
  for (int i = 0; i <= str.length(); ++i) {
    ow_.StartList("")->RenderNull("")->EndList();
    DoErrorTest(str, i, "Parsing terminated before end of input.");
  }
}

// Random unknown text in the value.
TEST_F(JsonStreamParserTest, UnknownCharactersAsValue) {
  StringPiece str = "*&#25";
  for (int i = 0; i <= str.length(); ++i) {
    DoErrorTest(str, i, "Expected a value.");
  }
}

TEST_F(JsonStreamParserTest, UnknownCharactersInArray) {
  StringPiece str = "[*&#25]";
  for (int i = 0; i <= str.length(); ++i) {
    ow_.StartList("");
    DoErrorTest(str, i, "Expected a value or ] within an array.");
  }
}

TEST_F(JsonStreamParserTest, UnknownCharactersInObject) {
  StringPiece str = "{'key': *&#25}";
  for (int i = 0; i <= str.length(); ++i) {
    ow_.StartObject("");
    DoErrorTest(str, i, "Expected a value.");
  }
}

}  // namespace converter
}  // namespace util
}  // namespace protobuf
}  // namespace google
