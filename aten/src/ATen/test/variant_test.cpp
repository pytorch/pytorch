#include <gtest/gtest.h>

#include <c10/util/variant.h>

namespace testns {

namespace enumtype {
  struct EnumBase {
    virtual std::string name() const;
  };
  // NOTE: We need to provide the default constructor for each struct,
  // otherwise Clang 3.8 would complain:
  // ```
  // error: default initialization of an object of const type 'const enumtype::Enum1'
  // without a user-provided default constructor
  // ```
  struct Enum1 {
    Enum1() {}
    std::string name() const override {
      return "Enum1";
    }
  };
  struct Enum2 {
    Enum2() {}
    std::string name() const override {
      return "Enum2";
    }
  };
  struct Enum3 {
    Enum3() {}
    std::string name() const override {
      return "Enum3";
    }
  };
} // namespace enumtype

struct enum_name {
  constexpr std::string operator()(enumtype::EnumBase* v) const {
    return v->name();
  }
};

const enumtype::Enum1 kEnum1;
const enumtype::Enum2 kEnum2;
const enumtype::Enum3 kEnum3;

} // namespace testns

std::string func(c10::variant<testns::enumtype::Enum1, testns::enumtype::Enum2, testns::enumtype::Enum3> v) {
  if (c10::get_if<testns::enumtype::Enum1>(&v)) {
    return "Enum1";
  } else if (c10::get_if<testns::enumtype::Enum2>(&v)) {
    return "Enum2";
  } else if (c10::get_if<testns::enumtype::Enum3>(&v)) {
    return "Enum3";
  } else {
    return "Unsupported enum";
  }
}

TEST(VariantTest, Basic) {
  ASSERT_EQ(func(testns::kEnum1), "Enum1");
  ASSERT_EQ(func(testns::kEnum2), "Enum2");
  ASSERT_EQ(func(testns::kEnum3), "Enum3");

  c10::variant<testns::enumtype::Enum1, testns::enumtype::Enum2, testns::enumtype::Enum3> v;
  {
    v = testns::kEnum1;
    ASSERT_EQ(c10::visit(testns::enum_name{}, &v), "Enum1");
  }
  {
    v = testns::kEnum2;
    ASSERT_EQ(c10::visit(testns::enum_name{}, &v), "Enum2");
  }
  {
    v = testns::kEnum3;
    ASSERT_EQ(c10::visit(testns::enum_name{}, &v), "Enum3");
  }
}
