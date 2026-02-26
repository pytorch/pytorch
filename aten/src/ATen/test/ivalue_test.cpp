#include <ATen/ATen.h>
#include <ATen/core/Dict.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/irange.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

// Snippets for checking assembly.
c10::IValue inspectTupleConstruction() {
  std::tuple<std::string, std::string> s = std::make_tuple(
      "abcdefghijklmnopqrstuvwxyz", "ABCDEFGHIJKLMNOPQRSTUVWXYZ");
  return c10::IValue(s);
}

namespace c10 {

TEST(IValueTest, Basic) {
  c10::List<int64_t> foo({3, 4, 5});
  ASSERT_EQ(foo.use_count(), 1);
  IValue bar{foo};
  ASSERT_EQ(foo.use_count(), 2);
  auto baz = bar;
  ASSERT_EQ(foo.use_count(), 3);
  auto foo2 = std::move(bar);
  ASSERT_EQ(foo.use_count(), 3);
  ASSERT_TRUE(foo2.isIntList());
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  ASSERT_TRUE(bar.isNone());
  foo2 = IValue(4.0);
  ASSERT_TRUE(foo2.isDouble());
  ASSERT_EQ(foo2.toDouble(), 4.0);
  ASSERT_EQ(foo.use_count(), 2);
  ASSERT_TRUE(baz.toIntVector() == std::vector<int64_t>({3, 4, 5}));
  ASSERT_TRUE(baz.toDimVector() == at::DimVector({3, 4, 5}));

  auto move_it = std::move(baz).toIntList();
  ASSERT_EQ(foo.use_count(), 2);
  // NOLINTNEXTLINE(bugprone-use-after-move)
  ASSERT_TRUE(baz.isNone());
  IValue i(4);
  ASSERT_TRUE(i.isInt());
  ASSERT_EQ(i.toInt(), 4);
  IValue dlist(c10::List<double>({3.5}));
  ASSERT_TRUE(dlist.isDoubleList());
  ASSERT_TRUE(dlist.toDoubleVector() == std::vector<double>({3.5}));
  std::move(dlist).toDoubleList();
  // NOLINTNEXTLINE(bugprone-use-after-move)
  ASSERT_TRUE(dlist.isNone());
  dlist = IValue(c10::List<double>({3.4}));
  ASSERT_TRUE(dlist.toDoubleVector() == std::vector<double>({3.4}));
  dlist = IValue(std::vector<double>({3.3, 3.2}));
  ASSERT_TRUE(dlist.toDoubleVector() == std::vector<double>({3.3, 3.2}));
  IValue blist(std::vector<bool>{true, false});
  ASSERT_TRUE(blist.isList());
  const auto blistRef = blist.toListRef();
  ASSERT_EQ(blistRef.size(), 2);
  ASSERT_TRUE(blistRef[0].toBool());
  ASSERT_FALSE(blistRef[1].toBool());
  IValue the_list(
      at::ivalue::Tuple::create({IValue(3.4), IValue(4), IValue(foo)}));
  ASSERT_EQ(foo.use_count(), 3);
  ASSERT_TRUE(the_list.isTuple());
  auto first = the_list.toTupleRef().elements()[1];
  ASSERT_EQ(first.toInt(), 4);
  // Make sure toTupleRef has test coverage too.
  first = the_list.toTupleRef().elements()[1];
  ASSERT_EQ(first.toInt(), 4);
  at::Tensor tv = at::rand({3, 4});
  IValue ten(tv);
  ASSERT_EQ(tv.use_count(), 2);
  auto ten2 = ten;
  ASSERT_EQ(tv.use_count(), 3);
  ASSERT_TRUE(ten2.toTensor().equal(ten.toTensor()));
  std::move(ten2).toTensor();
  ASSERT_EQ(tv.use_count(), 2);

  auto elem1 = c10::complex<double>(3, 4);
  auto elem2 = c10::complex<double>(3, -4);
  auto elem3 = c10::complex<double>(5, 0);
  c10::List<c10::complex<double>> foo1({elem1, elem2, elem3});
  ASSERT_EQ(foo1.use_count(), 1);
  IValue bar1{foo1};
  ASSERT_EQ(foo1.use_count(), 2);
  auto baz1 = bar1;
  ASSERT_EQ(foo1.use_count(), 3);
  auto foo12 = std::move(bar1);
  ASSERT_EQ(foo1.use_count(), 3);
  ASSERT_TRUE(foo12.isComplexDoubleList());
  ASSERT_EQ(foo12.toComplexDoubleList(), foo1);

  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  ASSERT_TRUE(bar1.isNone());
  auto foo3 = IValue(c10::complex<double>(3, 4));
  ASSERT_TRUE(foo3.isComplexDouble());
  ASSERT_EQ(foo3.toComplexDouble(), c10::complex<double>(3,4));

  ASSERT_TRUE(baz1.toComplexDoubleVector() == std::vector<c10::complex<double>>({elem1, elem2, elem3}));
  IValue complex_tuple(
      at::ivalue::Tuple::create({IValue(c10::complex<double>(3.4, 4.7)), IValue(foo1)}));
  ASSERT_TRUE(complex_tuple.isTuple());
  ASSERT_EQ(complex_tuple.toTupleRef().elements()[0].toComplexDouble(), c10::complex<double>(3.4, 4.7));
  ASSERT_EQ(complex_tuple.toTupleRef().elements()[1], foo1);
}

TEST(IValueTest, BasicStorage) {
  at::Storage emptyStorage;
  at::Storage nonemptyStorage(at::rand({3, 4}).storage());
  IValue ivEmpty(emptyStorage);
  IValue ivNonempty(nonemptyStorage);

  ASSERT_TRUE(ivEmpty.isStorage());
  ASSERT_TRUE(ivNonempty.isStorage());
  ASSERT_EQ(emptyStorage.unsafeGetStorageImpl(), ivEmpty.toStorage().unsafeGetStorageImpl());
  ASSERT_EQ(nonemptyStorage.unsafeGetStorageImpl(), ivNonempty.toStorage().unsafeGetStorageImpl());
}

TEST(IValueTest, ComplexDict) {
  typedef c10::complex<double> c_type;
  c10::Dict<c_type, c_type> m;
  auto num1 = c_type(2.3, -3.5);
  auto num2 = c_type(0, 5);
  m.insert(num1, 2 * num1);
  m.insert(num2, 2 * num2);
  IValue dict(std::move(m));
  auto m_ = dict.toGenericDict();
  ASSERT_EQ(m_.at(num1), 2 * num1);
  ASSERT_EQ(m_.at(num2), 2 * num2);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
static std::array<IValue, 16> makeSampleIValues() {
  return {
    IValue(),
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
    at::rand({3, 4}),
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
    at::rand({3, 4}).storage(),
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
    1.5,
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
    c10::complex<double>(2.5, -0.5),
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
    42,
    true,
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
    std::make_tuple(23, "hello"),
    "hello",
    c10::make_intrusive<caffe2::Blob>(),
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
    c10::List<int64_t>({1, 2, 3}),
    c10::Dict<std::string, std::string>(),
    c10::make_intrusive<ivalue::Future>(FloatType::get()),
    c10::Device(c10::DeviceType::CPU, 0),
    c10::Stream(c10::Stream::DEFAULT, c10::Device(c10::DeviceType::CPU, 0)),
    c10::make_intrusive<ivalue::Object>(c10::StrongTypePtr(nullptr, ClassType::create("class1", {})), 1),
  };
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
static std::array<IValue, 16> makeMoreSampleIValues() {
  return {
    IValue(),
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
    at::rand({3, 4}),
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
    at::rand({3, 4}).storage(),
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
    2.5,
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
    c10::complex<double>(2.7, -0.3),
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
    43,
    false,
    std::make_tuple(1, "goodbye"),
    "goodbye",
    c10::make_intrusive<caffe2::Blob>(),
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
    c10::List<int64_t>({4, 5, 6}),
    c10::Dict<std::string, std::string>(),
    c10::make_intrusive<ivalue::Future>(IntType::get()),
    c10::Device(c10::DeviceType::CUDA, 2),
    c10::Stream(c10::Stream::DEFAULT, c10::Device(c10::DeviceType::CUDA, 1)),
    c10::make_intrusive<ivalue::Object>(c10::StrongTypePtr(nullptr, ClassType::create("class2", {})), 2),
  };}

// IValue::operator== doesn't seem to work on Tensors.
#define EXPECT_IVALUE_EQ(a, b)                          \
  EXPECT_EQ((a).isTensor(), (b).isTensor());            \
  if ((a).isTensor()) {                                 \
    EXPECT_TRUE((a).toTensor().equal((b).toTensor()));  \
  } else {                                              \
    EXPECT_EQ((a), (b));                                \
  }

TEST(IValueTest, Swap) {
  // swap() has the following 3 cases: tensor, intrusive_ptr, or
  // neither. Exercise all pairs of the three.

  auto sampleInputs = makeSampleIValues();
  auto sampleTargets = makeMoreSampleIValues();
  for (const auto& input: sampleInputs) {
    for (const auto& target: sampleTargets) {
      IValue a(input);
      IValue b(target);
      EXPECT_IVALUE_EQ(a, input);
      EXPECT_IVALUE_EQ(b, target);
      a.swap(b);
      EXPECT_IVALUE_EQ(a, target);
      EXPECT_IVALUE_EQ(b, input);
    }
  }
}

TEST(IValueTest, CopyConstruct) {
  auto sampleInputs = makeSampleIValues();
  for (const IValue& v: sampleInputs) {
    IValue copy(v);
    EXPECT_IVALUE_EQ(copy, v);
  }
}

TEST(IValueTest, MoveConstruct) {
  auto sampleInputs = makeSampleIValues();
  for (const IValue& v: sampleInputs) {
    IValue source(v);
    IValue target(std::move(source));
    EXPECT_IVALUE_EQ(target, v);
    // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
    EXPECT_TRUE(source.isNone());
  }
}

TEST(IValueTest, CopyAssign) {
  auto sampleInputs = makeSampleIValues();
  auto sampleTargets = makeMoreSampleIValues();

  for (const IValue& input: sampleInputs) {
    for (const IValue& target: sampleTargets) {
      IValue copyTo(target);
      IValue copyFrom(input);
      copyTo = copyFrom;
      EXPECT_IVALUE_EQ(copyTo, input);
      EXPECT_IVALUE_EQ(copyFrom, input);
      EXPECT_IVALUE_EQ(copyTo, copyFrom);
    }
  }
}

TEST(IValueTest, MoveAssign) {
  auto sampleInputs = makeSampleIValues();
  auto sampleTargets = makeMoreSampleIValues();

  for (const IValue& input: sampleInputs) {
    for (const IValue& target: sampleTargets) {
      IValue moveTo(target);
      IValue moveFrom(input);
      moveTo = std::move(moveFrom);
      EXPECT_IVALUE_EQ(moveTo, input);
      // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
      EXPECT_TRUE(moveFrom.isNone());
    }
  }
}

TEST(IValueTest, Tuple) {
  std::tuple<int64_t, at::Tensor> t = std::make_tuple(123, at::randn({1}));
  auto iv = IValue(t);
  auto t_ = iv.to<std::tuple<int64_t, at::Tensor>>();
  ASSERT_EQ(std::get<0>(t_), 123);
  ASSERT_EQ(
      std::get<1>(t_).item().to<float>(), std::get<1>(t).item().to<float>());
}

TEST(IValueTest, unsafeRemoveAttr) {
  auto cu = std::make_shared<CompilationUnit>();
  auto cls = ClassType::create("foo.bar", cu);
  cls->addAttribute("attr1", TensorType::get());
  cls->addAttribute("attr2", TensorType::get());
  auto obj = c10::ivalue::Object::create(
      c10::StrongTypePtr(cu, cls), cls->numAttributes());
  obj->unsafeRemoveAttr("attr1");
  // attr1 is not removed in the type
  ASSERT_TRUE(cls->hasAttribute("attr1"));
  ASSERT_TRUE(cls->hasAttribute("attr2"));
  ASSERT_TRUE(obj->slots().size() == 1);
}

TEST(IValueTest, TuplePrint) {
  {
    IValue tp = std::make_tuple(3);

    std::stringstream ss;
    ss << tp;
    ASSERT_EQ(ss.str(), "(3,)");
  }

  {
    IValue tp = std::make_tuple(3, 3);
    std::stringstream ss;
    ss << tp;
    ASSERT_EQ(ss.str(), "(3, 3)");
  }
}

TEST(IValueTest, ComplexIValuePrint) {
  {
    IValue complex(c10::complex<double>(2, -3));
    std::stringstream ss;
    ss << complex;
    ASSERT_EQ(ss.str(), "2.-3.j");
  }

  {
    IValue complex(c10::complex<double>(2, 0));
    std::stringstream ss;
    ss << complex;
    ASSERT_EQ(ss.str(), "2.+0.j");
  }

  {
    IValue complex(c10::complex<double>(0, 3));
    std::stringstream ss;
    ss << complex;
    ASSERT_EQ(ss.str(), "0.+3.j");
  }
}

TEST(IValueTest, Complex) {
  auto c = c10::complex<double>(2, 3);
  auto c_ = c10::complex<double>(2, -3);
  IValue c1(c), c2(c_), c3{at::Scalar(c)};

  ASSERT_TRUE(c1.isComplexDouble());
  ASSERT_TRUE(c3.isComplexDouble());

  ASSERT_EQ(c, c1.toComplexDouble());
  ASSERT_FALSE(c1 == c2);
  ASSERT_TRUE(c1 == c3);

  ASSERT_TRUE(c1.isScalar());
  ASSERT_TRUE(c2.toScalar().equal(c_));
}

TEST(IValueTest, BasicFuture) {
  auto f1 = c10::make_intrusive<ivalue::Future>(IntType::get());
  ASSERT_FALSE(f1->completed());

  f1->markCompleted(IValue(42));
  ASSERT_TRUE(f1->completed());
  ASSERT_EQ(42, f1->value().toInt());
  IValue iv(f1);
  ASSERT_EQ(42, iv.toFuture()->value().toInt());
}

TEST(IValueTest, FutureCallbacks) {
  auto f2 = c10::make_intrusive<ivalue::Future>(IntType::get());
  int calledTimesA = 0;
  int calledTimesB = 0;
  f2->addCallback([&calledTimesA](ivalue::Future& f2) {
    ASSERT_TRUE(f2.completed());
    ASSERT_EQ(f2.value().toInt(), 43);
    ++calledTimesA;
  });
  f2->markCompleted(IValue(43));
  ASSERT_EQ(calledTimesA, 1);
  ASSERT_EQ(calledTimesB, 0);
  // Post-markCompleted()
  f2->addCallback([&calledTimesB](ivalue::Future& f2) {
    ASSERT_TRUE(f2.completed());
    ASSERT_EQ(f2.value().toInt(), 43);
    ++calledTimesB;
  });
  ASSERT_EQ(calledTimesA, 1);
  ASSERT_EQ(calledTimesB, 1);
  ASSERT_FALSE(f2->hasError());
}

TEST(IValueTest, FutureExceptions) {
  auto f3 = c10::make_intrusive<ivalue::Future>(IntType::get());
  int calledTimes = 0;
  f3->addCallback([&calledTimes](ivalue::Future& f3) {
    ASSERT_TRUE(f3.completed());
    try {
      (void)f3.value();
    } catch (const std::exception& e) {
      if (std::string(e.what()) == "My Error") {
        ++calledTimes;
      }
    }
  });
  ivalue::Future::FutureError err("My Error");
  f3->setError(std::make_exception_ptr(err));
  ASSERT_EQ(calledTimes, 1);
  ASSERT_TRUE(f3->hasError());
  ASSERT_EQ(f3->tryRetrieveErrorMessage(), std::string("My Error"));
}

TEST(IValueTest, FutureSetError) {
  auto f1 = c10::make_intrusive<ivalue::Future>(IntType::get());
  f1->setError(std::make_exception_ptr(std::runtime_error("foo")));
  try {
    f1->setError(std::make_exception_ptr(std::runtime_error("bar")));
    FAIL() << "Expected to throw";
  } catch (std::exception& e) {
    EXPECT_THAT(e.what(), ::testing::HasSubstr("Error already set"));
    EXPECT_THAT(e.what(), ::testing::HasSubstr("foo"));
    EXPECT_THAT(e.what(), ::testing::HasSubstr("bar"));
  }
}

TEST(IValueTest, ValueEquality) {
  EXPECT_EQ(IValue("asdf"), IValue("asdf"));
  EXPECT_NE(IValue("asdf"), IValue("ASDF"));
  EXPECT_NE(IValue("2"), IValue(2));
  EXPECT_EQ(IValue(1), IValue(1));

  // Check the equals() variant that returns an IValue
  auto res = IValue("asdf").equals("asdf");
  EXPECT_TRUE(res.isBool());
  EXPECT_TRUE(res.toBool());

  res = IValue("asdf").equals(1);
  EXPECT_TRUE(res.isBool());
  EXPECT_FALSE(res.toBool());
}

TEST(IValueTest, TensorEquality) {
  auto rawTensor = torch::zeros({2, 3});
  auto rawTensorCopy = rawTensor.clone();
  auto t = IValue(rawTensor);
  auto tCopy = IValue(rawTensorCopy);

  // This should throw, because elementwise equality is ambiguous for
  // multi-element Tensors.
  auto testEquality = []() {
    return IValue(torch::ones({2, 3})) == IValue(torch::rand({2, 3}));
  };
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_ANY_THROW(testEquality());

  // equals() should return a tensor of all `true`.
  IValue eqTensor = t.equals(tCopy);
  EXPECT_TRUE(eqTensor.isTensor());
  auto booleanTrue = torch::ones({2, 3}).to(torch::kBool);
  EXPECT_TRUE(eqTensor.toTensor().equal(booleanTrue));

  // Test identity checking
  EXPECT_TRUE(t.is(t));
  EXPECT_FALSE(t.is(tCopy));
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  IValue tReference = t;
  EXPECT_TRUE(t.is(tReference));
}

TEST(IValueTest, ListEquality) {
  IValue c1 = std::vector<int64_t>{0, 1, 2, 3};
  IValue c2 = std::vector<int64_t>{0, 1, 2, 3};
  IValue c3 = std::vector<int64_t>{0, 1, 2, 3, 4};
  EXPECT_EQ(c1, c1);
  EXPECT_EQ(c1, c2);
  EXPECT_FALSE(c1.is(c2));
  EXPECT_NE(c1, c3);
  EXPECT_NE(c2, c3);
}

TEST(IValueTest, DictEquality) {
  auto innerDict = c10::Dict<std::string, std::string>();
  innerDict.insert("foo", "bar");

  auto d1 = c10::Dict<std::string, c10::Dict<std::string, std::string>>();
  d1.insert("one", innerDict);
  d1.insert("two", innerDict);
  d1.insert("three", innerDict);
  auto c1 = IValue(d1);

  auto d2 = c10::Dict<std::string, c10::Dict<std::string, std::string>>();
  d2.insert("one", innerDict.copy());
  d2.insert("two", innerDict.copy());
  d2.insert("three", innerDict.copy());
  auto c2 = IValue(d2);

  auto d3 = c10::Dict<std::string, c10::Dict<std::string, std::string>>();
  d3.insert("one", innerDict.copy());
  d3.insert("two", innerDict.copy());
  d3.insert("three", innerDict.copy());
  d3.insert("four", innerDict.copy());
  auto c3 = IValue(d3);

  auto d4 = c10::Dict<std::string, c10::Dict<std::string, std::string>>();
  d4.insert("one", innerDict.copy());
  d4.insert("two", innerDict.copy());
  auto innerDictNotEqual = c10::Dict<std::string, std::string>();
  innerDictNotEqual.insert("bar", "foo");
  d4.insert("three", innerDictNotEqual);
  auto c4 = IValue(d4);

  EXPECT_EQ(c1, c1);
  EXPECT_EQ(c1, c2);
  EXPECT_FALSE(c1.is(c2));
  EXPECT_NE(c1, c3);
  EXPECT_NE(c2, c3);
  EXPECT_NE(c1, c4);
  EXPECT_NE(c2, c4);
}

TEST(IValueTest, DictEqualityDifferentOrder) {
  auto d1 = c10::Dict<std::string, int64_t>();
  d1.insert("one", 1);
  d1.insert("two", 2);
  auto d2 = c10::Dict<std::string, int64_t>();
  d2.insert("two", 2);
  d2.insert("one", 1);

  EXPECT_EQ(d1, d2);
}

TEST(IValueTest, ListNestedEquality) {
  IValue c1 = std::vector<std::vector<int64_t>>({{0}, {0, 1}, {0, 1, 2}});
  IValue c2 = std::vector<std::vector<int64_t>>({{0}, {0, 1}, {0, 1, 2}});
  IValue c3 = std::vector<std::vector<int64_t>>({{1}, {0, 1}, {0, 1, 2}});
  EXPECT_EQ(c1, c1);
  EXPECT_EQ(c1, c2);
  EXPECT_NE(c1, c3);
  EXPECT_NE(c2, c3);
}

TEST(IValueTest, StreamEquality) {
  at::Device device1 =  at::Device(kCUDA, 0);
  at::Device device2 = at::Device(kCUDA, 1);
  c10::Stream stream1 = c10::Stream(c10::Stream::Default::DEFAULT, device1);
  c10::Stream stream2 = c10::Stream(c10::Stream::Default::DEFAULT, device2);
  IValue lhs(stream1);
  IValue rhs_different(stream2);
  IValue rhs_same(stream1);
  EXPECT_FALSE(lhs.equals(rhs_different).toBool());
  EXPECT_TRUE(lhs.equals(rhs_same).toBool());
}

TEST(IValueTest, EnumEquality) {
  auto cu = std::make_shared<CompilationUnit>();
  IValue int_ivalue_1(1);
  IValue int_ivalue_2(2);
  IValue str_ivalue_1("1");
  auto int_enum_type1 = EnumType::create(
      "enum_class_1",
      IntType::get(),
      {{"enum_name_1", int_ivalue_1}, {"enum_name_2", int_ivalue_2}},
      cu);
  auto int_enum_type2 = EnumType::create(
      "enum_class_2",
      IntType::get(),
      {{"enum_name_1", int_ivalue_1}, {"enum_name_2", int_ivalue_2}},
      cu);
  auto string_enum_type = EnumType::create(
      "enum_class_3", StringType::get(), {{"enum_name_1", str_ivalue_1}}, cu);

  EXPECT_EQ(
      IValue(c10::make_intrusive<ivalue::EnumHolder>(
          int_enum_type1, "enum_name_1", int_ivalue_1)),
      IValue(c10::make_intrusive<ivalue::EnumHolder>(
          int_enum_type1, "enum_name_1", int_ivalue_1))
  );

  EXPECT_NE(
      IValue(c10::make_intrusive<ivalue::EnumHolder>(
          int_enum_type1, "enum_name_1", int_ivalue_1)),
      IValue(c10::make_intrusive<ivalue::EnumHolder>(
          int_enum_type2, "enum_name_1", int_ivalue_1))
  );

  EXPECT_NE(
      IValue(c10::make_intrusive<ivalue::EnumHolder>(
          int_enum_type1, "enum_name_1", int_ivalue_1)),
      IValue(c10::make_intrusive<ivalue::EnumHolder>(
          int_enum_type1, "enum_name_2", int_ivalue_2))
  );

  EXPECT_NE(
      IValue(c10::make_intrusive<ivalue::EnumHolder>(
          int_enum_type1, "enum_name_1", int_ivalue_1)),
      IValue(c10::make_intrusive<ivalue::EnumHolder>(
          string_enum_type, "enum_name_1", str_ivalue_1))
  );
}

TEST(IValueTest, isPtrType) {
  IValue tensor(at::rand({3, 4}));
  IValue undefinedTensor((at::Tensor()));
  IValue integer(42);
  IValue str("hello");

  EXPECT_TRUE(tensor.isPtrType());
  EXPECT_FALSE(undefinedTensor.isPtrType());
  EXPECT_FALSE(integer.isPtrType());
  EXPECT_TRUE(str.isPtrType());
}

TEST(IValueTest, isAliasOf) {
  auto sampleIValues = makeSampleIValues();
  for (auto& iv: sampleIValues) {
    for (auto& iv2: sampleIValues) {
      if (&iv == &iv2 && iv.isPtrType()) {
        EXPECT_TRUE(iv.isAliasOf(iv2));
      } else {
        EXPECT_FALSE(iv.isAliasOf(iv2));
      }
    }
  }
}

TEST(IValueTest, toSymIntList) {
  std::vector<int64_t> int_list = {2, 3};
  auto iv = IValue(int_list);
  auto result = iv.toSymIntList();
  EXPECT_EQ(result.size(), 2);
  EXPECT_EQ(result.get(0), 2);
  EXPECT_EQ(result.get(1), 3);
}

TEST(IValueTest, toSymIntListTemplate) {
  std::vector<int64_t> int_list = {2, 3};
  auto iv = IValue(int_list);
  auto result = iv.to<c10::List<c10::SymInt>>();
  EXPECT_EQ(result.size(), 2);
  EXPECT_EQ(result.get(0), 2);
  EXPECT_EQ(result.get(1), 3);
}

TEST(IValueTest, toSymIntVector) {
  std::vector<int64_t> int_list = {2, 3};
  auto iv = IValue(int_list);
  auto result = iv.to<std::vector<c10::SymInt>>();
  EXPECT_EQ(result.size(), 2);
  EXPECT_EQ(result[0], 2);
  EXPECT_EQ(result[1], 3);
}

TEST(IValueTest, internalToPointer) {
  IValue tensor(at::rand({3, 4}));
  IValue str("hello");

  EXPECT_EQ(tensor.internalToPointer(), tensor.unsafeToTensorImpl());
  EXPECT_NE(str.internalToPointer(), nullptr);

  IValue nullStr((c10::intrusive_ptr<ivalue::ConstantString>()));
  ASSERT_TRUE(nullStr.isString());
  EXPECT_EQ(nullStr.internalToPointer(), nullptr);
}

TEST(IValueTest, IdentityComparisonAndHashing) {
  at::Tensor t1 = at::rand({3, 4});
  at::Tensor t2 = at::rand({3, 4});
  IValue tv1(t1), tv2(t2);
  IValue tv1b(t1);

  EXPECT_EQ(tv1.hash(), tv1b.hash());
  EXPECT_NE(tv1.hash(), tv2.hash());

  EXPECT_TRUE(tv1.is(tv1));
  EXPECT_TRUE(tv1.is(tv1b));
  EXPECT_TRUE(tv1b.is(tv1));
  EXPECT_TRUE(tv2.is(tv2));

  EXPECT_FALSE(tv1.is(tv2));
  EXPECT_FALSE(tv2.is(tv1));

  IValue none;
  IValue undefinedTensor((at::Tensor()));

  EXPECT_TRUE(none.is(undefinedTensor));
  EXPECT_TRUE(undefinedTensor.is(none));

  // Is this a bug? We should probably have a is b => a.hash() == b.hash()
  EXPECT_NE(none.hash(), undefinedTensor.hash());

  auto sampleIValues = makeSampleIValues();
  auto sampleIValues2 = makeSampleIValues();
  auto moreSampleIValues = makeMoreSampleIValues();

  ASSERT_EQ(sampleIValues.size(), moreSampleIValues.size());
  for (const auto ii : c10::irange(sampleIValues.size())) {
    if (sampleIValues[ii].isComplexDouble() ||
        sampleIValues[ii].isBlob() ||
        sampleIValues[ii].isList() ||
        sampleIValues[ii].isFuture() ||
        sampleIValues[ii].isStream() ||
        sampleIValues[ii].isObject() ||
        sampleIValues[ii].isGenericDict()) {
      // Not hashable.
      continue;
    }
    // Tuples may or may not have the same hash across instantiations.
    if (!sampleIValues[ii].isTuple()) {
      // Constant strings will have the same pointer value.
      if (sampleIValues[ii].isPtrType() && !sampleIValues[ii].isString()) {
        EXPECT_NE(sampleIValues[ii].hash(), sampleIValues2[ii].hash())
          << " at index " << ii;
      } else {
        EXPECT_EQ(sampleIValues[ii].hash(), sampleIValues2[ii].hash())
          << " at index " << ii;
      }
    }
    if (!sampleIValues[ii].isNone() && !moreSampleIValues[ii].isNone()) {
      EXPECT_NE(sampleIValues[ii].hash(), moreSampleIValues[ii].hash())
        << " at index " << ii;
    }
  }
}

// Sparse tensors do not work with static CPU dispatch
#ifndef ATEN_CPU_STATIC_DISPATCH
TEST(IValueTest, IdentityAndHashing_SparseCOO) {
  using namespace torch::indexing;

  at::Tensor t1 = at::rand({3, 4}).to_sparse();
  at::Tensor t2 = at::rand({3, 4}).to_sparse();
  at::Tensor t3 = at::rand({3, 4});

  IValue tv1(t1), tv1b(t1), tv2(t2), tv3(t3);

  EXPECT_EQ(tv1.hash(), tv1b.hash());
  EXPECT_NE(tv1.hash(), tv2.hash());

  EXPECT_TRUE(tv1.is(tv1b));
  EXPECT_FALSE(tv1.is(tv2));

  EXPECT_TRUE(tv1.isAliasOf(tv1b));
  EXPECT_FALSE(tv1.isAliasOf(tv2));
  EXPECT_FALSE(tv1.isAliasOf(tv3));

  std::vector<int64_t> idx_array1 = {0, 1, 1, 0, 0, 1};
  at::Tensor idx1 = torch::from_blob(
      idx_array1.data(),
      {2, 3},
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  std::vector<int64_t> idx_array2 = {1, 1, 2, 0, 1, 2};
  at::Tensor idx2 = torch::from_blob(
      idx_array2.data(),
      {2, 3},
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  std::vector<int32_t> val_array = {3, -5, 7};
  at::Tensor val = torch::from_blob(
      val_array.data(),
      {3},
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
  at::Tensor sparse1 = torch::sparse_coo_tensor(
      idx1, val, {3, 3}, torch::TensorOptions().dtype(torch::kInt32));
  at::Tensor sparse2 = torch::sparse_coo_tensor(
      idx2, val, {3, 3}, torch::TensorOptions().dtype(torch::kInt32));

  IValue idx1_v(idx1), idx2_v(idx2);
  IValue val_v(val);
  IValue sparse1_v(sparse1), sparse2_v(sparse2);

  EXPECT_TRUE(sparse1_v.isAliasOf(sparse2_v));
  EXPECT_TRUE(sparse1_v.isAliasOf(idx1_v));
  EXPECT_TRUE(sparse1_v.isAliasOf(val_v));
  EXPECT_TRUE(sparse2_v.isAliasOf(idx2_v));
  EXPECT_TRUE(sparse2_v.isAliasOf(val_v));
  EXPECT_FALSE(idx1_v.isAliasOf(idx2_v));
  EXPECT_FALSE(idx1_v.isAliasOf(val_v));
  EXPECT_FALSE(sparse1_v.isAliasOf(idx2_v));
}
#endif // ATEN_CPU_STATIC_DISPATCH

TEST(IValueTest, getSubValues) {
  // Scalars have no subvalues.
  IValue integer(42), float_(1.5), complex(c10::complex<double>(2, 3));

  IValue::HashAliasedIValues subvalues;

  integer.getSubValues(subvalues);
  EXPECT_TRUE(subvalues.empty());

  subvalues.clear();

  float_.getSubValues(subvalues);
  EXPECT_TRUE(subvalues.empty());

  subvalues.clear();

  complex.getSubValues(subvalues);
  EXPECT_TRUE(subvalues.empty());

  subvalues.clear();

  at::Tensor t1(at::rand({3, 4})), t2(at::rand({3, 4}));
  IValue tv1(t1), tv2(t2);
  IValue list(std::vector<at::Tensor>{t1, t2});
  IValue tuple(ivalue::Tuple::create({tv1, tv2}));

  c10::Dict<int64_t, at::Tensor> m;
  m.insert(1, t1);
  m.insert(2, t2);

  IValue dict(std::move(m));

  auto objType = ClassType::create(std::nullopt, {});
  objType->addAttribute("t1", tv1.type());
  objType->addAttribute("t2", tv2.type());

  auto o = ivalue::Object::create(StrongTypePtr(nullptr, objType), 2);
  o->setSlot(0, tv1);
  o->setSlot(1, tv2);

  IValue object(o);
  tv1.getSubValues(subvalues);
  EXPECT_EQ(subvalues.size(), 1);
  EXPECT_EQ(subvalues.count(tv1), 1);

  subvalues.clear();

  for (auto& container: {list, tuple, dict, object}) {
    container.getSubValues(subvalues);
    EXPECT_EQ(subvalues.size(), 3);
    EXPECT_EQ(subvalues.count(container), 1);
    EXPECT_EQ(subvalues.count(tv1), 1);
    EXPECT_EQ(subvalues.count(tv2), 1);

    subvalues.clear();
  }
}

TEST(IValueTest, ScalarBool) {
  Scalar expected(true);
  IValue v(expected);
  Scalar actual = v.toScalar();
  EXPECT_TRUE(actual.isBoolean());
  EXPECT_TRUE(actual.toBool());
}

TEST(IValueTest, ToWeakAndBack) {
  auto sampleInputs = makeSampleIValues();
  for (const auto& sample: sampleInputs) {
    WeakIValue weak(sample);
    EXPECT_IVALUE_EQ(sample, weak.lock());
  }
}

// Storage and Generator did not set is_intrusive_ptr if they were
// undefined, which led use_count to return 1 instead of 0 for these
// cases.
TEST(IValueTest, UseCountCornerCases) {
  at::Storage undefinedStorage;
  at::Generator undefinedGenerator;
  at::Tensor undefinedTensor;

  IValue ivEmptyStorage(undefinedStorage);
  IValue ivEmptyGenerator(undefinedGenerator);
  IValue ivEmptyTensor(undefinedTensor);

  ASSERT_EQ(1, ivEmptyStorage.use_count());
  ASSERT_EQ(1, ivEmptyGenerator.use_count());
  ASSERT_EQ(0, ivEmptyTensor.use_count());
}

// TODO(gmagogsfm): Add type conversion test?

using ivalue::TupleElements;

namespace {
void validateTupleElements(TupleElements& te, c10::ArrayRef<IValue> contents) {
  EXPECT_EQ(te.empty(), contents.empty());
  EXPECT_EQ(te.size(), contents.size());
  for (const auto idx: c10::irange(contents.size())) {
    EXPECT_IVALUE_EQ(te[idx], contents[idx]);
    EXPECT_IVALUE_EQ(te.at(idx), contents[idx]);
    EXPECT_IVALUE_EQ(*(te.begin() + idx), contents[idx]);
  }
  if (!contents.empty()) {
    EXPECT_IVALUE_EQ(te.back(), contents.back());
  }
  auto v = std::move(te).vec();
  EXPECT_EQ(v.size(), contents.size());
  for (const auto idx: c10::irange(contents.size())) {
    EXPECT_IVALUE_EQ(v[idx], contents[idx]);
  }
}
} // namespace

TEST(TupleElementsTest, Basic) {
  TupleElements empty;
  validateTupleElements(empty, {});
  TupleElements size1(1);
  validateTupleElements(size1, {1});
  TupleElements size2(1, 2);
  validateTupleElements(size2, {1, 2});
  TupleElements size3(1, 2, 3);
  validateTupleElements(size3, {1, 2, 3});

  auto sampleIValuesArray = makeSampleIValues();
  TupleElements large(std::vector<IValue>(sampleIValuesArray.begin(), sampleIValuesArray.end()));
  validateTupleElements(large, sampleIValuesArray);
}

namespace {

std::array<TupleElements(*)(), 3> factories = {
  []() { return TupleElements();},
  []() { return  TupleElements(1, 2, 3);},
  []() { return TupleElements(std::vector<IValue>({1, 2, 3, "hello"})); }
};

std::array<std::vector<IValue>, 3> expectedContents = {
  std::vector<IValue>(),
  std::vector<IValue>({1, 2, 3}),
  std::vector<IValue>({1, 2, 3, "hello"}),
};

}

TEST(TupleElementsTest, Resize) {
  std::array<std::vector<IValue>, 3> newContents = {std::vector<IValue>(), std::vector<IValue>({4, 5, 6}), std::vector<IValue>({7, 8, 9, "hello"})};

  for (auto factory : factories) {
    for (const auto& contents : newContents) {
      auto te = factory();
      auto contentsCopy = contents;
      te.setContents(std::move(contentsCopy));
      validateTupleElements(te, contents);
    }
  }
}

TEST(TupleElementsTest, CopyAndMoveConstruct) {
  int idx = 0;
  for (auto fromFactory : factories) {
    auto toMoveFrom = fromFactory();
    TupleElements movedInto(std::move(toMoveFrom));
    validateTupleElements(movedInto, expectedContents[idx]);
    auto toCopyFrom = fromFactory();
    TupleElements copiedInto(toCopyFrom);
    validateTupleElements(copiedInto, expectedContents[idx]);
    idx++;
  }
}

TEST(TupleElementsTest, CopyAndMoveAssign) {
  int fromIdx = 0;
  for (auto fromFactory : factories) {
    for (auto toFactory : factories) {
      auto from = fromFactory();
      auto to = toFactory();
      auto copyFrom = fromFactory();
      auto toCopy = toFactory();
      to = std::move(from);
      validateTupleElements(to, expectedContents[fromIdx]);
      toCopy = copyFrom;
      validateTupleElements(toCopy, expectedContents[fromIdx]);
    }
    fromIdx++;
  }
}

} // namespace c10
