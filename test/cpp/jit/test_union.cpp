#include <gtest/gtest.h>

#include <test/cpp/jit/test_utils.h>
#include "torch/csrc/jit/ir/ir.h"
#include "aten/src/ATen/core/type.cpp"

namespace torch {
namespace jit {

/***


███╗   ██╗ ██████╗ ████████╗███████╗██╗
████╗  ██║██╔═══██╗╚══██╔══╝██╔════╝██║
██╔██╗ ██║██║   ██║   ██║   █████╗  ██║
██║╚██╗██║██║   ██║   ██║   ██╔══╝  ╚═╝
██║ ╚████║╚██████╔╝   ██║   ███████╗██╗
╚═╝  ╚═══╝ ╚═════╝    ╚═╝   ╚══════╝╚═╝


I need to test some legit subtyping relationships with Union/Optional.
As you can see, I've created classes A, B1, B2, and C below.
Unfortunately, I need to define a TypeKind for each class. This fails
because all TypeKinds are defined in a macro in `jit_type_base.h`. It
would be terrible engineering practice to add some random test types to
types to our actual codebase, so I definitely don't want to do that! On
the other hand, I can't figure out a way to conditionally add TypeKinds.
`operator=` (which is one of the main things I need to test) often uses
TypeKinds as part of its testing logic, so I truly do need this to work;
I can't just make them all use `AnyType::Kind` or something like that.

Let's discuss potential solutions as part of the next round of code
reviews

***/



// Fixture to make test cases easier to read
class UnionTypeTest : public ::testing::Test {
 public:
    // None
    const TypePtr none = NoneType::get();

    // List[str]
    const TypePtr l1 = ListType::ofStrings();

    // Optional[int]
    const TypePtr opt1 = OptionalType::create(IntType::get());

    // Optional[float]
    const TypePtr opt2 = OptionalType::create(FloatType::get());

    // Optional[List[str]]
    const TypePtr opt3 = OptionalType::create(ListType::ofStrings());

    // Tuple[Optional[int], int]
    const TypePtr tup1 = TupleType::create({OptionalType::create(IntType::get()), IntType::get()});

    // Tuple[int, int]
    const TypePtr tup2 = TupleType::create({IntType::get(), IntType::get()});

    //// AType
    //const TypePtr a = AType::get();

    //// B1Type
    //const TypePtr b1 = B1Type::get();

    //// B2Type
    //const TypePtr b2 = B2Type::get();

    //// CType
    //const TypePtr c = CType::get();

    bool hasType(UnionTypePtr u, TypePtr t) {
        auto res = std::find(u->getTypes().begin(),
                             u->getTypes().end(),
                             t);
        return res != u->getTypes().end();
    }

    /***
    *         -- TYPE HIERARCHY --
    *
    *                  ┏━━━┓
    *            ╭─────┨ A ┠─────╮
    *            │     ┗━━━┛     │
    *          ┏━┷━━┓         ┏━━┷━┓
    *        ╭─┨ B1 ┃         ┃ B2 ┃
    *        │ ┗━━━━┛         ┗━━━━┛
    *      ┏━┷━┓
    *      ┃ C ┃
    *      ┗━━━┛
    *
    ***/
    //struct AType;
    //using ATypePtr = std::shared_ptr<AType>;
    //struct AType : public Type {
    //    public:
    //        bool operator==(const Type& rhs) const {
    //            return rhs.kind() == kind();
    //        }

    //        std::string str() const override {
    //            return "for testing purposes only";
    //        }

    //        static ATypePtr get() {
    //            static ATypePtr value(new AType());
    //            return value;
    //        }

    //    private:
    //        AType() : Type(TypeKind::AType) {}
    //};

    //struct B1Type;
    //using B1TypePtr = std::shared_ptr<B1Type>;
    //struct B1Type : public AType {
    //    public:
    //        static B1TypePtr get() {
    //            static B1TypePtr value(new B1Type());
    //            return value;
    //        }

    //    private:
    //        B1Type() : Type(TypeKind::B1Type) {}
    //};

    //struct B2Type;
    //using B2TypePtr = std::shared_ptr<B2Type>;
    //struct B2Type : public AType {
    //    public:
    //        static B2TypePtr get() {
    //            static B2TypePtr value(new B2Type());
    //            return value;
    //        }

    //    private:
    //        B2Type() : Type(TypeKind::B2Type) {}
    //};

    //struct CType;
    //using CTypePtr = std::shared_ptr<CType>;
    //struct CType : public B1Type {

    //    public:
    //        static CTypePtr get() {
    //            static ATypePtr value(new CType());
    //            return value;
    //        }

    //    private:
    //        CType() : Type(TypeKind::CType) {}
    //};
};

TEST_F(UnionTypeTest, UnionCreate_OptionalT1AndOptionalT2) {
    // Goal: Union[int, float, None]
    const UnionTypePtr u = UnionType::create({opt1, opt2});

    ASSERT_EQ(u->getTypes().size(), 3);
    ASSERT_TRUE(UnionTypeTest::hasType(u, IntType::get()));
    ASSERT_TRUE(UnionTypeTest::hasType(u, FloatType::get()));
    ASSERT_TRUE(UnionTypeTest::hasType(u, NoneType::get()));
}

TEST_F(UnionTypeTest, UnionCreate_OptionalTAndT) {
    // Goal: Union[int, None]
    const UnionTypePtr u = UnionType::create({opt1, IntType::get()});

    ASSERT_EQ(u->getTypes().size(), 2);
    ASSERT_TRUE(UnionTypeTest::hasType(u, IntType::get()));
    ASSERT_TRUE(UnionTypeTest::hasType(u, NoneType::get()));
}

TEST_F(UnionTypeTest, UnionCreate_TupleWithSubtypingRelationship) {
    // Goal: Union[Tuple[Optional[int], int], str]
    const UnionTypePtr u = UnionType::create({StringType::get(), tup1, tup2});

    ASSERT_EQ(u->getTypes().size(), 2);
    ASSERT_TRUE(UnionTypeTest::hasType(u, StringType::get()));
    ASSERT_TRUE(UnionTypeTest::hasType(u, tup1));
}

TEST_F(UnionTypeTest, UnionCreate_ContainerTAndT) {
    // Goal: Union[List[str], str]
    const UnionTypePtr u = UnionType::create({l1, StringType::get()});

    ASSERT_EQ(u->getTypes().size(), 2);
    ASSERT_TRUE(UnionTypeTest::hasType(u, StringType::get()));
    ASSERT_TRUE(UnionTypeTest::hasType(u, ListType::ofStrings()));
}

TEST_F(UnionTypeTest, UnionCreate_OptionalContainerTAndContainerTAndT) {
    // Goal: Union[List[str], None, str]
    const UnionTypePtr u = UnionType::create({l1, opt3, StringType::get()});

    ASSERT_EQ(u->getTypes().size(), 3);
    ASSERT_TRUE(UnionTypeTest::hasType(u, StringType::get()));
    ASSERT_TRUE(UnionTypeTest::hasType(u, ListType::ofStrings()));
}

//TEST_F(UnionTypeTest, UnionCreate_FullTypeHierarchy) {
//    // Goal: Union[AType, str]
//    const TypePtr u1 = UnionType::create({a, b1, c, StringType::get()});

//    // Goal: Union[B1Type, B2Type, str]
//    const TypePtr u2 = UnionType::create({b1, b2, StringType::get()});

//    // Goal: Union[AType, str]
//    const TypePtr u3 = UnionType::create({b1, b2, a, StringType::get()});

//    ASSERT_EQ(u1->getTypes().size(), 2);
//    ASSERT_TRUE(UnionTypeTest::hasType(u1, StringType::get()));
//    ASSERT_TRUE(UnionTypeTest::hasType(u1, AType::get()));

//    ASSERT_EQ(u2->getTypes().size(), 3);
//    ASSERT_TRUE(UnionTypeTest::hasType(u2, StringType::get()));
//    ASSERT_TRUE(UnionTypeTest::hasType(u2, B1ype::get()));
//    ASSERT_TRUE(UnionTypeTest::hasType(u2, B2Type::get()));

//    ASSERT_EQ(u3->getTypes().size(), 2);
//    ASSERT_TRUE(UnionTypeTest::hasType(u3, StringType::get()));
//    ASSERT_TRUE(UnionTypeTest::hasType(u3, AType::get()));
//}

TEST_F(UnionTypeTest, SubtypingWithOptional_Basic) {
    // Union[int, None]
    const UnionTypePtr union1 = UnionType::create({IntType::get(), NoneType::get()});

    // Union[int, str, None]
    const UnionTypePtr union2 = UnionType::create({IntType::get(), StringType::get(), NoneType::get()});

    // Union[int, str, List[str]]
    const UnionTypePtr union3 = UnionType::create({IntType::get(), StringType::get(), ListType::ofStrings()});

    ASSERT_TRUE(none->isSubtypeOf(opt1));
    ASSERT_TRUE(none->isSubtypeOf(union1));
    ASSERT_TRUE(none->isSubtypeOf(union2));
    ASSERT_FALSE(none->isSubtypeOf(union3));

    ASSERT_FALSE(opt1->isSubtypeOf(none));
    ASSERT_TRUE(opt1->isSubtypeOf(union1));
    ASSERT_TRUE(opt1->isSubtypeOf(union2));
    //ASSERT_FALSE(opt1->isSubtypeOf(union3));

    //ASSERT_TRUE(opt1 == union1);

    //ASSERT_FALSE(union1->isSubtypeOf(none));
    //ASSERT_TRUE(union1->isSubtypeOf(opt1));
    //ASSERT_TRUE(union1->isSubtypeOf(union2));
    //ASSERT_FALSE(union1->isSubtypeOf(union3));

    //ASSERT_FALSE(union2->isSubtypeOf(union1));
}

//TEST_F(UnionTypeTest, SubtypingWithOptional_FullTypeHierarchy) {
//    // Union[B1, B2, None]
//    const TypePtr u1 = UnionType::create({b1, b2, NoneType::get()});

//    // Union[B1, None]
//    const TypePtr u2 = UnionType::create({b1, NoneType::get()});

//    // Optional[A]
//    const TypePtr opt_local = OptionalType::create(a);

//    ASSERT_TRUE(opt_local->isSubtypeOf(u1));
//    ASSERT_FALSE(u1->isSubtypeOf(opt_local));

//    ASSERT_TRUE(opt_local->isSubtypeOf(u2));
//    ASSERT_TRUE(u2->isSubtypeOf(opt_local));

//    ASSERT_TRUE(u2->isSubtypeOf(u1));
//    ASSERT_FALSE(u1->isSubtypeOf(u2));
//}

} // namespace torch
} // namespace jit
