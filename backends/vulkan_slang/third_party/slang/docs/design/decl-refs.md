Understanding Declaration References (Out of Date)
====================================

This document is intended as a reference for developers working on the Slang compiler implementation.

As you work on the code, you'll probably notice a lot of places where we use the `DeclRef<T>` type:

* Expressions like `VarExpr` and `MemberExpr` are subclasses of `DeclRefExpr`, which holds a `DeclRef<Decl>`.

* The most common subclass of `Type` is `DeclRefType`, which holds a `DeclRef<Decl>` for the type declaration.

* Named types (references to `typedef`s) hold a `DeclRef<TypedefDecl>`

* The name lookup process relies a lot on `DeclRef<ContainerDecl>`

So what in the world is a `DeclRef`?

The short answer is that a `DeclRef` packages up two things:

1. A pointer to a `Decl` in the parsed program AST

2. A set of "substitutions" to be applied to that decl

Why do we need `DeclRef`s?
--------------------------

In a compiler for a simple language, we might represent a reference to a declaration as simply a pointer to the AST node for the declaration, or some kind of handle/ID that references that AST node.
A representation like that will work in simple cases, for example:

```hlsl
struct Cell { int value };

Cell a = { 3 };
int b = a.value + 4;
```

In this case, the expression node for `a.value` can directly reference the declaration of the field `Cell::value`, and from that we can conclude that the type of the field (and hence the expression) is `int`.

In contrast, things get more complicated as soon as we have a language with generics:

```hlsl
struct Cell<T> { T value; };

// ...

Cell<int> a = { 3 };
int b = a.value + 4;
```

In this case, if we try to have the expression `a.value` only reference `Cell::value`, then the best we can do is conclude that the field has type `T`.

In order to correctly type the `a.value` expression, we need enough additional context to know that it references `Cell<int>::value`, and from that to be able to conclude that a reference to `T` in that context is equivalent to `int`.

We can represent that information as a substitution which maps `T` to `int`:

```
[ Cell::T => int ]
```

Then we can encode a reference to `Cell<int>::value` as a reference to the single declaration `Cell::value` with such a substitution applied:

```
Cell::value [Cell::T => int]
```

If we then want to query the type of this field, we can first look up the type stored on the AST (which will be a reference to `Cell::T`) and apply the substitutions from our field reference to get:

```
Cell::T [Cell::T => int]
```

Of course, we can then simplify the reference by applying the substitutions, to get:

```
int
```

How is this implemented?
------------------------

At the highest level, a `DeclRef` consists of a pointer to a declaration (a `Decl*`) plus a single-linked list of `Substution`s.
These substitutions fill in the missing information for any declarations on the ancestor chain for the declaration.

Each ancestor of a declaration can introduce an expected substitution along the chain:

* Most declarations don't introduce any substitutions: e.g., when referencing a non-generic `struct` we don't need any addition information.

* A surrounding generic declaration requires a `GenericSubstitution` which specifies the type argument to be plugged in for each type parameter of the declaration.

* A surrounding `interface` declaration usually requires a `ThisTypeSubstitution` that identifies the specific type on which an interface member has been looked up.

All of the expected substitutions should be in place in the general case, even when we might not have additional information. E.g., within a generic declaration like this:

```hlsl
struct Cell<T>
{
	void a();
	void b() { a(); }
}
```

The reference to `a` in the body of `b` will be represented as a declaration reference to `Cell::a` with a substitution that maps `[Cell::T => Cell::T]`. This might seem superfluous, but it makes it clear that we are "applying" the generic to arguments (even if they are in some sense placeholder arguments), and not trying to refer to an unspecialized generic.

There are a few places in the compiler where we might currently bend these rules, but experience has shown that failing to include appropriate substitutions is more often than not a source of bugs.

What in the world is a "this type" substitution?
------------------------------------------------

When using interface-constrained generics, we need a way to invoke methods of the interface on instances of a generic parameter type.
For example, consider this code:

```hlsl
interface IVehicle
{
	associatedtype Driver;
	Driver getDriver();
}

void ticketDriver<V : IVehicle>(V vehicle)
{
	V.Driver driver = vehicle.getDriver();
	sentTicketTo(driver);
}
```

In the expression `vehicle.getDriver`, we are referencing the declaration of `IVehicle::getDriver`, and so a naive reading tells us that the return type of the call is `IVehicle.Driver`, but that is an associated type and not a concrete type. It is clear in context that the expression `vehicle.getDriver()` should result in a `V.Driver`.

The way the compiler encodes that is that we treat the expression `v.getDriver` as first "up-casting" the value `v` (of type `V`) to the interface `IVehicle`. We know this is valid because of the generic constraint `V : IVehicle`. The result of the up-cast operation is an expression with a type that references `IVehicle`, but with a substitution to track the fact that the underlying implementation type is `V`. This amounts to something like:

```
IVehicle [IVehicle.This => V]
```

where `IVehicle.This` is a way to refer to "the concrete type that is implementing `IVehicle`".

Looking up the `getDriver` method on this up-cast expression yields a reference to:

```
IVehicle::getDriver [IVehicle.This => V]
```

And extracting the return type of that method gives us a reference to the type:

```
IVehicle::Driver [IVehicle.This => V]
```

which turns out to be exactly what the front end produces when it evaluates the type reference `V.Driver`.

As this example shows, a "this type" substitution allows us to refer to interface members while retaining knowledge of the specific type on which those members were looked up, so that we can compute correct references to things like associated types.

What does any of this mean for me?
----------------------------------

When working in the Slang compiler code, try to be aware of whether you should be working with a plain `Decl*` or a full `DeclRef`.
There are many queries like "what is the return type of this function?" that typically only make sense if you are applying them to a `DeclRef`.

The `syntax.h` file defines helpers for most of the existing declaration AST nodes for querying properties that should represent substitutions (the type of a variable, the return type of a function, etc.).
If you are writing code that is working with a `DeclRef`, try to use these accessors and avoid being tempted to extract the bare declaration and start querying it.

Some things like `Modifier`s aren't (currently) affected by substitutions, so it can make sense to query them on a bare declaration instead of a `DeclRef`.

Conclusion
----------

Working with `DeclRef`s can be a bit obtuse at first, but they are the most elegant solution we've found to the problems that arise when dealing with generics and interfaces in the compiler front-end. Hopefully this document gives you enough context to see why they are important, and hints at how their representation in the compiler helps us implement some cases that would be tricky otherwise.
