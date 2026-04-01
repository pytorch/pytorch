Existential Types
=================

This document attempts to provide some background on "existential types" as they pertain to the design and implementation of Slang.
The features described here are *not* reflected in the current implementation, so this is mostly a sketch of where we can go with the language and compiler.

Background: Generics and Universal Quantification
-------------------------------------------------

Currently Slang supports using interfaces as generic constraints. Let's use a contrived example:

```hlsl
interface IImage { float4 getValue(float2 uv); }

float4 offsetImage<T : IImage>(T image, float2 uv)
{
	float2 offset = ...;
	return image.getValue(uv + offset)
}
```

Generics like this are a form of "universal quantification" in the terminology of type theory.
This makes sense, because *for all* types `T` that satisfy the constraints, `offsetImage` provides an implementation of its functionality.

When we think of translating `offsetImage` to code, we might at first only think about how we can specialize it once we have a particular type `T` in mind.
However, we can also imagine trying to generate one body of code that can implement `offsetImage` for *any* type `T`, given some kind of runtime representation of types.
For example, we might generate C++ code like:

```c++
struct IImageWitnessTable { float4 (*getValue)(void* obj, float2 uv); };

float4 offsetImage(Type* T, IImageWitnessTable* W, void* image, float2 uv)
{
	float2 offset = ...;
	return W->getvalue(image, uv + offset);
}
```

This translation takes the generic parameters and turns them into ordinary runtime parameters: the type `T` becomes a pointer to a run-time type representation, while the constraint that `T : IImage` becomes a "witness table" of function pointers that, we assume, implements the `IImage` interface for `T`. So, the syntax of generics is *not* tied to static specialization, and can admit a purely runtime implementation as well.

Readers who are familiar with how languages like C++ are implemented might see the "witness table" above and realize that it is kind of like a virtual function table, just being passed alongside the object, rather than stored in its first word.

Using Interfaces Like Types
---------------------------

It is natural for a user to want to write code like the following:

```hlsl
float4 modulateImage(IImage image, float2 uv)
{
	float4 factor = ...;
	return factor * image.getValue(uv);
}
```

Unlike `offsetImage`, `modulateImage` is trying to use the `IImage` interface as a *type* and not just a constraint.

This code appears to be asking for a dynamic implementation rather than specialization (we'll get back to that...) and so we should be able to implement it similarly to our translation of `offsetImage` to C++.
Something like the following makes a lot of sense:

```c++
struct IImage { Type* T; IImageWitnessTable* W; void* obj; };

float4 modulateImage(IImage image, float2 uv)
{
	float4 factor = ...;
	return factor * image.W->getvalue(image.obj, uv);
}
```

Similar to the earlier example, there is a one-to-one mapping of the parameters of the Slang function the user wrote to the parameters of the generated C++ function.
To make this work, we had to bundle up the information that used to be separate parameters to the generic as a single value of type `IImage`.

Existential Types
-----------------

It turns out that when we use `IImage` as a type, it is what we'd call an *existential* type.
That is because if I give you a value `img` of type `IImage` in our C++ model, then you know that *there exists* some type `img.T`, a witness table `img.W` proving the type implements `IImage`, and a value `img.obj` of that type.

Existential types are the bread and butter of object-oriented programming.
If I give you an `ID3D11Texture2D*` you don't know what its concrete type is, and you just trust me that some concrete type *exists* and that it implements the interface.
A C++ class or COM component can implement an existential type, with the constraint that the interfaces that a given type can support is limited by the way that virtual function tables are intrusively included inside the memory of the object, rather than externalized.
Many modern languages (e.g., Go) support adapting existing types to new interfaces, so that a "pointer" of interface type is actually a fat pointer: one for the object, and one for the interface dispatch table.
Our examples so far have assumed that the type `T` needs to be passed around separately from the witness table `W`, but that isn't strictly required in some implementations.

In type theory, the most important operation you can do with an existential type is to "open" it, which means to have a limited scope in which you can refer to the constituent pieces of a "bundled up" value of a type like `IImage`.
We could imagine "opening" an existential as something like:

```
void doSomethingCool<T : IImage>(T val);

void myFunc(IImage img)
{
	open img as obj:T in
	{
		// In this scope we know that `T` is a type conforming to `IImage`,
		// and `obj` is a value of type `T`.
		//
		doSomethingCool<T>(obj);
	}
}
```

Self-Conformance
----------------

The above code with `doSomethingCool` and `myFunc` invites a much simpler solution:

```
void doSomethingCool<T : IImage>(T val);

void myFunc(IImage img)
{
	doSomethingCool(img);
}
```

This seems like an appealing thing for a language to support, but there are some subtle reasons why this isn't possible to support in general.
If we think about what `doSomethingCool(img)` is asking for, it seems to be trying to invoke the function `doSomethingCool<IImage>`.
That function only accepts type parameters that implement the `IImage` interface, so we have to ask ourselves:

Does the (existential) type `IImage` implement the `IImage` interface?

Knowing the implementation strategy outline above, we can re-phrase this question to: can we construct a witness table that implements the `IImage` interface for values of type `IImage`?

For simple interfaces this is sometimes possible, but in the general case there are other desirable language features that get in the way:

* When an interface has associated types, there is no type that can be chosen as the associated type for the interface's existential type. The "obvious" approach of using the constraints on the associated type can lead to unsound logic when interface methods take associated types as parameters.

* When an interface uses the "this type" (e.g., an `IComparable` interface with a `compareTo(ThisType other)` method), it isn't correct to simplify the this type to the interface type (just because you have two `IComarable` values doesn't mean you can compare them - they have to be of the same concrete type!)

* If we allow for `static` method on interfaces, then what implementation would we use for these methods on the interface's existential type?

Encoding Existentials in the IR
-------------------------------

Existentials are encoded in the Slang IR quite simply. We have an operation `makeExistential(T, obj, W)` that takes a type `T`, a value `obj` that must have type `T`, and a witness table `W` that shows how `T` conforms to some interface `I`. The result of the `makeExistential` operation is then a value of the type `I`.

Rather than include an IR operation to "open" an existential, we can instead just provide accessors for the pieces of information in an existential: one to extract the type field, one to extract the value, and one to extract the witness table. These would idiomatically be used like:

```
let e : ISomeInterface = /* some existential */
let T : Type = extractExistentialType(e);
let W : WitnessTbale = extractExistentialWitnessTable(e);
let obj : T = extractExistentialValue(e);
```

Note how the operation to extract `obj` gets its result type from the previously-executed extraction of the type.

Simplifying Code Using Existentials
-----------------------------------

It might seem like IR code generated using existentials can only be implemented using dynamic dispatch.
However, within a local scope it is clear that we can simplify expressions whenever `makeExistential` and `extractExistential*` operations are paired.
For example:

```
let e : ISomeInterface = makeExistential(A, a, X);
...
let B = extractExistentialType(e);
let b : B = extractExistentialValue(e);
let Y = extractExistentialWitnessTable(e);
```

It should be clear in context that we can replace `B` with `A`, `b` with `a`, and `Y` with `X`, after which all of the `extract*` operations and the `makeExistential` operation are dead and can be eliminated.

This kind of simplification works within a single function, as long as there is no conditional logic involving existentials.
We require further transformation passes to allow specialization in more general cases:

* Copy propagation, redundancy elimination and other dataflow optimizations are needed to simplify use of existentials within functions
* Type legalization passes, including some amount of scalarization, are needed to "expose" existential-type fields that are otherwise buried in a type
* Function specialization, is needed so that a function with existential parameters is specialized based on the actual types used at call sites

Transformations just like these are already required when working with resource types (textures/samplers) on targets that don't support first-class computation on resources, so it is possible to share some of the same logic.
Similarly, any effort we put into validation (to ensure that code is written in a way that *can* be simplified) can hopefully be shared between existentials and resources.

Compositions
------------

So far I've only talked about existential types based on a single interface, but if you look at the encoding as a tuple `(obj, T, W)` there is no real reason that can't be generalized to hold multiple witness tables: `(obj, T, W0, ... WN)`. Interface compositions could be expressed at the language level using the `&` operator on interface (or existential) types.

The IR encoding doesn't need to change much to support compositions: we just need to allow multiple witness tables on `makeExistential` and have an index operand on `extractExistentialWitnessTable` to get at the right one.

The hardest part of supporting composition of interfaces is actually in how to linearize the set of interfaces in a way that is stable, so that changing a function from using `IA & IB` to `IB & IA` doesn't change the order in which witness tables get packed into an existential value.

Why are we passing along the type?
----------------------------------

I'm glossing over something pretty significant here, which is why anybody would pass around the type as part of the existential value, when none of our examples so far have made use of it.
This sort of thing isn't very important for languages where interface polymorphism is limited to heap-allocated "reference" types (or values that have been "boxed" into reference types), because the dynamic type of an object can almost always be read out of the object itself.

When dealing with a value type, though, we have to deal with things like making *copies*:

```
interface IWritable { [mutating] void write(int val); }

struct Cell : IWritable { int data; void write(int val) { data = val; } }

T copyAndClobber<T : IWritable>(T obj)
{
	T copy = obj;
	obj.write(9999);
	return copy;
}

void test()
{
	Cell cell = { 0 };
	Cell result = copyAndClobber(cell);
	// what is in `result.data`?
}
```

If we call `copyAndClober` on a `Cell` value, then does the line `obj.write` overwrite the data in the explicit `copy` that was made?
It seems clear that a user would expect `copy` to be unaffected in the case where `T` is a value type.

How does that get implemented in our runtime version of things? Let's imagine some C++ translation:

```
void copyAndClobber(Type* T, IWriteableWitnessTable* W, void* obj, void* _returnVal)
{
    void* copy = alloca(T->sizeInBytes);
    T->copyConstruct(copy, obj);

    W->write(obj, 9999);
    T->moveConstruct(_returnVal, copy);
}
```

Because this function returns a value of type `T` and we don't know how big that is, let's assume the caller is passing in a pointer to the storage where we should write the result.
Now, in order to have a local `copy` of the `obj` value that was passed in, we need to allocate some scratch storage, and only the type `T` can know how many bytes we need.
Furthermore, when copying `obj` into that storage, or subsequently copying the `copy` variable into the function result, we need the copy/move semantics of type `T` to be provided by somebody.

This is the reason for passing through the type `T` as part of an existential value.

If we only wanted to deal with reference types, this would all be greatly simplified, because the `sizeInBytes` and the copy/move semantics would be fixed: everything is a single pointer.

All of the same issues arise if we're making copies of existential values:

```
IWritable copyAndClobberExistential(IWritable obj)
{
	IWritable copy = obj;
	obj.write(9999);
	return copy;
}
```

If we want to stay consistent and say that `copy` is an actual copy of `obj` when the underlying type is a value rather than a reference type, then we need the copy/move operations for `IWritable` to handle invoking the copy/move operations of the underlying encapsulated type.

Aside: it should be clear from these examples that implementing generics and existential types with dynamic dispatch has a lot of complexity when we have to deal with value types (because copying requires memory allocation).
It is likely that a first implementation of dynamic dispatch support for Slang would restrict it to reference types (and would thus add a `class` keyword for defining reference types).
