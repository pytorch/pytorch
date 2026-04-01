Interfaces Design
=================

This document intends to lay out the proposed design for a few inter-related features in Slang:

- Interfaces
- Associated Types
- Generics

Introduction
------------

The basic problem here is not unique to shader programming: you want to write code that accomplished one task, while abstracting over how to accomplish another task.
As an example, we might want to write code to integrate incident radiance over a list of lights, while not concerning ourself with how to evaluate a reflectance function at each of those lights.

If we were doing this task on a CPU, and performance wasn't critical, we could probably handle this with higher-order functions or an equivalent mechanism like function pointers:

    float4 integrateLighting(
    	Light[] lights,
    	float4 (*brdf)(float3 wi, float3 wi, void* userData),
    	void const* brdfUserData)
    {
    	float4 result = 0;
    	for(/* ... */) {
    		// ...
    		result += brdf(wi, wo, brdfUserDat);
    	}
    	return result;
    }

Depending on the scenario, we might be able to generate statically specialized code by using templates instead:

    template<typename BRDF>
    float4 integrateLighting(Light[] lights, BRDF const& brdf)
    {
    	// ...
    	result += brdf(wi, wo);
    	// ...
    }

Current shading languages support neither higher-order functions nor templates/generics, so neither of these options is viable.
Instead practitioners typically use preprocessor techniques to either stich together the final code, or to substitute in different function/type definitions to make a definition like `integrateLighting` reusable.

These ad hoc approaches actually work well in practice; we aren't proposing to replace them *just* to make code abstractly "cleaner."
Rather, we've found that the ad hoc approaches end up interacting poorly with the resource binding model in modern APIs, so that *something* less ad hoc is required to achieve our performance goals.
At that point, we might as well ensure that the mechanism we introduce is also a good fit for the problem.

Overview
--------

The basic idea for our approach is as follows:

- Start with the general *semantics* of a generic-based ("template") approach

- Use the accumulated experience of the programming language community to ensure that our generics are humane (in other words: not like C++)

- Expore the possibility of syntax sugar to let people use more traditional OOP-style syntax when it can reduce verbosity without harming understanding

In general, our conceptual model is being ripped off wholesale from Rust and Swift.
The basic design principle is "when in doubt, do what Swift does."

Interfaces
----------

An **interface** in Slang is akin to a `protocol` in Swift or a `trait` in Rust.
The choice of the `interface` keyword is to highlight the overlap with the conceptually similar construct that appeared in Cg, and then later in HLSL.

### Declaring an interface

An interface is a named collection of **requirements**; any type that **implements** the interface must provide definitions that satisfy those requirements.

Here is a simple interface, with one requirement:

    interface Light
    {
    	float3 illuminate(float3 P_world);
    }

The `Light` interface requires a (member) function called `illuminate` with the given signature.

### Declaring that a type implementats an interface

A user-defined `struct` type can declare that it implements an interface, by using conventional "inheritance" syntax:

    struct PointLight : Light
    {
    	float3 P_light;

    	float3 illuminate(float3 P_world)
    	{
    		float distance = length(P_light - P_world);
    		// ...
    	}
    }

It is a static error if a type declares that it implements an interface, but it does not provide all of the requirements:

    struct BadLight : Light
    {
    	// ERROR: type 'BadLight' cannot implement 'Light'
    	// because it does not provide the required 'illuminate' function
    }

### Interface Inheritance

While this document does not propose general notions of inheritance be added to Slang, it does make sense to allow an interface to inherit from zero or more other interfaces:

    interface InfinitessimalLight : Light
    {
    	float3 getDirection(float3 P_world);
    }

In this case the `InfinitessimalLight` interface inherits from `Light`, and declares one new requirement.
In order to check that a type implements `InfinitessimalLight`, the compiler will need to check both that it implements `Light` and that it provides the new "direct" requirements in `InfinitessimalLight`.

Declaring that a type implements an interface also implicitly declares that it implements all the interfaces that interface transitively inherits from:

    struct DirectionalLight : InfinitessimalLight
    {
    	float3 L;
    	float3 dir;

    	float3 getDirection(float3 P_world) { return dir; }

    	float3 illuminate(float3 P_world)
    	{
    		// Okay, this is the point where I recognize
    		// that this function definition is not
    		// actually reasonable for a light...
    }



### Interfaces and Extensions

It probably needs its own design document, but Slang currently has very basic support for `extension` declarations that can add members to an existing type.
These blocks correspond to `extension` blocks in Swift, or `impl` blocks in Rust.
This can be used to declare that a type implements an interface retroactively:

    extension PointLight : InfinitessimalLight
    {
    	float3 getDirection(float3 P_world)
    	{
    		return normalize(P_light - P_world);
    	}
    }

In this case we've used an extension to declare the `PointLight` also implements `InfinitessimalLight`. For the extension to type-check we need to provide the new required function (the compiler must recognize that the implementation of `Light` was already provided by the original type definition).

There are some subtleties around using extensions to add interface implementations:

- If the type already provides a method that matches a requireemnt, can the extension "see" it to satisfying new requirements?

- When can one extension "see" members (or interface implementations) added by another?

A first implementation can probably ignore the issue of interface implementations added by extensions, and only support them directly on type definitions.

Generics
--------

All of the above discussion around interfaces neglected to show how to actually *use* the fact that, e.g., `PointLight` implements the `Light` interface.
That is intentional, because at the most basic level, interfaces are designed to be used in the context of **generics**.

### Generic Declarations

The Slang compiler currently has some ad hoc support for generic declarations that it uses to implement the HLSL standard module (which has a few generic types).
The syntax for those is currently very bad, and it makes sense to converge on the style for generic declarations used by C# and Swift:

    float myGenericFunc<T>(T someValue);

Types can also be generic:

    struct MyStruct<T> { float a; T b; }

Ideally we should also allow interfaces and interface requirements to be generic, but there will probably be some limits due to implementation complexity.

### Type Constraints

Unlike C++, Slang needs to be able to type-check the body of a generic function ahead of time, so it can't rely on `T` having particular members:

    // This generic is okay, because it doesn't assume anything about `T`
    // (other than the fact that it can be passed as input/output)
    T okayGeneric<T>(T a) { return a; }

    // This generic is not okay, because it assumes that `T` supports
    // certain operators, and we have no way of knowing it this is true:
    T notOkayGeneric<T>(T a) { return a + a; }

In order to rely on non-trivial operations in a generic parameter type like `T`, the user must **constrain** the type parameter using an interface:

    float3 mySurfaceShader<L : Light>(L aLight)
    {
    	return aLight.illuminate(...);
    }

In this example, we have constrained the type parameter `L` so that it must implement the interface `Light`.
As a result, in the body of the function, the compiler can recognize that `aLight`, which is of type `L`, must implement `Light` and thus have a member `illuminate`.

When calling a function with a constrained type parameter, the compiler must check that the actual type argument (whether provided explicitly or inferred) implements the interface given in the constraint:

    mySurfaceShader<PointLight>(myPointLight);  // OK
    mySurfaceShader(myPointLight);				// equivalent to previous
    mySurfaceShader(3.0f); // ERROR: `float` does not implement `Light`

Note that in the erroneous case, the error is reported at the call site, rather than in the body of the callee (as it would be for C++ templates).

For cases where we must constrain a type parameter to implement multiple interfaces, we can join the interface types with `&`:

	interface Foo { void foo(); }
	interface Bar { void bar(); }

    void myFunc<T : Foo & Bar>(T val)
    {
    	val.foo();
    	val.bar();
    }

If we end up with very complicated type constraints, then it makes sense to support a "`where` clause" that allows requirements to be stated outside of the generic parameter list:

    void myFunc<T>(T val)
        where T : Foo,
        	  T : Bar
    {}

Bot the use of `&` and `where` are advanced features that we might cut due to implementation complexity.

### Value Parameters

Because HLSL has generics like `vector<float,3>` that already take non-type parameters, the language will need *some* degree of support for generic parameters that aren't types (at least integers need to be supported).
We need syntax for this that doesn't bloat the common case.

In this case, I think that what I've used in the current Slang implementation is reasonable, where a value parameter needs a `let` prefix:

    void someFunc<
    	T, 					// type parameter
    	T : X, 				// type parameter with constraint
    	T = Y, 				// type parameter with default
    	T : X = Y, 			// type parameter with constraint and default
    	let N : int,		// value parameter (type must be explicit)
    	let N : int = 3>	// value parameter with default
    	()
    { ... }

We should also extend the `where` clauses to support inequality constraints on (integer) value parameters to enforce rules about what ranges of integers are valid.
The front-end should issue error messages if it can statically determine these constraints are violated, but it should probably defer full checking until the IR (maybe... we need to think about how much of a dependent type system we are willing to have).

Associated Types
----------------

While the syntax is a bit different, the above mechanisms have approximately the same capabilities as Cg interfaces.
What the above approach can't handle (and neither can Cg) is a reusable definition of a surface material "pattern" that might blend multiple material layers to derive parameters for a specific BRDF.

That is, suppose we have two BRDFs: one with two parameters, and one with six.
Different surface patterns may want to target different BRDFs.
So if we write a `Material` interface like:

    interface Material
    {
    	BRDFParams evaluatePattern(float2 uv);
    }

Then what should `BRDFParams` be? The two-parameter or six-parameter case?

An **associated type** is a concept that solves exactly this problem.
We don't care *what* the concrete type of `BRDFParams` is, so long as *every* implementation of `Material` has one.
The exact `BRDFParams` type can be different for each implementation of `Material`; the type is *associated* with a particular implementation.

We will crib our syntax for this entirely from Swift, where it is verbose but explicit:

    interface Material
    {
    	associatedtype BRDFParams;

    	BRDFParams evaluatePattern(float2 uv);

    	float3 evaluateBRDF(BRDFParams param, float3 wi, float3 wo);
    }

In this example we've added an associated type requirement so that every implementation of `Material` must supply a type named `BRDFParams` as a member.
We've also added a requirement that is a function to evaluate the BRDF given its parameters and incoming/outgoing directions.

Using this declaration one can now define a generic function that works on any material:

    float3 evaluateSurface<M : Material, L : Light>(
    	M material,
    	L[] lights,
    	float3 P_world,
    	float2 uv)
    {
    	P.BRDFParams brdfParams = material.evaluatePattern(uv);
    	for(...)
    	{
    		L light = lights[i];
    		// ...
    		float3 reflectance = material.evaluateBRDF(brdfParams, ...);
    	}
    }

Some quick notes:

- The use of `associatedtype` (for associated types) and `typealias` (for `typedef`-like definitions) as distinct keywords in Swift was well motivated by their experience (they used to use `typealias` for both). I would avoid having the two cases be syntactically identical.

- Swift has a pretty involved inference system where a type doesn't actually need to explicitly provide a type member with the chosen name. Instead, if you have a required method that takes or returns the associated type, then the compiler can infer what the type is by looking at the signature of the methods that meet other requirements. This is a complex and magical feature, and we shouldn't try to duplicate it.

- Both Rust and Swift call this an "associated type." They are related to "virtual types" in things like Scala (which are in turn related to virtual classes in beta/gbeta). There are similar ideas that arise in Haskell-like languages with type classes (IIRC, the term "functional dependencies" is relevant).

### Alternatives

I want to point out a few alternatives to the `Material` design above, just to show that associated types seem to be an elegant solution compared to the alternatives.

First, note that we could break `Material` into two interfaces, so long as we are allowed to place type constraints on associated types:

    interface BRDF
    {
    	float3 evaluate(float3 wi, float3 wo);
    }

    interface Material
    {
    	associatedtype B : BRDF;

    	B evaluatePattern(float2 uv);
    }

This refactoring might be cleaner if we imagine that a shader library would have family of reflectance functions (implementing `BRDF`) and then a large library of material patterns (implementing `Material`) - we wouldn't want each and every material to have to implement a dummy `evaluateBRDF` that just forwards to a BRDF instance nested in it.

Looking at that type `B` there, we might start to wonder if we could just replace this with a generic type parameter on the interface:

    interface Material< B : BRDF >
    {
    	B evaluatePattern(float2 uv);
    }

This would change any type that implements `Material`:

    // old:
    struct MyMaterial : Material
    {
    	typealias B = GGX;

    	GGX evaluatePattern(...) { ... }
    }

    // new:
    struct MyMaterial : Material<GGX>
    {
    	GGX evaluatePattern(...) { ... }
    }

That doesn't seem so bad, but it ignores the complexity that arises at any use sites, e.g.:

    float3 evaluateSurface<B : BRDF, M : Material<B>, L : Light>(
    	M material,
    	L[] lights,
    	float3 P_world,
    	float2 uv)
    { ... }

The type `B` which is logically an implementation detail of `M` now surfaces to the generic parameter list of any function that wants to traffic in materials.
This reduces the signal/noise ratio for anybody reading the code, and also means that any top-level code that is supposed to be specializing this function (suppose this was a fragment entry point) now needs to understand how to pick apart the `Material` it has on the host side to get the right type parameters.

This kind of issue has existed in the PL community at least as far back as the ML module system (it is tough to name search, but the concepts of "parameterization" vs. "fibration" is relevant here), and the Scala researchers made a clear argument (I think it was in the paper on "un-types") that there is a categorical distinction between the types that are logicall the *inputs* to an abstraction, and the types that are logically the *outputs*. Generic type parameters and associated types handle these two distinct roles.

Returning an Interface
----------------------

The revised `Material` definition:

    interface BRDF
    {
    	float3 evaluate(float3 wi, float3 wo);
    }

    interface Material
    {
    	associatedtype B : BRDF;

    	B evaluatePattern(float2 uv);
    }

has a function `evaluatePattern` that returns a type that implements an interface.
In the case where the return type is concrete, this isn't a problem (and the nature of associated types means that `B` will be concrete in any actual concrete implementation of `Material`).

There is an open question of whether it is ever necessary (or even helpful) to have a function that returns a value of *some* type known to implement an interface, without having to state that type in the function signature.
This is a point that has [come up](https://github.com/rust-lang/rfcs/blob/master/text/1951-expand-impl-trait.md) in the Rust world, where they have discussed using a keyword like `some` to indicate the existential nature of the result type:

	// A function that returns *some* implementation of `Light`
	func foo<T>() -> some Light;

The Rust proposal linked above has them trying to work toward `impl` as the keyword, and allowing it in both argument and result positions (to cover both universal and existential quantification).

In general, such a feature would need to have many constraints:

- The concrete return type must be fixed (even if clients of the function should be insulated from the choice), given the actual generic arguments provided.

- If the existential is really going to be sealed, then the caller shouldn't be allowed to assume anything *except* that two calls to the same function with identical generic arguments should yield results of identical type.

Under those constraints, it is pretty easy to see that an existential-returning method like:

    interface Foo<T>
    {
    	func foo<U>() -> some Bar;
    }

can in principle be desugared into:

    interface Foo<T>
    {
    	associatedtype B<U> : Bar;

    	func foo<U>() -> B<U>;
    }

with particular loss in what can be expressed.
The same desugaring approach should apply to global-scope functions that want to return an existential type (just with a global `typealias` instead of an `associatedtype`).


It might be inconvenient for the user to have to explicitly write the type-level expression that yields the result type (consider cases where C++ template metaprogrammers would use `auto` as a result type), but there is really no added power.


Object-Oriented Sugar
---------------------

Having to explicitly write out generic parameter lists is tedious, especially in the (common) case where we will have exactly one parameter corresponding to each generic type parameter:

	// Why am I repeating myself?!
	//
    void foo<L : Light, M : Material, C : Camera)(
    	     L   light, M   material, C   camera);

The intent seems to be clear if we instead write:

    void foo(Light light, Material material, Camera camera);

We could consider the latter to be sugar for the former, and allow users to write in familiar syntax akin to what ws already supported in Cg.

We'd have to be careful with such sugar, though, because there is a real and meaningful difference between saying:

- "`material` has type `Material` which is an interface type"
- "`material` has type `M` where `M` implements `Material`"

In particular, if we start to work with associated types:

    let b = material.evaluatePattern(...);

It makes sense to say that `b` has type `M.BRDF`.
It does **not** make sense to say that `b` has type `Material.BRDF`, because there is no such concrete type.

(A third option is to say that `b` has type `material.BRDF`, which is basically the point where you have "virtual types" because we are now saying the type is a member of the *instance* and not of an enclosing *type*)

Note that the issue of having or not having object-oriented sugar is technically orthogonal from whether we allow "existential return types."
However, allowing the user to think of interfaces in traidtional OOP terms leads to it being more likely that they will try to declare:

- functions that return an interface type
- local variables of interface type (which they might even assign to!)
- fields of interface type in their `struct`s

All of these complicate the desugaring step, because we would de facto have types/functions that mix up two stages of evaluation: a compile-time type-level step and a run-time value-level step.
Ultimately, we'd probably need to express these by having a multi-stage IR (with two stages) which we optimize in the staged setting before stage-splitting to get separate type-level and value-level operations (akin to the desugaring for existential return types I described above).

My sense is that a certain amount of multi-stage programming may already be needed to deal with certain HLSL/GLSL idioms. In particular:

- GLSL supports passing unsigned arrays (e.g., `int[] a`) to a function, and then having the function use the size of the array (`a.length`) to do loops, etc. These would need to be lowered to distinct SPIR-V code for every array size used (if I understand the restrictions correctly), and so the feature is perhaps best thought of as passing both a compile-time integer parameter and a run-time array parameter (where the size comes from that parameter)

- HLSL and GLSL both have built-in functions where certain parameters are required to be compile-time constants. A feature-complete front-end must detect when calls to these functions are valid, and report errors to the user. In order to make the errors easier to explain to the user, it would be helpful to have an explicit notion of constant-rate computation, and require that the user express explicit constant-rate parameters/expressions.

All of this ties into the question of whether we need/want to support more general kinds of compile-time evaluation for specialization (e.g., statically-determine `if` statements or loops).

Other Languages
---------------

It is worth double-checking whether implementing all of this from scratch in Slang is a good idea, or if there is somewhere else we can achieve similar results more quickly:

- The Metal shading language has much of what we'd want. It is based on C++ templates, which are maybe not the ideal mechanism, and the compiler is closed-source so we can't easily add functionality. Still, it should be possible to prototype a lot of what we want on top of Metal 2.

- The open-source HLSL compiler doesn't support any of the new ideas here, but it may be that adding them to `dxc` would be faster than adding them to the Slang project code. Using `dxc` is a no-go for some of the other Slang requirements (that come from our users on the Falcor project).

- Swift already supports almost every thing on our list of requirements, but as it stands today there is no easy path to using it for low-level GPU code generation. It also fails to meet our goals for incremental adoption, high-level source output, etc.

  In the long run, however, the Swift compiler seems like an attractive intercept for this work, because their long-term roadmap seems like it will close a lot of the gap with what we've done so far.

Conclusion
----------

This document has described the basic syntax and semantics for three related features -- interfaces, generics, and associated types -- along with some commentary on longer-term directions.
My expectation is that we will use the syntax as laid down here, unless we have a very good reason to depart from it, and we will prioritize implementation work as needed to get interesting shader library functionality up and running.
