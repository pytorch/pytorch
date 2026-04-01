Capabilities (Out of Date)
============

Slang aims to be a portable language for shader programming, which introduces two complementary problems:

1. We need a way to indicate that certain constructs (types, functions, etc.) are only allowed on certain targets, so that a user gets a meaningful error if they try to do something that won't work on one or more of the APIs or platforms they want to target. Similarly, the user expects to get an error if they call a fragment-shader-specific function inside of, say, compute shader code, or vice versa.

2. If the same feature can be implemented across multiple platforms, but the best (or only) implementation path differs across platforms, then we need a way to express the platform specific code and pick the right implementation per-target.

Item (2) is traditionally handled with preprocessor techniques (e.g., `#ifdef`ing the body of a function based on target platform), but that of course requires that the user invoke the Slang front end once for each target platform, and target-specific coding in a library will then "infect" code that uses that library, forcing them to invoke the front-end once per target as well.

We are especially sensitive to this problem in the compiler itself, because we have to author and maintain the Slang standard modules, which needs to (1) expose the capabilities of many platforms and (2) work across all those platforms. It would be very unfortunate if we had to build different copies of our standard modules per-target.

The intention in Slang is to solve both of these problems with a system of *capabilities*.

What is a capability?
---------------------

For our purposes a capability is a discrete feature that a compilation target either does or does not support.
We could imagine defining a capability for the presence of texture sampling operations with implicit gradients; this capability would be supported when generating fragment shader kernel code, but not when generating code for other stages.

Let's imagine a language syntax that the standard modules could use to define some *atomic* capabilities:

```
capability implicit_gradient_texture_fetches;
```
We can then imagine using attributes to indicate that a function requires a certain capability:

```
struct Texture2D
{
	...

	// Implicit-gradient sampling operation.
	[availableFor(implicit_gradient_texture_fetches)]
	float4 Sample(SamplerState s, float2 uv);
}
```

(Note that the `[availableFor(...)]` syntax is just a straw-man to write up examples, and a better name would be desirable if/when we implement this stuff.)

Given those declarations, we could then check when compiling code if the user is trying to call `Texture2D.Sample` in code compiled for a target that *doesn't* support implicit-gradient texture fetches, and issue an appropriate error.
The details on how to sequence this all in the compiler will be covered later.

Derived Capabilities
--------------------

Once we can define atomic capabilities, the next step is to be able to define *derived* capabilities.
Let's imagine that we extend our `capability` syntax so that we can define a new capability that automatically implies one or more other capabilities:

```
capability fragment : implicit_gradient_texture_fetches;
```

Here we've said that whenever the `fragment` capability is available, we can safely assume that the `implicit_gradient_texture_fetches` capability is available (but not vice versa).

Given even a rudimentary tool like that, we can start to build up capabilities that relate closely to the "profiles" in things like D3D:

```
capability d3d;
capability sm_5_0 : d3d;
capability sm_5_1 : sm_5_0;
capability sm_6_0 : sm_5_1;
...

capability d3d11 : d3d, sm_5_0;
capability d3d12 : d3d, sm_6_0;

capability khronos;
capability glsl_400 : khronos;
capability glsl_410 : glsl_400;
...

capability vulkan : khronos, glsl_450;
capability opengl : khronos;
```

Here we are saying that `sm_5_1` supports everything `sm_5_0` supports, and potentially more. We are saying that `d3d12` supports `sm_6_0` but maybe not, e.g., `sm_6_3`.
We are expressing that fact that having a `glsl_*` capability means you are on some Khronos API target, but that it doesn't specify which one.
(The exact details of these declarations obviously aren't the point; getting a good hierarchy of capabilities will take time.)

Capability Composition
----------------------

Sometimes we'll want to give a distinct name to a specific combination of capabilities, but not say that it supports anything new:

```
capability ps_5_1 = sm_5_1 & fragment;
```

Here we are saying that the `ps_5_1` capability is *equivalent* to the combination of `sm_5_1` and `fragment` (that is, if you support both `sm_5_1` and `fragment` then you support `ps_5_1` and vice versa).

Compositions should be allowed in `[availableFor(...)]` attributes (e.g., `[availableFor(vulkan & glsl_450)]`), but pre-defined compositions should be favored when possible.

When composing things with `&` it is safe for the compiler to filter out redundancies based on what it knows so that, e.g., `ps_5_0 & fragment` resolves to just `ps_5_0`.

Once we have an `&` operator for capabilities, it is easy to see that "derived" capabilities are really syntax sugar, so that a derived capability like:

```
capability A : B, C
```

could have been written instead as :

```
capability A_atomic
capability A = A_atomic & B & C
```

Where the `A_atomic` capability guarantees that `A` implies `B` and `C` but not vice versa.

It is also useful to think of an `|` operator on capabilities.
In particular if a function has multiple `[availableFor(...)]` attributes:

```
[availableFor(vulkan & fragment)]
[availableFor(d3d12 & fragment)]
void myFunc();
```

This function should be equivalent to one with just a single `[availableFor((vulkan & fragment) | (d3d12 & fragment))]` which is equivalent to `[availableFor((vulkan | d3d12) & fragment)]`.
Simplification should generally push toward "disjunctive normal form," though, rather than pursue simplifications like that.
Note that we do *not* include negation, so that capabilities are not general Boolean expressions.

Validation
----------

For a given function definition `F`, the front end will scan its body and see what it calls, and compose the capabilities required by the called functions using `&` (simplifying along the way). Call the resulting capability (in disjunctive normal form) `R`.

If `F` doesn't have an `[availableFor(...)]` attribute, then we can derive its *effective* `[availableFor(...)]` capability as `R` (this probably needs to be expressed as an iterative dataflow problem over the call graph, to handle cycles).

If `F` *does* have one or more `[availableFor(...)]` clauses that amount to a declared capability `C` (again in disjunctive normal form), then we can check that `C` implies `R` and error out if it is not the case.
A reasonable implementation would track which calls introduced which requirements, and be able to explain *why* `C` does not capture the stated requirements.

For a shader entry point, we should check it as if it had an `[availableFor(...)]` that is the OR of all the specified target profiles (e.g., `sm_5_0 | glsl_450 | ...`) ANDed with the specified stage (e.g., `fragment`).
Any error here should be reported to the user.
If an entry point has an explicit `[availableFor(...)]` then we should AND that onto the profile computed above, so that the user can restrict certain entry points to certain profiles.

In order to support separate compilation, the functions that are exported from a module should probably either have explicit availability attributes, or else they will be compiled against a kind of "default capability" used for the whole module.
Downstream code that consumes such a module would see declarations with explicit capabilities only.
Picking an appropriate "default capability" to use when compiling modules is an important challenge; it would in practice define the "min spec" to use when compiling.

Capability Overriding
---------------------

It should be possible to define multiple versions of a function, having different `[availableFor(...)]` attributes:

```
[availableFor(vulkan)] void myFunc() { ... }

[availableFor(d3d12)] void myFunc() { ... }
```

For front-end checking, these should be treated as if they were a single definition of `myFunc` with an ORed capability (e.g., `vulkan | d3d12`).
Overload resolution will pick the "best" candidate at a call site based *only* on the signatures of the function (note that this differs greatly from how profile-specific function overloading works in Cg).

The front-end will then generate initial IR code for each definition of `myFunc`.
Each of the IR functions will have the *same* mangled name, but different bodies, and each will have appropriate IR decorations to indicate the capabilities it requires.

The choice of which definition to use is then put off until IR linking for a particular target.
At that point we can look at all the IR functions matching a given mangled name, filter them according to the capabilities of the target, and then select the "best" one.

In general a definition `A` of an IR symbol is better than another definition `B` if the capabilities on `A` imply those on `B` but not versa.
(In practice this probably needs to be "the capabilities on `A` intersected with those of the target," and similarly for `B`)

This approach allows us to defer profile-based choices of functions to very late in the process. The one big "gotcha" to be aware of is when functions are overloaded based on pipeline stage, where we would then have to be careful when generating DXIL or SPIR-V modules with multiple entry points (as a single function `f` might need to be specialized twice if it calls a stage-overloaded function `g`).

Capabilities in Other Places
----------------------------

So far I've talked about capabilities on functions, but they should also be allowed on other declarations including:

- Types, to indicate that code using that type needs the given capability
- Interface conformances, to indicate that a type only conforms to the interface when the capabilities are available
- Struct fields, to indicate that the field is only present in the type when the capabilities are present
- Extension declarations, to indicate that everything in them requires the specified capabilities

We should also provide a way to specify that a `register` or other layout modifier is only applicable for specific targets/stages. Such a capability nominally exists in HLSL today, but it would be much more useful if it could be applied to specify target-API-specific bindings.

Only functions should support overloading based on capability. In all other cases there can only be one definition of an entity, and capabilities just decide when it is available.

API Extensions as Capabilities
------------------------------

One clear use case for capabilities is to represent optional extensions, including cases where a feature is "built-in" in D3D but requires an extension in Vulkan:

```
capability KHR_secret_sauce : vulkan;

[available_for(sm_7_0)] // always available for D3D Shader Model 7.0
[available_for(KHR_secret_sauce)] // Need the "secret sauce" extension for Vulkan
void improveShadows();
```

When generating code for Vulkan, we should be able to tell the user that the `improveShadows()` function requires the given extension. The user should be able to express compositions of capabilities in their `-profile` option (and similarly for the API):

```
slangc code.slang -profile vulkan+KHR_secret_sauce
```
(Note that for the command line, it is beneficial to use `+` instead of `&` to avoid conflicts with shell interpreters)

An important question is whether the compiler should automatically infer required extensions without them being specified, so that it produces SPIR-V that requires extensions the user didn't ask for.
The argument against such inference is that users should opt in to non-standard capabilities they are using, but it would be unfortunate if this in turn requires verbose command lines when invoking the compiler.
It should be possible to indicate the capabilities that a module or entry point should be compiled to use without command-line complications.

(A related challenge is when a capability can be provided by two different extensions: how should the compiler select the "right" one to use?)

Disjoint Capabilities
---------------------

Certain compositions of capabilities make no sense. If a user declared a function as needing `vulkan & d3d12` they should probably get an error message.

Knowing that certain capabilities are disjoint can also help improve the overall user experience.
If a function requires `(vulkan & extensionA) | (d3d12 & featureb)` and we know we are compiling for `vulkan` we should be able to give the user a pointed error message saying they need to ask for `extensionA`, because adding `featureB` isn't going to do any good.

As a first-pass model we could have a notion of `abstract` capabilities that are used to model the root of hierarchies of disjoint capabilities:

```
abstract capability api;

abstract capability d3d : api;
capability d3d11 : d3d;
capability d3d12 : d3d;

abstract capability khronos : api;
capability vulkan : khronos;
capability opengl : khronos;
```

As a straw man:  we could have a rule that to decide if non-abstract capabilities `A` and `B` are disjoint, we look for their common ancestor in the tree of capabilities.
If the common ancestor is abstract, they are disjoint, and if not they not disjoint.
We'd also know that if the user tries to compile for a profile that includes an abstract capability but *not* some concrete capability derived from it, then that is an error (we can't generate code for just `d3d`).

The above is an over-simplification because we don't have a *tree* of capabilities, but a full *graph*, so we'd need an approach that works for the full case.

Interaction with Generics/Interfaces
------------------------------------

It should be possible for an interface requirement to have a capability requirement attached to it.
This would mean that users of the interface can only use the method/type/whatever when the capability is present (just like for any other function):

```
interface ITexture
{
	float4 sampleLevel(float2 uv, float lod);

	[availableFor(fragment)]
	float4 sample(float2 uv); // can only call this from fragment code
}
```
When implementing an interface, any capability constraints we put on a member that satisfies an interface requirement would need to guarantee that either:

- the capabilities on our method are implied by those on the requirement (we don't require more), or

- the capabilities on the method are implied by those on the type itself, or its conformance to the interface (you can't use the conformance without the capabilities), or

- the capabilities are already implied by those the whole module is being compiled for

In each case, you need to be sure that `YourType` can't be passed as a generic argument to some function that uses just the `ITexture` interface above and have them call a method on your type from a profile that doesn't have the required capabilities.

Interaction with Heterogeneity
------------------------------

If Slang eventually supports generating CPU code as well as shaders, it should use capabilities to handle the CPU/GPU split similar to how they can be used to separate out vertex- and fragment-shader functionality.
Something like a `cpu` profile that works as a catch-all for typical host CPU capabilities would be nice, and could be used as a convenient way to mark "host" functions in a file that is otherwise compiled for a "default profile" that assumes GPU capabilities.

Conclusion
----------

Overall, the hope is that in many cases developers will be able to use capability-based partitioning and overloading of APIs to build code that only has to pass through the Slang front-end once, but that can then go through back-end code generation for each target.
In cases where this can't be achieved, the way that capability-based overloading is built into the Slang IR design means that we should be able to merge multiple target-specific definitions into one IR module, so that a module can employ target-specific specializations while still presenting a single API to consumers.
