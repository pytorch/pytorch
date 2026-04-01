---
layout: user-guide
permalink: /user-guide/introduction
---

Introduction
============

Welcome to the _Slang User's Guide_, an introduction to the Slang language, compiler, and API.

Why use Slang?
--------------

The Slang system helps real-time graphics developers write cleaner and more maintainable GPU code, without sacrificing run-time performance.
Slang extends the HLSL language with thoughtfully selected features from modern general-purpose languages that support improved developer productivity and code quality.
These features have been carefully implemented with an understanding of GPU performance.

Some of the benefits of Slang include:

* Slang is backwards compatible with most existing HLSL code

* _Parameter blocks_ allow shader parameters to be grouped by update rate in order to take advantage of Direct3D 12 descriptor tables and Vulkan descriptor sets, without verbose and error-prone per-parameter markup

* _Interfaces_ and _generics_ provide first-class alternatives to hacky preprocessor-based or string-pasting shader specialization. Preprocessor hacks can be replaced with a well-understood language feature already used in Rust, Swift, C#, Java, and more.

* _Automatic differentiation_ greatly simplifies the implementation of learning-based techniques in shaders. Slang supports automatically generating both forward derivative and backward derivative propagation functions from forward computation code.

* Slang supports a first-class _module_ system, which enables true separate compilation and semantic checking of shader code. 

* Slang supports compute, rasterization, and ray-tracing shaders

* The same Slang compiler can generate code for DX bytecode, DXIL, SPIR-V, HLSL, GLSL, CUDA, and more

* Slang provides a robust and feature-complete reflection API, which provides binding/offset/layout information about all shader parameters in a consistent format across all the supported targets

Who is Slang for?
-----------------

Slang aims to be the best language possible for real-time graphics developers who care about code quality, portability and performance.

### Real-Time Graphics Developers

Slang is primarily intended for developers creating real-time graphics applications that run on end-user/client machines, such as 3D games and digital content creation (DCC) tools.

Slang can still provide value in other scenarios -- offline rather than real-time rendering, non-graphics GPU programming, or for applications that run on a server instead of client machines -- but the system has been designed first and foremost around the requirements of real-time graphics.

### From Hobbyists to Professionals

The Slang language is simple and familiar enough for hobbyist developers to use, but scales up to the demands of professional development teams creating next-generation game renderers.

### Developers of Multi-Platform Applications

The Slang system builds for multiple OSes, supports many graphics APIs, and works with GPUs from multiple hardware vendors.
The project is completely open-source and patches to support additional platforms are welcome.

Even for developers who only care about a single target platform or graphics API, Slang can provide a better programming experience than the default/native GPU language for that API.

### Developers with an existing investment in HLSL code

One of Slang's key features is its high degree of compatibility with existing HLSL code.
Developers who are currently responsible for large HLSL codebases but find themselves chafing at the restrictions of that language can incrementally adopt the features of Slang to improve the quality of their codebase over time.

Developers who do not have an existing investment in HLSL code, or who already have a large codebase in some other language will need to carefully consider the trade-offs in migrating to a new language (whether Slang or something else).

Who is this guide for?
----------------------

The content of this guide is written for real-time graphics programmers with a moderate or higher experience level.
It assumes the reader has previously used a real-time shading language like HLSL, GLSL, or MetalSL together with an API like Direct3D 11/12, Vulkan, or Metal.
We also assume that the reader is familiar enough with C/C++ to understand code examples and API signatures in those languages.

If you are new to programming entirely, this guide is unlikely to be helpful.
If you are an experienced programmer but have never worked in real-time graphics with GPU shaders, you may find some of the terminology or concepts from the domain confusing.

If you've only ever used OpenGL or Direct3D 11 before, some references to concepts in "modern" graphics APIs like D3D12/Vulkan/Metal may be confusing.
This effect may be particularly pronounced for OpenGL users.

It may be valuable for a user with limited experience with "modern" graphics APIs to work with both this guide and a guide to their chosen API (e.g., Direct3D 12, Vulkan, or Metal) so that concepts in each can reinforce the other.

When introducing Slang language features, this guide may make reference to languages such as Swift, Rust, C#, or Java.
Readers who almost exclusively use C/C++ may find certain features surprising or confusing, especially if they insist on equating concepts with the closest thing in C++ (assuming "generics `==` templates").

Goals and Non-Goals
-------------------

The rest of this guide introduces the services provided by the Slang system and explains how to use them to solve challenges in real-time graphics programming.
When services are introduced one after another, it may be hard to glimpse the bigger picture: why these particular services? Why these implementations? Why these APIs?

Before we dive into actually _using_ Slang, let us step back and highlight some of the key design goals (and non-goals) that motivate the design:

* **Performance**: Real-time graphics demands high performance, which motivates the use of GPUs. Whenever possible, the benefits of using Slang must not come at the cost of performance. When a choice involves a performance trade-off the *user* of the system should be able to make that choice.

* **Productivity**: Modern GPU codebases are large and growing. Productivity in a large codebase is less about _writing_ code quickly, and more about having code that is understandable, maintainable, reusable, and extensible. Language concepts like "modularity" or "separate compilation" are valuable if they foster greater developer productivity.

* **Portability**: Real-time graphics developers need to support a wide variety of hardware, graphics APIs, and operating systems. These platforms differ greatly in the level of functionality they provide. Some systems hand-wave portability concerns out of existence by enforcing a "lowest common denominator" approach and/or raising their "min spec" to exclude older or less capable platforms; our goals differ greatly. We aspire to keep our "min spec" as low as is practical (e.g., supporting Direct3D 11 and not just Direct3D 12), while also allowing each target to expose its distinguishing capabilities.

* **Ease of Adoption**: A language feature or service is worthless if nobody can use it. When possible, the system should be compatible with existing code and approaches. New language features should borrow syntax and semantics from other languages users might be familiar with. APIs and tools might need to support complicated and detailed use-cases, but should also provide conveniences and short-cuts for the most common cases.

* **Predictability**: Code should do what it appears to, consistently, across as many platforms as possible. Whenever possible the compiler should conform to programmer expectation, even in the presence of "undefined behavior." Tools and optimization passes should keep their behavior as predictable as possible; simple tools empower the user to do smart things.

* **Limited Scope**: The Slang system is a language, compiler, and module. It is not an engine, not a renderer, and not a "framework." The Slang system explicitly does *not* assume responsibility for interacting with GPU APIs to load code, allocate resources, bind parameters, or kick off work. While a user *may* use the Slang runtime library in their application, they are not *required* to do so.

The ordering here is significant, with earlier goals generally being more important than later ones.
