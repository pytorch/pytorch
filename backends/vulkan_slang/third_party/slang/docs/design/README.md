Slang Design and Implementation Notes
=====================================

This directory contains documents that are primarily intended for developers working on the Slang implementation.
They are not intended to be helpful to Slang users.

These documents can only be trusted to reflect the state of the codebase or the plans of their authors at the time they were written. Changes to the implementation are not expected to always come with matching changes to these documents, so some amount of drift is to be expected.

Developers interested in contributing to Slang might want to start with the [Overview](overview.md) document, which describes the overall compilation pipeline that Slang uses and the purpose of the various steps (both implemented and planned).

The [Coding Conventions](coding-conventions.md) document describes the conventions that should be followed in all code added to the Slang project.

The [Interfaces](interfaces.md) document describes the high-level design plan for Slang's interfaces and generics features.

The [Declaration References](decl-refs.md) document is intended to help out developers who are mystified by the heavily used `DeclRef` type in the compiler implementation.

The [Intermediate Representation (IR)](ir.md) document describes the design of Slang's internal IR.

The [Existential Types](existential-types.md) document goes into some detail about what "existential types" are in the context of the Slang language, and explains how we may go about supporting them.

The [Capabilities](capabilities.md) document explains the proposed model for how Slang will support general notions of profile- or capability-based overloading/dispatch.

The [Casting](casting.md) document explains how casting works in the slang C++ compiler code base.

The [Experimental API Interfaces](experimental.md) document explains how experimental Slang API changes are to be deployed.