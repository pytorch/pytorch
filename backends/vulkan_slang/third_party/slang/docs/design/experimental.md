Deploying Experimental API Additions
====================================

This page intends to provide guidance to Slang developers when extending the Slang API, particularly when working on experimental features.
It applies to the "COM-lite" Slang API, rather than the deprecated C Slang API (sp* functions).

* Note: This guidance relates to Slang API changes, not to language changes. That is, what Slang does with shader source code across releases is not discussed here.

The goal is to maintain binary compatibility as much as possible between Slang releases, and to aid applications in dealing with changes to Slang.

Slang is distributed as a dynamic library, and there is an expectation from Slang API users that upgrading by installing an updated slang.dll or slang.so will not break their application unnecessarily.

ABI compatibility within the Slang API can be preserved between releases if some rules are followed by developers.

Slang API uses a "COM-lite" structure wherein functionality is exposed through interfaces on objects. If the interfaces never change, ABI compatibility is preserved, but changes happen. When adding or changing interfaces, please observe the following:

1. It is preferred to create *new* COM interfaces when adding new functionality.
* This maintains ABI compatibility.
* Applications must acquire access to the new functionality using QueryInterface(), which will gracefully fail if the slang.dll/slang.so does not implement the functionality.

2. Changes to existing virtual methods in COM interfaces should be avoided, as that is an ABI breakage.
* If a change is required though, change the interface's UUID.

3. New virtual methods _may_ be added (only) to the end of existing COM interface structs.
* This does not disturb the ABI compatibility of the associated vtable. Old apps can remain unaware of the new function pointers appended to the end of the vtable.
* A UUID change is not necessary.
* Note that in the event that a Slang application which uses the added feature is run with an old slang.dll/slang.so, the experience for the user is not as clean as if the added method belongs to a new interface.

Adding Experimental Interfaces
==============================

When the above recommendations cannot be followed, as with features that are expected to be iterated on or are regarded as temporary, there are additional recommendations.

Interfaces that are expected to change must be marked `_Experimental` in their class name and in their UUID name.

For example,


```csharp
/* Experimental interface for doing something cool. This interface is susceptible to ABI breakage. */
struct ICoolNewFeature_Experimental : public ISlangUnknown
{
    SLANG_COM_INTERFACE(0x8e12e8e3, 0x5fcd, 0x433e, { 0xaf, 0xcb, 0x13, 0xa0, 0x88, 0xbc, 0x5e, 0xe5 })

    virtual SLANG_NO_THROW SlangResult SLANG_MCALL coolMethod() = 0;
};

#define SLANG_UUID_ICoolNewFeature_Experimental ICoolNewFeature_Experimental::getTypeGuid()
```

Note: Use uuidgen to generate IIDs new interfaces.

Removing Experimental Interfaces
================================

By the nature of being marked "Experimental", users have been warned that the interfaces are not officially supported and may be removed. You may simply delete the class and UUID, e.g. "ICoolNewFeature_Experimental" struct may be deleted from slang.h along with the definition of SLANG_UUID_ICoolNewFeature_Experimental.

This will show up in applications as QueryInterface failures.

It is nice, but not required, to retain the interface declarations for some time after removing internal support before deleting them from slang.h, so that applications have time to remove their dependence on the unsupported feature while still being able to compile in the interim.

Changing Experimental Interfaces
================================

Backwards incompatible changes to Slang COM interfaces should be accompanied with a UUID change.

In the event that an old application runs with a new slang library, applications are more capable of gracefully handling an unavailable interface than a changed one. The former may be still be functional, or include a helpful error message, whereas the latter is most likely a crash of some sort.

Promoting Experimental Interfaces
=================================

The class name and the UUID name should be changed in slang.h and in the slang source code, e.g. Rename "ICoolNewFeature_Experimental" to just "ICoolFeature".

The SLANG_UUID for the interface should be renamed to omit "EXPERIMENTAL" but its value should remain the same. This is because, if there are no backwards incompatible changes that accompany the promotion from experimental to permanent, applications written against the experimental version can continue working against Slang libraries where the interface was promoted to permanent.
