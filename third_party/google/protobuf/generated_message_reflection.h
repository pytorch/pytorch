// Protocol Buffers - Google's data interchange format
// Copyright 2008 Google Inc.  All rights reserved.
// https://developers.google.com/protocol-buffers/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Author: kenton@google.com (Kenton Varda)
//  Based on original Protocol Buffers design by
//  Sanjay Ghemawat, Jeff Dean, and others.
//
// This header is logically internal, but is made public because it is used
// from protocol-compiler-generated code, which may reside in other components.

#ifndef GOOGLE_PROTOBUF_GENERATED_MESSAGE_REFLECTION_H__
#define GOOGLE_PROTOBUF_GENERATED_MESSAGE_REFLECTION_H__

#include <string>
#include <vector>
#include <google/protobuf/stubs/casts.h>
#include <google/protobuf/stubs/common.h>
// TODO(jasonh): Remove this once the compiler change to directly include this
// is released to components.
#include <google/protobuf/generated_enum_reflection.h>
#include <google/protobuf/message.h>
#include <google/protobuf/metadata.h>
#include <google/protobuf/unknown_field_set.h>


namespace google {
namespace upb {
namespace google_opensource {
class GMR_Handlers;
}  // namespace google_opensource
}  // namespace upb

namespace protobuf {
class DescriptorPool;
class MapKey;
class MapValueRef;
}

namespace protobuf {
namespace internal {
class DefaultEmptyOneof;

// Defined in this file.
class GeneratedMessageReflection;

// Defined in other files.
class ExtensionSet;             // extension_set.h

// THIS CLASS IS NOT INTENDED FOR DIRECT USE.  It is intended for use
// by generated code.  This class is just a big hack that reduces code
// size.
//
// A GeneratedMessageReflection is an implementation of Reflection
// which expects all fields to be backed by simple variables located in
// memory.  The locations are given using a base pointer and a set of
// offsets.
//
// It is required that the user represents fields of each type in a standard
// way, so that GeneratedMessageReflection can cast the void* pointer to
// the appropriate type.  For primitive fields and string fields, each field
// should be represented using the obvious C++ primitive type.  Enums and
// Messages are different:
//  - Singular Message fields are stored as a pointer to a Message.  These
//    should start out NULL, except for in the default instance where they
//    should start out pointing to other default instances.
//  - Enum fields are stored as an int.  This int must always contain
//    a valid value, such that EnumDescriptor::FindValueByNumber() would
//    not return NULL.
//  - Repeated fields are stored as RepeatedFields or RepeatedPtrFields
//    of whatever type the individual field would be.  Strings and
//    Messages use RepeatedPtrFields while everything else uses
//    RepeatedFields.
class LIBPROTOBUF_EXPORT GeneratedMessageReflection : public Reflection {
 public:
  // Constructs a GeneratedMessageReflection.
  // Parameters:
  //   descriptor:    The descriptor for the message type being implemented.
  //   default_instance:  The default instance of the message.  This is only
  //                  used to obtain pointers to default instances of embedded
  //                  messages, which GetMessage() will return if the particular
  //                  sub-message has not been initialized yet.  (Thus, all
  //                  embedded message fields *must* have non-NULL pointers
  //                  in the default instance.)
  //   offsets:       An array of ints giving the byte offsets, relative to
  //                  the start of the message object, of each field.  These can
  //                  be computed at compile time using the
  //                  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET() macro, defined
  //                  below.
  //   has_bits_offset:  Offset in the message of an array of uint32s of size
  //                  descriptor->field_count()/32, rounded up.  This is a
  //                  bitfield where each bit indicates whether or not the
  //                  corresponding field of the message has been initialized.
  //                  The bit for field index i is obtained by the expression:
  //                    has_bits[i / 32] & (1 << (i % 32))
  //   unknown_fields_offset:  Offset in the message of the UnknownFieldSet for
  //                  the message.
  //   extensions_offset:  Offset in the message of the ExtensionSet for the
  //                  message, or -1 if the message type has no extension
  //                  ranges.
  //   pool:          DescriptorPool to search for extension definitions.  Only
  //                  used by FindKnownExtensionByName() and
  //                  FindKnownExtensionByNumber().
  //   factory:       MessageFactory to use to construct extension messages.
  //   object_size:   The size of a message object of this type, as measured
  //                  by sizeof().
  GeneratedMessageReflection(const Descriptor* descriptor,
                             const Message* default_instance,
                             const int offsets[],
                             int has_bits_offset,
                             int unknown_fields_offset,
                             int extensions_offset,
                             const DescriptorPool* pool,
                             MessageFactory* factory,
                             int object_size,
                             int arena_offset,
                             int is_default_instance_offset = -1);

  // Similar with the construction above. Call this construction if the
  // message has oneof definition.
  // Parameters:
  //   offsets:       An array of ints giving the byte offsets.
  //                  For each oneof field, the offset is relative to the
  //                  default_oneof_instance. These can be computed at compile
  //                  time using the
  //                  PROTO2_GENERATED_DEFAULT_ONEOF_FIELD_OFFSET() macro.
  //                  For each none oneof field, the offset is related to
  //                  the start of the message object.  These can be computed
  //                  at compile time using the
  //                  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET() macro.
  //                  Besides offsets for all fields, this array also contains
  //                  offsets for oneof unions. The offset of the i-th oneof
  //                  union is offsets[descriptor->field_count() + i].
  //   default_oneof_instance: The default instance of the oneofs. It is a
  //                  struct holding the default value of all oneof fields
  //                  for this message. It is only used to obtain pointers
  //                  to default instances of oneof fields, which Get
  //                  methods will return if the field is not set.
  //   oneof_case_offset:  Offset in the message of an array of uint32s of
  //                  size descriptor->oneof_decl_count().  Each uint32
  //                  indicates what field is set for each oneof.
  //   other parameters are the same with the construction above.
  GeneratedMessageReflection(const Descriptor* descriptor,
                             const Message* default_instance,
                             const int offsets[],
                             int has_bits_offset,
                             int unknown_fields_offset,
                             int extensions_offset,
                             const void* default_oneof_instance,
                             int oneof_case_offset,
                             const DescriptorPool* pool,
                             MessageFactory* factory,
                             int object_size,
                             int arena_offset,
                             int is_default_instance_offset = -1);
  ~GeneratedMessageReflection();

  // Shorter-to-call helpers for the above two constructions that work if the
  // pool and factory are the usual, namely, DescriptorPool::generated_pool()
  // and MessageFactory::generated_factory().

  static GeneratedMessageReflection* NewGeneratedMessageReflection(
      const Descriptor* descriptor,
      const Message* default_instance,
      const int offsets[],
      int has_bits_offset,
      int unknown_fields_offset,
      int extensions_offset,
      const void* default_oneof_instance,
      int oneof_case_offset,
      int object_size,
      int arena_offset,
      int is_default_instance_offset = -1);

  static GeneratedMessageReflection* NewGeneratedMessageReflection(
      const Descriptor* descriptor,
      const Message* default_instance,
      const int offsets[],
      int has_bits_offset,
      int unknown_fields_offset,
      int extensions_offset,
      int object_size,
      int arena_offset,
      int is_default_instance_offset = -1);

  // implements Reflection -------------------------------------------

  const UnknownFieldSet& GetUnknownFields(const Message& message) const;
  UnknownFieldSet* MutableUnknownFields(Message* message) const;

  int SpaceUsed(const Message& message) const;

  bool HasField(const Message& message, const FieldDescriptor* field) const;
  int FieldSize(const Message& message, const FieldDescriptor* field) const;
  void ClearField(Message* message, const FieldDescriptor* field) const;
  bool HasOneof(const Message& message,
                const OneofDescriptor* oneof_descriptor) const;
  void ClearOneof(Message* message, const OneofDescriptor* field) const;
  void RemoveLast(Message* message, const FieldDescriptor* field) const;
  Message* ReleaseLast(Message* message, const FieldDescriptor* field) const;
  void Swap(Message* message1, Message* message2) const;
  void SwapFields(Message* message1, Message* message2,
                  const vector<const FieldDescriptor*>& fields) const;
  void SwapElements(Message* message, const FieldDescriptor* field,
                    int index1, int index2) const;
  void ListFields(const Message& message,
                  vector<const FieldDescriptor*>* output) const;

  int32  GetInt32 (const Message& message,
                   const FieldDescriptor* field) const;
  int64  GetInt64 (const Message& message,
                   const FieldDescriptor* field) const;
  uint32 GetUInt32(const Message& message,
                   const FieldDescriptor* field) const;
  uint64 GetUInt64(const Message& message,
                   const FieldDescriptor* field) const;
  float  GetFloat (const Message& message,
                   const FieldDescriptor* field) const;
  double GetDouble(const Message& message,
                   const FieldDescriptor* field) const;
  bool   GetBool  (const Message& message,
                   const FieldDescriptor* field) const;
  string GetString(const Message& message,
                   const FieldDescriptor* field) const;
  const string& GetStringReference(const Message& message,
                                   const FieldDescriptor* field,
                                   string* scratch) const;
  const EnumValueDescriptor* GetEnum(const Message& message,
                                     const FieldDescriptor* field) const;
  int GetEnumValue(const Message& message,
                   const FieldDescriptor* field) const;
  const Message& GetMessage(const Message& message,
                            const FieldDescriptor* field,
                            MessageFactory* factory = NULL) const;

  const FieldDescriptor* GetOneofFieldDescriptor(
      const Message& message,
      const OneofDescriptor* oneof_descriptor) const;

 private:
  bool ContainsMapKey(const Message& message,
                      const FieldDescriptor* field,
                      const MapKey& key) const;
  bool InsertOrLookupMapValue(Message* message,
                              const FieldDescriptor* field,
                              const MapKey& key,
                              MapValueRef* val) const;
  bool DeleteMapValue(Message* message,
                      const FieldDescriptor* field,
                      const MapKey& key) const;
  MapIterator MapBegin(
      Message* message,
      const FieldDescriptor* field) const;
  MapIterator MapEnd(
      Message* message,
      const FieldDescriptor* field) const;
  int MapSize(const Message& message, const FieldDescriptor* field) const;

 public:
  void SetInt32 (Message* message,
                 const FieldDescriptor* field, int32  value) const;
  void SetInt64 (Message* message,
                 const FieldDescriptor* field, int64  value) const;
  void SetUInt32(Message* message,
                 const FieldDescriptor* field, uint32 value) const;
  void SetUInt64(Message* message,
                 const FieldDescriptor* field, uint64 value) const;
  void SetFloat (Message* message,
                 const FieldDescriptor* field, float  value) const;
  void SetDouble(Message* message,
                 const FieldDescriptor* field, double value) const;
  void SetBool  (Message* message,
                 const FieldDescriptor* field, bool   value) const;
  void SetString(Message* message,
                 const FieldDescriptor* field,
                 const string& value) const;
  void SetEnum  (Message* message, const FieldDescriptor* field,
                 const EnumValueDescriptor* value) const;
  void SetEnumValue(Message* message, const FieldDescriptor* field,
                    int value) const;
  Message* MutableMessage(Message* message, const FieldDescriptor* field,
                          MessageFactory* factory = NULL) const;
  void SetAllocatedMessage(Message* message,
                           Message* sub_message,
                           const FieldDescriptor* field) const;
  Message* ReleaseMessage(Message* message, const FieldDescriptor* field,
                          MessageFactory* factory = NULL) const;

  int32  GetRepeatedInt32 (const Message& message,
                           const FieldDescriptor* field, int index) const;
  int64  GetRepeatedInt64 (const Message& message,
                           const FieldDescriptor* field, int index) const;
  uint32 GetRepeatedUInt32(const Message& message,
                           const FieldDescriptor* field, int index) const;
  uint64 GetRepeatedUInt64(const Message& message,
                           const FieldDescriptor* field, int index) const;
  float  GetRepeatedFloat (const Message& message,
                           const FieldDescriptor* field, int index) const;
  double GetRepeatedDouble(const Message& message,
                           const FieldDescriptor* field, int index) const;
  bool   GetRepeatedBool  (const Message& message,
                           const FieldDescriptor* field, int index) const;
  string GetRepeatedString(const Message& message,
                           const FieldDescriptor* field, int index) const;
  const string& GetRepeatedStringReference(const Message& message,
                                           const FieldDescriptor* field,
                                           int index, string* scratch) const;
  const EnumValueDescriptor* GetRepeatedEnum(const Message& message,
                                             const FieldDescriptor* field,
                                             int index) const;
  int GetRepeatedEnumValue(const Message& message,
                           const FieldDescriptor* field,
                           int index) const;
  const Message& GetRepeatedMessage(const Message& message,
                                    const FieldDescriptor* field,
                                    int index) const;

  // Set the value of a field.
  void SetRepeatedInt32 (Message* message,
                         const FieldDescriptor* field, int index, int32  value) const;
  void SetRepeatedInt64 (Message* message,
                         const FieldDescriptor* field, int index, int64  value) const;
  void SetRepeatedUInt32(Message* message,
                         const FieldDescriptor* field, int index, uint32 value) const;
  void SetRepeatedUInt64(Message* message,
                         const FieldDescriptor* field, int index, uint64 value) const;
  void SetRepeatedFloat (Message* message,
                         const FieldDescriptor* field, int index, float  value) const;
  void SetRepeatedDouble(Message* message,
                         const FieldDescriptor* field, int index, double value) const;
  void SetRepeatedBool  (Message* message,
                         const FieldDescriptor* field, int index, bool   value) const;
  void SetRepeatedString(Message* message,
                         const FieldDescriptor* field, int index,
                         const string& value) const;
  void SetRepeatedEnum(Message* message, const FieldDescriptor* field,
                       int index, const EnumValueDescriptor* value) const;
  void SetRepeatedEnumValue(Message* message, const FieldDescriptor* field,
                            int index, int value) const;
  // Get a mutable pointer to a field with a message type.
  Message* MutableRepeatedMessage(Message* message,
                                  const FieldDescriptor* field,
                                  int index) const;

  void AddInt32 (Message* message,
                 const FieldDescriptor* field, int32  value) const;
  void AddInt64 (Message* message,
                 const FieldDescriptor* field, int64  value) const;
  void AddUInt32(Message* message,
                 const FieldDescriptor* field, uint32 value) const;
  void AddUInt64(Message* message,
                 const FieldDescriptor* field, uint64 value) const;
  void AddFloat (Message* message,
                 const FieldDescriptor* field, float  value) const;
  void AddDouble(Message* message,
                 const FieldDescriptor* field, double value) const;
  void AddBool  (Message* message,
                 const FieldDescriptor* field, bool   value) const;
  void AddString(Message* message,
                 const FieldDescriptor* field, const string& value) const;
  void AddEnum(Message* message,
               const FieldDescriptor* field,
               const EnumValueDescriptor* value) const;
  void AddEnumValue(Message* message,
                    const FieldDescriptor* field,
                    int value) const;
  Message* AddMessage(Message* message, const FieldDescriptor* field,
                      MessageFactory* factory = NULL) const;
  void AddAllocatedMessage(
      Message* message, const FieldDescriptor* field,
      Message* new_entry) const;

  const FieldDescriptor* FindKnownExtensionByName(const string& name) const;
  const FieldDescriptor* FindKnownExtensionByNumber(int number) const;

  bool SupportsUnknownEnumValues() const;

  // This value for arena_offset_ indicates that there is no arena pointer in
  // this message (e.g., old generated code).
  static const int kNoArenaPointer = -1;

  // This value for unknown_field_offset_ indicates that there is no
  // UnknownFieldSet in this message, and that instead, we are using the
  // Zero-Overhead Arena Pointer trick. When this is the case, arena_offset_
  // actually indexes to an InternalMetadataWithArena instance, which can return
  // either an arena pointer or an UnknownFieldSet or both. It is never the case
  // that unknown_field_offset_ == kUnknownFieldSetInMetadata && arena_offset_
  // == kNoArenaPointer.
  static const int kUnknownFieldSetInMetadata = -1;

 protected:
  void* MutableRawRepeatedField(
      Message* message, const FieldDescriptor* field, FieldDescriptor::CppType,
      int ctype, const Descriptor* desc) const;

  const void* GetRawRepeatedField(
      const Message& message, const FieldDescriptor* field,
      FieldDescriptor::CppType, int ctype,
      const Descriptor* desc) const;

  virtual MessageFactory* GetMessageFactory() const;

  virtual void* RepeatedFieldData(
      Message* message, const FieldDescriptor* field,
      FieldDescriptor::CppType cpp_type,
      const Descriptor* message_type) const;

 private:
  friend class GeneratedMessage;

  // To parse directly into a proto2 generated class, the class GMR_Handlers
  // needs access to member offsets and hasbits.
  friend class upb::google_opensource::GMR_Handlers;

  const Descriptor* descriptor_;
  const Message* default_instance_;
  const void* default_oneof_instance_;
  const int* offsets_;

  int has_bits_offset_;
  int oneof_case_offset_;
  int unknown_fields_offset_;
  int extensions_offset_;
  int arena_offset_;
  int is_default_instance_offset_;
  int object_size_;

  static const int kHasNoDefaultInstanceField = -1;

  const DescriptorPool* descriptor_pool_;
  MessageFactory* message_factory_;

  template <typename Type>
  inline const Type& GetRaw(const Message& message,
                            const FieldDescriptor* field) const;
  template <typename Type>
  inline Type* MutableRaw(Message* message,
                          const FieldDescriptor* field) const;
  template <typename Type>
  inline const Type& DefaultRaw(const FieldDescriptor* field) const;
  template <typename Type>
  inline const Type& DefaultOneofRaw(const FieldDescriptor* field) const;

  inline const uint32* GetHasBits(const Message& message) const;
  inline uint32* MutableHasBits(Message* message) const;
  inline uint32 GetOneofCase(
      const Message& message,
      const OneofDescriptor* oneof_descriptor) const;
  inline uint32* MutableOneofCase(
      Message* message,
      const OneofDescriptor* oneof_descriptor) const;
  inline const ExtensionSet& GetExtensionSet(const Message& message) const;
  inline ExtensionSet* MutableExtensionSet(Message* message) const;
  inline Arena* GetArena(Message* message) const;
  inline const internal::InternalMetadataWithArena&
      GetInternalMetadataWithArena(const Message& message) const;
  inline internal::InternalMetadataWithArena*
      MutableInternalMetadataWithArena(Message* message) const;

  inline bool GetIsDefaultInstance(const Message& message) const;

  inline bool HasBit(const Message& message,
                     const FieldDescriptor* field) const;
  inline void SetBit(Message* message,
                     const FieldDescriptor* field) const;
  inline void ClearBit(Message* message,
                       const FieldDescriptor* field) const;
  inline void SwapBit(Message* message1,
                      Message* message2,
                      const FieldDescriptor* field) const;

  // This function only swaps the field. Should swap corresponding has_bit
  // before or after using this function.
  void SwapField(Message* message1,
                 Message* message2,
                 const FieldDescriptor* field) const;

  void SwapOneofField(Message* message1,
                      Message* message2,
                      const OneofDescriptor* oneof_descriptor) const;

  inline bool HasOneofField(const Message& message,
                            const FieldDescriptor* field) const;
  inline void SetOneofCase(Message* message,
                           const FieldDescriptor* field) const;
  inline void ClearOneofField(Message* message,
                              const FieldDescriptor* field) const;

  template <typename Type>
  inline const Type& GetField(const Message& message,
                              const FieldDescriptor* field) const;
  template <typename Type>
  inline void SetField(Message* message,
                       const FieldDescriptor* field, const Type& value) const;
  template <typename Type>
  inline Type* MutableField(Message* message,
                            const FieldDescriptor* field) const;
  template <typename Type>
  inline const Type& GetRepeatedField(const Message& message,
                                      const FieldDescriptor* field,
                                      int index) const;
  template <typename Type>
  inline const Type& GetRepeatedPtrField(const Message& message,
                                         const FieldDescriptor* field,
                                         int index) const;
  template <typename Type>
  inline void SetRepeatedField(Message* message,
                               const FieldDescriptor* field, int index,
                               Type value) const;
  template <typename Type>
  inline Type* MutableRepeatedField(Message* message,
                                    const FieldDescriptor* field,
                                    int index) const;
  template <typename Type>
  inline void AddField(Message* message,
                       const FieldDescriptor* field, const Type& value) const;
  template <typename Type>
  inline Type* AddField(Message* message,
                        const FieldDescriptor* field) const;

  int GetExtensionNumberOrDie(const Descriptor* type) const;

  // Internal versions of EnumValue API perform no checking. Called after checks
  // by public methods.
  void SetEnumValueInternal(Message* message,
                            const FieldDescriptor* field,
                            int value) const;
  void SetRepeatedEnumValueInternal(Message* message,
                                    const FieldDescriptor* field,
                                    int index,
                                    int value) const;
  void AddEnumValueInternal(Message* message,
                            const FieldDescriptor* field,
                            int value) const;


  Message* UnsafeArenaReleaseMessage(Message* message,
                                     const FieldDescriptor* field,
                                     MessageFactory* factory = NULL) const;

  void UnsafeArenaSetAllocatedMessage(Message* message,
                                      Message* sub_message,
                                      const FieldDescriptor* field) const;

  internal::MapFieldBase* MapData(
      Message* message, const FieldDescriptor* field) const;

  GOOGLE_DISALLOW_EVIL_CONSTRUCTORS(GeneratedMessageReflection);
};

// Returns the offset of the given field within the given aggregate type.
// This is equivalent to the ANSI C offsetof() macro.  However, according
// to the C++ standard, offsetof() only works on POD types, and GCC
// enforces this requirement with a warning.  In practice, this rule is
// unnecessarily strict; there is probably no compiler or platform on
// which the offsets of the direct fields of a class are non-constant.
// Fields inherited from superclasses *can* have non-constant offsets,
// but that's not what this macro will be used for.
//
// Note that we calculate relative to the pointer value 16 here since if we
// just use zero, GCC complains about dereferencing a NULL pointer.  We
// choose 16 rather than some other number just in case the compiler would
// be confused by an unaligned pointer.
#define GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(TYPE, FIELD)    \
  static_cast<int>(                                           \
      reinterpret_cast<const char*>(                          \
          &reinterpret_cast<const TYPE*>(16)->FIELD) -        \
      reinterpret_cast<const char*>(16))

#define PROTO2_GENERATED_DEFAULT_ONEOF_FIELD_OFFSET(ONEOF, FIELD)     \
  static_cast<int>(                                                   \
      reinterpret_cast<const char*>(&(ONEOF->FIELD))                  \
      - reinterpret_cast<const char*>(ONEOF))

// There are some places in proto2 where dynamic_cast would be useful as an
// optimization.  For example, take Message::MergeFrom(const Message& other).
// For a given generated message FooMessage, we generate these two methods:
//   void MergeFrom(const FooMessage& other);
//   void MergeFrom(const Message& other);
// The former method can be implemented directly in terms of FooMessage's
// inline accessors, but the latter method must work with the reflection
// interface.  However, if the parameter to the latter method is actually of
// type FooMessage, then we'd like to be able to just call the other method
// as an optimization.  So, we use dynamic_cast to check this.
//
// That said, dynamic_cast requires RTTI, which many people like to disable
// for performance and code size reasons.  When RTTI is not available, we
// still need to produce correct results.  So, in this case we have to fall
// back to using reflection, which is what we would have done anyway if the
// objects were not of the exact same class.
//
// dynamic_cast_if_available() implements this logic.  If RTTI is
// enabled, it does a dynamic_cast.  If RTTI is disabled, it just returns
// NULL.
//
// If you need to compile without RTTI, simply #define GOOGLE_PROTOBUF_NO_RTTI.
// On MSVC, this should be detected automatically.
template<typename To, typename From>
inline To dynamic_cast_if_available(From from) {
#if defined(GOOGLE_PROTOBUF_NO_RTTI) || (defined(_MSC_VER)&&!defined(_CPPRTTI))
  return NULL;
#else
  return dynamic_cast<To>(from);
#endif
}

// Tries to downcast this message to a generated message type.
// Returns NULL if this class is not an instance of T.
//
// This is like dynamic_cast_if_available, except it works even when
// dynamic_cast is not available by using Reflection.  However it only works
// with Message objects.
//
// TODO(haberman): can we remove dynamic_cast_if_available in favor of this?
template <typename T>
T* DynamicCastToGenerated(const Message* from) {
  // Compile-time assert that T is a generated type that has a
  // default_instance() accessor, but avoid actually calling it.
  const T&(*get_default_instance)() = &T::default_instance;
  (void)get_default_instance;

  // Compile-time assert that T is a subclass of google::protobuf::Message.
  const Message* unused = static_cast<T*>(NULL);
  (void)unused;

#if defined(GOOGLE_PROTOBUF_NO_RTTI) || \
  (defined(_MSC_VER) && !defined(_CPPRTTI))
  bool ok = &T::default_instance() ==
            from->GetReflection()->GetMessageFactory()->GetPrototype(
                from->GetDescriptor());
  return ok ? down_cast<T*>(from) : NULL;
#else
  return dynamic_cast<T*>(from);
#endif
}

template <typename T>
T* DynamicCastToGenerated(Message* from) {
  const Message* message_const = from;
  return const_cast<T*>(DynamicCastToGenerated<const T>(message_const));
}

}  // namespace internal
}  // namespace protobuf

}  // namespace google
#endif  // GOOGLE_PROTOBUF_GENERATED_MESSAGE_REFLECTION_H__
