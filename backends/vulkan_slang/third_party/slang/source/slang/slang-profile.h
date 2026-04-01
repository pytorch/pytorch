#ifndef SLANG_PROFILE_H_INCLUDED
#define SLANG_PROFILE_H_INCLUDED

#include "../core/slang-basic.h"
#include "slang-capability.h"
#include "slang.h"

namespace Slang
{
// Flavors of translation unit
enum class SourceLanguage : SlangSourceLanguageIntegral
{
    Unknown = SLANG_SOURCE_LANGUAGE_UNKNOWN, // should not occur
    Slang = SLANG_SOURCE_LANGUAGE_SLANG,
    HLSL = SLANG_SOURCE_LANGUAGE_HLSL,
    GLSL = SLANG_SOURCE_LANGUAGE_GLSL,
    C = SLANG_SOURCE_LANGUAGE_C,
    CPP = SLANG_SOURCE_LANGUAGE_CPP,
    CUDA = SLANG_SOURCE_LANGUAGE_CUDA,
    SPIRV = SLANG_SOURCE_LANGUAGE_SPIRV,
    Metal = SLANG_SOURCE_LANGUAGE_METAL,
    WGSL = SLANG_SOURCE_LANGUAGE_WGSL,
    CountOf = SLANG_SOURCE_LANGUAGE_COUNT_OF,
};

// TODO(tfoley): This should merge with the above...
enum class Language
{
    Unknown,
#define LANGUAGE(TAG, NAME) TAG,
#include "slang-profile-defs.h"
};

enum class ProfileFamily
{
    Unknown,
#define PROFILE_FAMILY(TAG) TAG,
#include "slang-profile-defs.h"
};

enum class ProfileVersion
{
    Unknown,
#define PROFILE_VERSION(TAG, FAMILY) TAG,
#include "slang-profile-defs.h"
};


void printDiagnosticArg(StringBuilder& sb, ProfileVersion val);

enum class Stage : SlangStageIntegral
{
    Unknown = SLANG_STAGE_NONE,
#define PROFILE_STAGE(TAG, NAME, VAL) TAG = VAL,
#define PROFILE_STAGE_ALIAS(TAG, NAME, VAL) TAG = VAL,
#include "slang-profile-defs.h"
};

bool isRaytracingStage(Stage inStage);

const char* getStageName(Stage stage);

void printDiagnosticArg(StringBuilder& sb, Stage val);

ProfileFamily getProfileFamily(ProfileVersion version);

struct Profile
{
    typedef uint32_t RawVal;
    enum RawEnum : RawVal
    {
        Unknown,

#define PROFILE(TAG, NAME, STAGE, VERSION) \
    TAG = (uint32_t(ProfileVersion::VERSION) << 16) | uint32_t(Stage::STAGE),
#define PROFILE_ALIAS(TAG, DEF, NAME) TAG = DEF,
#include "slang-profile-defs.h"
    };

    Profile() {}
    Profile(RawEnum raw)
        : raw(raw)
    {
    }
    explicit Profile(RawVal raw)
        : raw(raw)
    {
    }
    explicit Profile(Stage stage) { setStage(stage); }
    explicit Profile(ProfileVersion version) { setVersion(version); }

    bool operator==(Profile const& other) const { return raw == other.raw; }
    bool operator!=(Profile const& other) const { return raw != other.raw; }

    Stage getStage() const { return Stage(uint32_t(raw) & 0xFFFF); }
    void setStage(Stage stage) { raw = (raw & ~0xFFFF) | uint32_t(stage); }

    ProfileVersion getVersion() const { return ProfileVersion((uint32_t(raw) >> 16) & 0xFFFF); }
    void setVersion(ProfileVersion version)
    {
        raw = (raw & 0x0000FFFF) | (uint32_t(version) << 16);
    }

    ProfileFamily getFamily() const { return getProfileFamily(getVersion()); }

    static Profile lookUp(UnownedStringSlice const& name);
    static Profile lookUp(char const* name);
    char const* getName();

    CapabilitySet getCapabilityName();

    List<CapabilitySet> additionalCapabilities;

    RawVal raw = Unknown;
};

struct StageInfo
{
    const char* name;
    Stage stage;
};

/// Note! There can be multiple stages with the same name
ConstArrayView<StageInfo> getStageInfos();

Stage findStageByName(String const& name);

UnownedStringSlice getStageText(Stage stage);

Stage getStageFromAtom(CapabilityAtom atom);
} // namespace Slang

#endif
