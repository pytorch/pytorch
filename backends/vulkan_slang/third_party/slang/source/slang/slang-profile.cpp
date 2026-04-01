// slang-profile.cpp
#include "slang-profile.h"

namespace Slang
{

ProfileFamily getProfileFamily(ProfileVersion version)
{
    switch (version)
    {
    default:
        return ProfileFamily::Unknown;

#define PROFILE_VERSION(TAG, FAMILY) \
    case ProfileVersion::TAG:        \
        return ProfileFamily::FAMILY;
#include "slang-profile-defs.h"
    }
}

bool isRaytracingStage(Stage inStage)
{
    switch (inStage)
    {
    case Stage::RayGeneration:
    case Stage::Miss:
    case Stage::Intersection:
    case Stage::ClosestHit:
    case Stage::Callable:
    case Stage::AnyHit:
        return true;
    default:
        return false;
    }
}

const char* getStageName(Stage stage)
{
    switch (stage)
    {
#define PROFILE_STAGE(ID, NAME, ENUM) \
    case Stage::ID:                   \
        return #NAME;

#include "slang-profile-defs.h"

    default:
        return nullptr;
    }
}

void printDiagnosticArg(StringBuilder& sb, Stage val)
{
    sb << getStageName(val);
}

void printDiagnosticArg(StringBuilder& sb, ProfileVersion val)
{
    sb << Profile(val).getName();
}


} // namespace Slang
