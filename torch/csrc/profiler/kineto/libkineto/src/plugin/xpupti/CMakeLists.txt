# find xpupti sdk
find_package(Pti GLOBAL REQUIRED)
get_target_property(PTI_VIEW_LOCATION Pti::pti_view LOCATION)
message(STATUS "pti_view library location = ${PTI_VIEW_LOCATION}")
