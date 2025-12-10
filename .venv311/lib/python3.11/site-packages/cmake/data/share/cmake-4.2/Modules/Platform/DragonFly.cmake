# DragonFly BSD was forked from FreeBSD and is still very close to it
# http://www.dragonflybsd.org
# see http://archive.netbsd.se/?ml=dfbsd-users&a=2007-07&m=4678361

include(Platform/FreeBSD)

# DragonFly BSD requires -z origin to enable $ORIGIN expansion in RPATH.
# This is not required for FreeBSD since 10.2-RELEASE.
set(CMAKE_SHARED_LIBRARY_RUNTIME_C_FLAG "-Wl,-z,origin,-rpath,")
