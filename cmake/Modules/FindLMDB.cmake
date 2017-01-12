# Try to find the LMBD libraries and headers
#  LMDB_FOUND - system has LMDB lib
#  LMDB_INCLUDE_DIR - the LMDB include directory
#  LMDB_LIBRARIES - Libraries needed to use LMDB

# FindCWD based on FindGMP by:
# Copyright (c) 2006, Laurent Montel, <montel@kde.org>
#
# Redistribution and use is allowed according to the terms of the BSD license.

# Adapted from FindCWD by:
# Copyright 2013 Conrad Steenberg <conrad.steenberg@gmail.com>
# Aug 31, 2013

find_path(LMDB_INCLUDE_DIR NAMES  lmdb.h PATHS "$ENV{LMDB_DIR}/include")
find_library(LMDB_LIBRARIES NAMES lmdb   PATHS "$ENV{LMDB_DIR}/lib" )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LMDB DEFAULT_MSG LMDB_INCLUDE_DIR LMDB_LIBRARIES)

if(LMDB_FOUND)
  message(STATUS "Found lmdb    (include: ${LMDB_INCLUDE_DIR}, library: ${LMDB_LIBRARIES})")
  mark_as_advanced(LMDB_INCLUDE_DIR LMDB_LIBRARIES)

  caffe_parse_header(${LMDB_INCLUDE_DIR}/lmdb.h
                     LMDB_VERSION_LINES MDB_VERSION_MAJOR MDB_VERSION_MINOR MDB_VERSION_PATCH)
  set(LMDB_VERSION "${MDB_VERSION_MAJOR}.${MDB_VERSION_MINOR}.${MDB_VERSION_PATCH}")
endif()
