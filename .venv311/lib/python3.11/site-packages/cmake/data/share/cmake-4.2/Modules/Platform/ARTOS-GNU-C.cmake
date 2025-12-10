# Define ARTOS to select proper behavior and tell preprocessor to accept C++ style comments.
string(APPEND CMAKE_C_FLAGS_INIT " -DARTOS -Xp -+")
# ac doesn't support -g properly and doesn't support the normal gcc optimization options. Just use the defaults set by ac.
string(APPEND CMAKE_C_FLAGS_DEBUG_INIT " ")
string(APPEND CMAKE_C_FLAGS_MINSIZEREL_INIT " -DNDEBUG")
string(APPEND CMAKE_C_FLAGS_RELEASE_INIT " -DNDEBUG")
string(APPEND CMAKE_C_FLAGS_RELWITHDEBINFO_INIT " -DNDEBUG")
# Most projects expect the stdio functions to be available.
set(CMAKE_C_STANDARD_LIBRARIES_INIT "stdio.a")
