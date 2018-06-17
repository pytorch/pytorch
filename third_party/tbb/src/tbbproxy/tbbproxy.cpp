/*
    Copyright (c) 2005-2018 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.




*/

#include "tbb/tbb_config.h"
#if !__TBB_WIN8UI_SUPPORT
#define TBB_PREVIEW_RUNTIME_LOADER 1
#include "tbb/runtime_loader.h"
#include "tbb/tbb_stddef.h"

// C standard headers.
#include <cctype>            // isspace
#include <cstdarg>           // va_list, etc.
#include <cstdio>            // fprintf, stderr, etc.
#include <cstdlib>           // malloc, free, abort.
#include <cstring>           // strlen, etc.

// C++ standard headers.
#include <typeinfo>

// OS-specific includes.
#if _WIN32 || _WIN64
    #include <windows.h>
    #define snprintf _snprintf
    #undef max
#else
    #include <dlfcn.h>    // dlopen, dlsym, dlclose, dlerror.
#endif

#if TBB_USE_ASSERT
    // We cannot use __TBB_ASSERT as it is because it calls a function from tbb library which may
    // be not yet loaded. Redefine __TBB_ASSERT not to call tbb functions.
    #undef __TBB_ASSERT
    #define __TBB_ASSERT( cond, msg ) {                                                            \
        if ( ! (cond) ) {                                                                          \
            say( "%s:%d: Assertion failed: %s.", __FILE__, __LINE__, (msg) );                      \
        } /* if */                                                                                 \
        /* TODO: abort? */                                                                         \
    }
#endif

// Declare here, define at the bottom.
extern "C" int __tbb_internal_runtime_loader_stub();

namespace tbb {

namespace interface6 {

namespace internal {

namespace runtime_loader {


/*
    ------------------------------------------------------------------------------------------------
    User interaction utilities.
    ------------------------------------------------------------------------------------------------
*/


// Print message to stderr. Do not call it directly, use say() or tell() instead.
static void _say( char const * format, va_list args ) {
    /*
        On 64-bit Linux* OS, vsnprintf() modifies args argument,
        so vsnprintf() crashes if it is called for the second time with the same args.
        To prevent the crash, we have to pass a fresh intact copy of args to vsnprintf() each time.

        On Windows* OS, unfortunately, standard va_copy() macro is not available. However, it
        seems vsnprintf() does not modify args argument.
    */
    #if ! ( _WIN32 || _WIN64 )
        va_list _args;
        __va_copy( _args, args );  // Make copy of args.
        #define args _args         // Substitute args with its copy, _args.
    #endif
    int len = vsnprintf( NULL, 0, format, args );
    #if ! ( _WIN32 || _WIN64 )
        #undef args                // Remove substitution.
        va_end( _args );
    #endif
    char * buf = reinterpret_cast< char * >( malloc( len + 1 ) );
    if ( buf != NULL ) {
        vsnprintf( buf, len + 1, format, args );
        fprintf( stderr, "TBB: %s\n", buf );
        free( buf );
    } else {
        fprintf( stderr, "TBB: Not enough memory for message: %s\n", format );
    }
} // _say


// Debug/test/troubleshooting printing controlled by TBB_VERSION environment variable.
// To enable printing, the variable must be set and not empty.
// Do not call it directly, use tell() instead.
static void _tell( char const * format, va_list args ) {
    char const * var = getenv( "TBB_VERSION" );
    if ( var != NULL && var[ 0 ] != 0 ) {
        _say( format, args );
    } // if
} // _tell


// Print message to stderr unconditionally.
static void say( char const * format, ... ) {
    va_list args;
    va_start( args, format );
    _say( format, args );
    va_end( args );
} // say


// Debug/test/troubleshooting printing controlled by TBB_VERSION environment variable.
// To enable printing, the variable must be set and not empty.
static void tell( char const * format, ... ) {
    va_list args;
    va_start( args, format );
    _tell( format, args );
    va_end( args );
} // tell


// Error reporting utility. Behavior depends on mode.
static tbb::runtime_loader::error_code error( tbb::runtime_loader::error_mode mode, tbb::runtime_loader::error_code err, char const * format, ... ) {
    va_list args;
    va_start( args, format );
    if ( mode == tbb::runtime_loader::em_abort ) {
        // In em_abort mode error message printed unconditionally.
        _say( format, args );
    } else {
        // In other modes printing depends on TBB_VERSION environment variable.
        _tell( format, args );
    } // if
    va_end( args );
    switch ( mode ) {
        case tbb::runtime_loader::em_abort : {
            say( "Aborting..." );
            #if TBB_USE_DEBUG && ( _WIN32 || _WIN64 )
                DebugBreak();
            #endif
            abort();
        } break;
        case tbb::runtime_loader::em_throw : {
            throw err;
        } break;
        case tbb::runtime_loader::em_status : {
            // Do nothing.
        } break;
    } // switch
    return err;
} // error


/*
    ------------------------------------------------------------------------------------------------
    General-purpose string manipulation utilities.
    ------------------------------------------------------------------------------------------------
*/


// Delete character ch from string str in-place.
static void strip( char * str, char ch ) {
    int in  = 0;  // Input character index.
    int out = 0;  // Output character index.
    for ( ; ; ) {
        if ( str[ in ] != ch ) {
            str[ out ] = str[ in ];
            ++ out;
        } // if
        if ( str[ in ] == 0 ) {
            break;
        } // if
        ++ in;
    } // forever
} // func strip


// Strip trailing whitespaces in-place.
static void trim( char * str ) {
    size_t len = strlen( str );
    while ( len > 0 && isspace( str[ len - 1 ] ) ) {
        -- len;
    } // while
    str[ len ] = 0;
} // func trim


#if _WIN32 || _WIN64
    // "When specifying a path, be sure to use backslashes (\), not forward slashes (/)."
    // (see http://msdn.microsoft.com/en-us/library/ms886736.aspx).
    const char proper_slash = '\\';
    inline char char_or_slash( char c ) { return c=='/'? '\\': c; }
#else
    const char proper_slash = '/';
    inline char char_or_slash( char c ) { return c; }
#endif

// Concatenate name of directory and name of file.
void cat_file( char const * dir, char const * file, char * buffer, size_t len ) {
    size_t i = 0;
    // Copy directory name
    for( ; i<len && *dir; ++i, ++dir ) {
        buffer[i] = char_or_slash(*dir);
    }
    // Append trailing slash if missed.
    if( i>0 && i<len && buffer[i-1]!=proper_slash ) {
        buffer[i++] = proper_slash;
    }
    // Copy file name
    __TBB_ASSERT( char_or_slash(*file)!=proper_slash, "File name starts with a slash" );
    for( ; i<len && *file; ++i, ++file ) {
        buffer[i] = *file;
    }
    // Append null terminator
    buffer[ i<len? i: len-1 ] = '\0';
} // cat_file


/*
    ------------------------------------------------------------------------------------------------
    Windows implementation of dlopen, dlclose, dlsym, dlerror.
    ------------------------------------------------------------------------------------------------
*/


#if _WIN32 || _WIN64

    // Implement Unix-like interface (dlopen, dlclose, dlsym, dlerror) via Win32 API functions.

    // Type of dlopen result.
    typedef HMODULE handle_t;

    enum rtld_flags_t {
        RTLD_NOW,
        RTLD_GLOBAL
    }; // enum rtld_flags_t

    // Unix-like dlopen().
    static handle_t dlopen( char const * name, rtld_flags_t ) {
        return LoadLibrary( name );
    } // dlopen

    // Unix-like dlsym().
    static void * dlsym( handle_t lib, char const * sym ) {
        return (void*)GetProcAddress( lib, sym );
    } // dlsym

    // Unix-like dlclose().
    static int dlclose( handle_t lib ) {
        return ! FreeLibrary( lib );
    } // dlclose

    // The function mimics Unix dlerror() function.
    // Note: Not thread-safe due to statically allocated buffer.
    static char * dlerror() {

        static char buffer[ 2048 ];  // Note: statically allocated buffer.

        DWORD err = GetLastError();
        if ( err == ERROR_SUCCESS ) {
            return NULL;
        } // if

        DWORD rc;
        rc =
            FormatMessage(
                FORMAT_MESSAGE_FROM_SYSTEM,
                NULL,
                err,
                MAKELANGID( LANG_NEUTRAL, SUBLANG_DEFAULT ), // Default language.
                reinterpret_cast< LPTSTR >( & buffer ),
                sizeof( buffer ),
                NULL
            );
        if ( rc == 0 ) {
            // FormatMessage() failed to format system error message. Buffer to short or another issue.
            snprintf( buffer, sizeof( buffer ), "System error %u.", err );
        } else {
            /*
                FormatMessage() returns Windows-style end-of-lines, "\r\n". When string is printed,
                printf() also replaces all the occurrences of "\n" with "\r\n" (again!), so sequences
                like "\r\r\r\n" appear in output. It is not too good. Stripping all "\r" normalizes
                string and returns it to canonical form, so printf() will produce correct end-of-line
                sequences.
            */
            strip( buffer, '\r' );   // Delete carriage returns if any.
            trim( buffer );          // Delete trailing newlines and spaces.
        } // if

        return buffer;

    } // dlerror

#else

    // Type of dlopen() result.
    typedef void * handle_t;

#endif


/*
    ------------------------------------------------------------------------------------------------
    Runtime loader stuff.
    ------------------------------------------------------------------------------------------------
*/


// Descriptor table declaration. It is defined in assembler file.
enum symbol_type_t {
    st_object   = 0,
    st_function = 1
}; // enum symbol_type_t
struct symbol_t {
    void *        addr;
    char const *  name;
    int           size;
    symbol_type_t type;
}; // symbol_t
extern "C" symbol_t __tbb_internal_runtime_loader_symbols[];

// Hooks for internal use (e. g. for testing).
tbb::runtime_loader::error_mode stub_mode = tbb::runtime_loader::em_abort;

static char const * tbb_dll_name = __TBB_STRING(__TBB_DLL_NAME);  // Name of TBB library.
static handle_t     handle       = NULL;                          // Handle of loaded TBB library or NULL.
static int          version      = 0;                             // Version of the loaded library.
static int          counter      = 0;                             // Number of runtime_loader objects using the loaded library.

#define ANOTHER_RTL "probably multiple runtime_loader objects work in parallel"


// One attempt to load library (dll_name can be a full path or just a file name).
static tbb::runtime_loader::error_code _load( char const * dll_name, int min_ver, int max_ver ) {

    tbb::runtime_loader::error_mode mode = tbb::runtime_loader::em_status;
    tbb::runtime_loader::error_code code = tbb::runtime_loader::ec_ok;

    /*
        If these variables declared at the first usage, Intel C++ Compiler may issue warning(s):
            transfer of control [goto error] bypasses initialization of: ...
        Declaring variables at the beginning of the function eliminates warnings.
    */
    typedef int (*int_func_t)( void );
    char const * get_ver_name = "TBB_runtime_interface_version"; // Name of function.
    int_func_t   get_ver_func = NULL;                            // Pointer to function.
    handle_t     _handle      = NULL;
    int          _version     = 0;
    int          total        = 0;
    int          not_found    = 0;

    // This function should be called iff there is no loaded library.
    __TBB_ASSERT( handle  == NULL, "Handle is invalid; "  ANOTHER_RTL );
    __TBB_ASSERT( version == 0,    "Version is invalid; " ANOTHER_RTL );
    __TBB_ASSERT( counter == 0,    "Counter is invalid; " ANOTHER_RTL );

    tell( "Loading \"%s\"...", dll_name );

    // First load the library.
    _handle = dlopen( dll_name, RTLD_NOW );
    if ( _handle == NULL ) {
        const char * msg = dlerror();
        code = error( mode, tbb::runtime_loader::ec_no_lib, "Loading \"%s\" failed; system error: %s", dll_name, msg );
        goto error;
    } // if

    // Then try to find out its version.
    /*
        g++ 3.4 issues error:
            ISO C++ forbids casting between pointer-to-function and pointer-to-object
        on reinterpret_cast<>. Thus, we have no choice but using C-style type cast.
    */
    get_ver_func = (int_func_t) dlsym( _handle, get_ver_name );
    if ( get_ver_func == NULL ) {
        code = error( mode, tbb::runtime_loader::ec_bad_lib, "Symbol \"%s\" not found; library rejected.", get_ver_name );
        goto error;
    } // if
    _version = get_ver_func();
    if ( ! ( min_ver <= _version && _version <= max_ver ) ) {
        code = error( mode, tbb::runtime_loader::ec_bad_ver, "Version %d is out of requested range; library rejected.", _version );
        goto error;
    } // if

    // Library is suitable. Mark it as loaded.
    handle   = _handle;
    version  = _version;
    counter += 1;
    __TBB_ASSERT( counter == 1, "Counter is invalid; " ANOTHER_RTL );

    // Now search for all known symbols.
    for ( int i = 0; __tbb_internal_runtime_loader_symbols[ i ].name != NULL; ++ i ) {
        symbol_t & symbol = __tbb_internal_runtime_loader_symbols[ i ];
        // Verify symbol descriptor.
        __TBB_ASSERT( symbol.type == st_object || symbol.type == st_function, "Invalid symbol type" );
        #if _WIN32 || _WIN64
            __TBB_ASSERT( symbol.type == st_function, "Should not be symbols of object type on Windows" );
        #endif
        if ( symbol.type == st_object ) {
            __TBB_ASSERT( symbol.addr != NULL, "Object address invalid" );
            __TBB_ASSERT( symbol.size > 0, "Symbol size must be > 0" );
            __TBB_ASSERT( symbol.size <= 0x1000, "Symbol size too big" );
        } else {                     // Function
            // __TBB_ASSERT( symbol.addr == reinterpret_cast< void * >( & stub ), "Invalid symbol address" );
            __TBB_ASSERT( symbol.size == sizeof( void * ), "Invalid symbol size" );
        } // if
        void * addr = dlsym( _handle, symbol.name );
        if ( addr != NULL ) {
            if ( symbol.type == st_object ) {
                if ( strncmp( symbol.name, "_ZTS", 4 ) == 0 ) {
                    // If object name begins with "_ZTS", it is a string, mangled type name.
                    // Its value must equal to name of symbol without "_ZTS" prefix.
                    char const * name = static_cast< char const * >( addr );
                    __TBB_ASSERT( strlen( name ) + 1 == size_t( symbol.size ), "Unexpected size of typeinfo name" );
                    __TBB_ASSERT( strcmp( symbol.name + 4, name ) == 0, "Unexpected content of typeinfo name" );
                    strncpy( reinterpret_cast< char * >( symbol.addr ), name, symbol.size );
                    reinterpret_cast< char * >( symbol.addr )[ symbol.size - 1 ] = 0;
                } else {
                    #if TBB_USE_ASSERT
                        // If object name begins with "_ZTI", it is an object of std::type_info class.
                        // Its protected value must equal to name of symbol without "_ZTI" prefix.
                        if ( strncmp( symbol.name, "_ZTI", 4 ) == 0 ) {
                            std::type_info const * info = static_cast< std::type_info const * >( addr );
                            __TBB_ASSERT( size_t( symbol.size ) >= sizeof( std::type_info ), "typeinfo size is too small" );
                            // std::type_info::name is not a virtual method, it is safe to call it.
                            __TBB_ASSERT( strcmp( symbol.name + 4, info->name() ) == 0, "Unexpected content of typeinfo" );
                        } // if
                    #endif
                    // Copy object content from libtbb into runtime_loader.
                    memcpy( symbol.addr, addr, symbol.size );
                }; // if
            } else {                     // Function
                symbol.addr = addr;
            } // if
        } else {
            char const * msg = dlerror();
            tell( "Symbol \"%s\" not found; system error: %s", symbol.name, msg );
            ++ not_found;
        } // if
        ++ total;
    } // for i

    if ( not_found > 0 ) {
        tell( "%d of %d symbols not found.", not_found, total );
    } // if

    tell( "The library successfully loaded." );
    return code;

    error:
        if ( _handle != NULL ) {
            int rc = dlclose( _handle );
            if ( rc != 0 ) {
                // Error occurred.
                __TBB_ASSERT( rc != 0, "Unexpected error: dlclose() failed" );
            } // if
        } // if
        _handle = NULL;
        return code;

} // _load


static tbb::runtime_loader::error_code load( tbb::runtime_loader::error_mode mode, char const * path[], int min_ver, int max_ver ) {
    // Check arguments first.
    if ( min_ver <= 0 ) {
        return error( mode, tbb::runtime_loader::ec_bad_arg, "tbb::runtime_loader::load(): Invalid value of min_ver argument: %d.", min_ver );
    } // if
    if ( max_ver <= 0 ) {
        return error( mode, tbb::runtime_loader::ec_bad_arg, "tbb::runtime_loader::load(): Invalid value of max_ver argument: %d.", max_ver );
    } // if
    if ( min_ver > max_ver ) {
        return error( mode, tbb::runtime_loader::ec_bad_arg, "tbb::runtime_loader::load(): min_ver and max_ver specify empty range: [%d, %d].", min_ver, max_ver );
    } // if
    if ( min_ver == max_ver ) {
        tell( "Searching for \"%s\" version %d...", tbb_dll_name, min_ver );
    } else if ( max_ver == INT_MAX ) {
        tell( "Searching for \"%s\" version %d+...", tbb_dll_name, min_ver );
    } else {
        tell( "Searching for \"%s\" version in range [%d, %d]...", tbb_dll_name, min_ver, max_ver );
    } // if
    // Then check whether a library already loaded.
    if ( handle != NULL ) {
        // Library already loaded. Check whether the version is compatible.
        __TBB_ASSERT( version > 0, "Version is invalid; " ANOTHER_RTL );
        __TBB_ASSERT( counter > 0, "Counter is invalid; " ANOTHER_RTL );
        if ( min_ver <= version && version <= max_ver ) {
            // Version is ok, let us use this library.
            tell( "Library version %d is already loaded.", version );
            counter += 1;
            return tbb::runtime_loader::ec_ok;
        } else {
            // Version is not suitable.
            return error( mode, tbb::runtime_loader::ec_bad_ver, "Library version %d is already loaded.", version );
        } // if
    } // if
    // There is no loaded library, try to load it using provided directories.
    __TBB_ASSERT( version == 0, "Version is invalid; " ANOTHER_RTL );
    __TBB_ASSERT( counter == 0, "Counter is invalid; " ANOTHER_RTL );
    size_t namelen = strlen(tbb_dll_name);
    size_t buflen = 0;
    char * buffer = NULL;
    for ( int i = 0; path[i] != NULL; ++ i ) {
        size_t len = strlen(path[i]) + namelen + 2; // 1 for slash and 1 for null terminator
        if( buflen<len ) {
            free( buffer );
            buflen = len;
            buffer = (char*)malloc( buflen );
            if( !buffer )
                return error( mode, tbb::runtime_loader::ec_no_lib, "Not enough memory." );
        }
        cat_file( path[i], tbb_dll_name, buffer, buflen );
        __TBB_ASSERT(strstr(buffer,tbb_dll_name), "Name concatenation error");
        tbb::runtime_loader::error_code ec = _load( buffer, min_ver, max_ver );
        if ( ec == tbb::runtime_loader::ec_ok ) {
            return ec;       // Success. Exiting...
        } // if
    } // for i
    free( buffer );
    return error( mode, tbb::runtime_loader::ec_no_lib, "No suitable library found." );
} // load




// Suppress "defined but not used" compiler warnings.
static void const * dummy[] = {
    (void *) & strip,
    (void *) & trim,
    & dummy,
    NULL
};


} // namespace runtime_loader

} // namespace internal


runtime_loader::runtime_loader( error_mode mode ) :
    my_mode( mode ),
    my_status( ec_ok ),
    my_loaded( false )
{
} // ctor


runtime_loader::runtime_loader( char const * path[], int min_ver, int max_ver, error_mode mode ) :
    my_mode( mode ),
    my_status( ec_ok ),
    my_loaded( false )
{
    load( path, min_ver, max_ver );
} // ctor


runtime_loader::~runtime_loader() {
} // dtor


tbb::runtime_loader::error_code runtime_loader::load( char const * path[], int min_ver, int max_ver ) {
    if ( my_loaded ) {
        my_status = tbb::interface6::internal::runtime_loader::error( my_mode, ec_bad_call, "tbb::runtime_loader::load(): Library already loaded by this runtime_loader object." );
    } else {
        my_status = internal::runtime_loader::load( my_mode, path, min_ver, max_ver );
        if ( my_status == ec_ok ) {
            my_loaded = true;
        } // if
    } // if
    return my_status;
} // load




tbb::runtime_loader::error_code runtime_loader::status() {
    return my_status;
} // status


} // namespace interface6

} // namespace tbb


// Stub function replaces all TBB entry points when no library is loaded.
int __tbb_internal_runtime_loader_stub() {
    char const * msg = NULL;
    if ( tbb::interface6::internal::runtime_loader::handle == NULL ) {
        msg = "A function is called while TBB library is not loaded";
    } else {
        msg = "A function is called which is not present in loaded TBB library";
    } // if
    return tbb::interface6::internal::runtime_loader::error( tbb::interface6::internal::runtime_loader::stub_mode, tbb::runtime_loader::ec_no_lib, msg );
} // stub

#endif // !__TBB_WIN8UI_SUPPORT //
// end of file //
