// slang-fiddle-script.cpp
#include "slang-fiddle-script.h"

#include "../external/lua/lapi.h"
#include "../external/lua/lauxlib.h"
#include "../external/lua/lualib.h"

namespace fiddle
{
DiagnosticSink* _sink = nullptr;
StringBuilder* _builder = nullptr;
Count _templateCounter = 0;

void diagnoseLuaError(lua_State* L)
{
    size_t size = 0;
    char const* buffer = lua_tolstring(L, -1, &size);
    String message = UnownedStringSlice(buffer, size);
    message = message + "\n";
    if (_sink)
    {
        _sink->diagnoseRaw(Severity::Error, message.getBuffer());
    }
    else
    {
        fprintf(stderr, "%s", message.getBuffer());
    }
}

int _handleLuaError(lua_State* L)
{
    diagnoseLuaError(L);
    return lua_error(L);
}

int _original(lua_State* L)
{
    // We ignore the text that we want to just pass
    // through unmodified...
    return 0;
}

int _raw(lua_State* L)
{
    size_t size = 0;
    char const* buffer = lua_tolstring(L, 1, &size);

    _builder->append(UnownedStringSlice(buffer, size));
    return 0;
}

int _splice(lua_State* L)
{
    auto savedBuilder = _builder;

    StringBuilder spliceBuilder;
    _builder = &spliceBuilder;

    lua_pushvalue(L, 1);
    auto result = lua_pcall(L, 0, 1, 0);

    _builder = savedBuilder;

    if (result != LUA_OK)
    {
        return _handleLuaError(L);
    }

    // The actual string value follows whatever
    // got printed to the output (unless it is
    // nil).
    //
    _builder->append(spliceBuilder.produceString());
    if (!lua_isnil(L, -1))
    {
        size_t size = 0;
        char const* buffer = luaL_tolstring(L, -1, &size);
        _builder->append(UnownedStringSlice(buffer, size));
    }
    return 0;
}

int _template(lua_State* L)
{
    auto templateID = _templateCounter++;

    _builder->append("\n#if FIDDLE_GENERATED_OUTPUT_ID == ");
    _builder->append(templateID);
    _builder->append("\n");

    lua_pushvalue(L, 1);
    auto result = lua_pcall(L, 0, 0, 0);
    if (result != LUA_OK)
    {
        return _handleLuaError(L);
    }

    _builder->append("\n#endif\n");

    return 0;
}

lua_State* L = nullptr;

void ensureLuaInitialized()
{
    if (L)
        return;

    L = luaL_newstate();
    luaL_openlibs(L);

    lua_pushcclosure(L, &_original, 0);
    lua_setglobal(L, "ORIGINAL");

    lua_pushcclosure(L, &_raw, 0);
    lua_setglobal(L, "RAW");

    lua_pushcclosure(L, &_splice, 0);
    lua_setglobal(L, "SPLICE");

    lua_pushcclosure(L, &_template, 0);
    lua_setglobal(L, "TEMPLATE");

    // TODO: register custom stuff here...
}

lua_State* getLuaState()
{
    ensureLuaInitialized();
    return L;
}


String evaluateScriptCode(String originalFileName, String scriptSource, DiagnosticSink* sink)
{
    StringBuilder builder;
    _builder = &builder;
    _templateCounter = 0;

    ensureLuaInitialized();

    String luaChunkName = "@" + originalFileName;

    if (LUA_OK != luaL_loadbuffer(
                      L,
                      scriptSource.getBuffer(),
                      scriptSource.getLength(),
                      luaChunkName.getBuffer()))
    {
        size_t size = 0;
        char const* buffer = lua_tolstring(L, -1, &size);
        String message = UnownedStringSlice(buffer, size);
        message = message + "\n";
        sink->diagnoseRaw(Severity::Error, message.getBuffer());
        SLANG_ABORT_COMPILATION("fiddle failed during Lua script loading");
    }

    if (LUA_OK != lua_pcall(L, 0, 0, 0))
    {
        size_t size = 0;
        char const* buffer = lua_tolstring(L, -1, &size);
        String message = UnownedStringSlice(buffer, size);
        message = message + "\n";
        sink->diagnoseRaw(Severity::Error, message.getBuffer());
        SLANG_ABORT_COMPILATION("fiddle failed during Lua script execution");
    }

    _builder = nullptr;
    return builder.produceString();
}
} // namespace fiddle
