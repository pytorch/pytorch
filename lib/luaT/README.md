<a name="luat.dok"/>
# Lua Torch C API #

luaT provides an API to interface Lua and C in Torch packages. It defines a
concept of _classes_ to Lua for Torch, and provides a mechanism to easily
handle these Lua classes from C.

It additionally provides few functions that `luaL` should have defined, and
defines several functions similar to `luaL` ones for better type error printing when using
`luaT` classes.

<a name="luat.memory.dok"/>
## Memory functions ##

Classical memory allocation functions which generate a Lua error in case of
problem.

<a name="luaT_alloc"/>
### void* luaT_alloc(lua_State *L, long size) ###

Allocates `size` bytes, and return a pointer on the allocated
memory. A Lua error will be generated if running out of memory.

<a name="luaT_realloc"/>
### void* luaT_realloc(lua_State *L, void *ptr, long size) ###

Realloc `ptr` to `size` bytes. `ptr` must have been previously
allocated with [luaT_alloc](#luaT_alloc) or
[luaT_realloc](#luaT_realloc), or the C `malloc` or `realloc`
functions. A Lua error will be generated if running out of memory.

<a name="luaT_free"/>
### void luaT_free(lua_State *L, void *ptr) ###

Free memory allocated at address `ptr`. The memory must have been
previously allocated with [luaT_alloc](#luaT_alloc) or
[luaT_realloc](#luaT_realloc), or the C `malloc` or `realloc`
functions.

<a name="luat.classcreate"/>
## Class creation and basic handling ##

A `luaT` class is basically either a Lua _table_ or _userdata_ with
an appropriate _metatable_. This appropriate metatable is created with
[luaT_newmetatable](#luaT_newmetatable). Contrary to luaL userdata
functions, luaT mechanism handles inheritance. If the class inherit from
another class, then the metatable will itself have a metatable
corresponding to the _parent metatable_: the metatables are cascaded
according to the class inheritance. Multiple inheritance is not supported.

<a name="luat.operatoroverloading"/>
### Operator overloading ###

The metatable of a `luaT` object contains `Lua` operators like
`__index`, `__newindex`, `__tostring`, `__add`
(etc...). These operators will respectively look for `__index__`,
`__newindex__`, `__tostring__`, `__add__` (etc...) in the
metatable. If found, the corresponding function or value will be returned,
else a Lua error will be raised.

If one wants to provide `__index__` or `__newindex__` in the
metaclass, these operators must follow a particular scheme:

  * `__index__` must either return a value _and_ `true` or return `false` only. In the first case, it means `__index__` was able to handle the given argument (for e.g., the type was correct). The second case means it was not able to do anything, so `__index` in the root metatable can then try to see if the metaclass contains the required value.

  * `__newindex__` must either return `true` or `false`. As for `__index__`, `true` means it could handle the argument and `false` not. If not, the root metatable `__newindex` will then raise an error if the object was a userdata, or apply a rawset if the object was a Lua table.

Other metaclass operators like `__tostring__`, `__add__`, etc... do not have any particular constraint.

<a name="luat_newmetatable"/>
### const char* luaT_newmetatable(lua_State *L, const char *tname, const char *parenttname, lua_CFunction constructor, lua_CFunction destructor, lua_CFunction factory) ###

This function creates a new metatable, which is the Lua way to define a new
object class. As for `luaL_newmetatable`, the metatable is registered in
the Lua registry table, with the key `tname`. In addition, `tname` is
also registered in the Lua registry, with the metatable as key (the
typename of a given object can be thus easily retrieved).

The class name `tname` must be of the form `modulename.classname`. The module name
If not NULL, `parenttname` must be a valid typename corresponding to the
parent class of the new class.

If not NULL, `constructor`, a function `new` will be added to the metatable, pointing to this given function. The constructor might also
be called through `modulename.classname()`, which is an alias setup by `luaT_metatable`.

If not NULL, `destructor` will be called when garbage collecting the object.

If not NULL, `factory` must be a Lua C function creating an empty object
instance of the class. This functions are used in Torch for serialization.

Note that classes can be partly defined in C and partly defined in Lua:
once the metatable is created in C, it can be filled up with additional
methods in Lua.

The return value is the value returned by [luaT_typenameid](#luat_typenameid).

<a name="luat_pushmetatable"/>
### int luaT_pushmetatable(lua_State *L, const name *tname) ###

Push the metatable with type name `tname` on the stack, it `tname` is a
valid Torch class name (previously registered with luaT_newmetatable).

On success, returns 1. If `tname` is invalid, nothing is pushed and it
returns 0.

<a name="luat_typenameid"/>
### const char* luaT_typenameid(lua_State *L, const char *tname) ###

If `tname` is a valid Torch class name, then returns a unique string (the
contents will be the same than `tname`) pointing on the string registered
in the Lua registry. This string is thus valid as long as Lua is
running. The returned string shall not be freed.

If `tname` is an invalid class name, returns NULL.

<a name="luat_typename"/>
### const char* luaT_typename(lua_State *L, int ud) ###

Returns the typename of the object at index `ud` on the stack. If it is
not a valid Torch object, returns NULL.

<a name="luat_pushudata"/>
### void luaT_pushudata(lua_State *L, void *udata, const char *tname) ###

Given a C structure `udata`, push a userdata object on the stack with
metatable corresponding to `tname`. Obviously, `tname` must be a valid
Torch name registered with [luaT_newmetatable](#luat_newmetatable).

<a name="luat_toudata"/>
### void *luaT_toudata(lua_State *L, int ud, const char *tname) ###

Returns a pointer to the original C structure previously pushed on the
stack with [luaT_pushudata](#luat_pushudata), if the object at index
`ud` is a valid Torch class name. Returns NULL otherwise.

<a name="luat_isudata"/>
### int luaT_isudata(lua_State *L, int ud, const char *tname) ###

Returns 1 if the object at index `ud` on the stack is a valid Torch class name `tname`.
Returns 0 otherwise.

<a name="luat_getfield"/>
### Checking fields of a table ###

This functions check that the table at the given index `ud` on the Lua
stack has a field named `field`, and that it is of the specified type.
These function raises a Lua error on failure.

<a name="luat_getfieldcheckudata"/>
## void *luaT_getfieldcheckudata(lua_State *L, int ud, const char *field, const char *tname) ##

Checks that the field named `field` of the table at index `ud` is a
Torch class name `tname`.  Returns the pointer of the C structure
previously pushed on the stack with [luaT_pushudata](#luat_pushudata) on
success. The function raises a Lua error on failure.

<a name="luat_getfieldchecklightudata"/>
## void *luaT_getfieldchecklightudata(lua_State *L, int ud, const char *field) ##

Checks that the field named `field` of the table at index `ud` is a
lightuserdata.  Returns the lightuserdata pointer on success. The function
raises a Lua error on failure.

<a name="luat_getfieldcheckint"/>
## int luaT_getfieldcheckint(lua_State *L, int ud, const char *field) ##

Checks that the field named `field` of the table at index `ud` is an
int. Returns the int value pointer on success. The function raises a Lua
error on failure.

<a name="luat_getfieldcheckstring"/>
## const char* luaT_getfieldcheckstring(lua_State *L, int ud, const char *field) ##

Checks that the field named `field` of the table at index `ud` is a
string. Returns a pointer to the string on success. The function raises a
Lua error on failure.

<a name="luat_getfieldcheckboolean"/>
## int luaT_getfieldcheckboolean(lua_State *L, int ud, const char *field) ##

Checks that the field named `field` of the table at index `ud` is a
boolean. On success, returns 1 if the boolean is `true`, 0 if it is
`false`. The function raises a Lua error on failure.

<a name="luat_getfieldchecktable"/>
## void luaT_getfieldchecktable(lua_State *L, int ud, const char *field) ##

Checks that the field named `field` of the table at index `ud` is a
table. On success, push the table on the stack. The function raises a Lua
error on failure.

<a name="luat_typerror"/>
### int luaT_typerror(lua_State *L, int ud, const char *tname) ###

Raises a `luaL_argerror` (and returns its value), claiming that the
object at index `ud` on the stack is not of type `tname`. Note that
this function does not check the type, it only raises an error.

<a name="luat_checkboolean"/>
### int luaT_checkboolean(lua_State *L, int ud) ###

Checks that the value at index `ud` is a boolean. On success, returns 1
if the boolean is `true`, 0 if it is `false`. The function raises a Lua
error on failure.

<a name="luat_optboolean"/>
### int luaT_optboolean(lua_State *L, int ud, int def) ###

Checks that the value at index `ud` is a boolean. On success, returns 1
if the boolean is `true`, 0 if it is `false`. If there is no value at
index `ud`, returns `def`. In any other cases, raises an error.

<a name="luat_registeratname"/>
### void luaT_registeratname(lua_State *L, const struct luaL_Reg *methods, const char *name) ###

This function assume a table is on the stack. It creates a table field
`name` in the table (if this field does not exist yet), and fill up
`methods` in this table field.

<a name="luat_classrootname"/>
### const char *luaT_classrootname(const char *tname) ###

Assuming `tname` is of the form `modulename.classname`, returns
`classname`. The returned value shall not be freed. It is a pointer
inside `tname` string.

<a name="luat_classmodulename"/>
### const char *luaT_classmodulename(const char *tname) ###

Assuming `tname` is of the form `modulename.classname`, returns
`modulename`. The returned value shall not be freed. It is valid until the
next call to `luaT_classrootname`.

<a name="luat_stackdump"/>
### void luaT_stackdump(lua_State *L) ###

This function print outs the state of the Lua stack. It is useful for debug
purposes.

