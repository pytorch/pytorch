<a name="torch.utility.dok"/>
# Torch utility functions #

This functions are used in all Torch package for creating and handling classes.
The most interesting function is probably [torch.class()](#torch.class) which allows
the user to create easily new classes. [torch.typename()](#torch.typename) might
also be interesting to check what is the class of a given Torch object.

The other functions are more for advanced users.

<a name="torch.class"/>
### [metatable] torch.class(name, [parentName]) ###

Creates a new `Torch` class called `name`. If `parentName` is provided, the class will inherit
`parentName` methods. A class is a table which has a particular metatable.

If `name` is of the form `package.className` then the class `className` will be added to the specified `package`.
In that case, `package` has to be a valid (and already loaded) package. If `name` does not contain any `"."`,
then the class will be defined in the global environment.

One [or two] (meta)tables are returned. These tables contain all the method
provided by the class [and its parent class if it has been provided]. After
a call to `torch.class()` you have to fill-up properly the metatable.

After the class definition is complete, constructing a new class _name_ will be achieved by a call to `_name_()`.
This call will first call the method ```lua__init()``` if it exists, passing all arguments of `_name_()`.

```lua
 require "torch"

 -- for naming convenience
 do
   --- creates a class "Foo"
   local Foo = torch.class('Foo')
 
   --- the initializer
   function Foo:__init()
     self.contents = "this is some text"
   end

   --- a method
   function Foo:print()
     print(self.contents)
   end

   --- another one
   function Foo:bip()
     print('bip')
   end

 end

 --- now create an instance of Foo
 foo = Foo()

 --- try it out
 foo:print()

 --- create a class torch.Bar which
 --- inherits from Foo
 do
   local Bar, parent = torch.class('torch.Bar', 'Foo')

   --- the initializer
   function Bar:__init(stuff)
     --- call the parent initializer on ourself
     parent.__init(self)
 
     --- do some stuff
     self.stuff = stuff
   end

   --- a new method
   function Bar:boing()
     print('boing!')
   end

   --- override parent's method
   function Bar:print()
     print(self.contents)
     print(self.stuff)
   end
 end

 --- create a new instance and use it
 bar = torch.Bar("ha ha!")
 bar:print() -- overrided method
 bar:boing() -- child method
 bar:bip()   -- parent's method

```

For advanced users, it is worth mentionning that `torch.class()` actually
calls [torch.newmetatable()](#torch.newmetatable).  with a particular
constructor. The constructor creates a Lua table and set the right
metatable on it, and then calls ```lua__init()``` if it exists in the
metatable. It also sets a [factory](#torch.factory) field ```lua__factory``` such that it
is possible to create an empty object of this class.

<a name="torch.typename"/>
### [string] torch.typename(object) ###

Checks if `object` has a metatable. If it does, and if it corresponds to a
`Torch` class, then returns a string containing the name of the
class. Returns `nil` in any other cases.

A Torch class is a class created with [torch.class()](#torch.class) or
[torch.newmetatable()](#torch.newmetatable).

<a name="torch.typename2id"/>
### [userdata] torch.typename2id(string) ###

Given a Torch class name specified by `string`, returns a unique
corresponding id (defined by a `lightuserdata` pointing on the internal
structure of the class). This might be useful to do a _fast_ check of the
class of an object (if used with [torch.id()](#torch.id)), avoiding string
comparisons.

Returns `nil` if `string` does not specify a Torch object.

<a name="torch.id"/>
### [userdata] torch.id(object) ###

Returns a unique id corresponding to the _class_ of the given Torch object.
The id is defined by a `lightuserdata` pointing on the internal structure
of the class.

Returns `nil` if `object` is not a Torch object.

This is different from the _object_ id returned by [torch.pointer()](#torch.pointer).

<a name="torch.newmetatable"/>
### [table] torch.newmetatable(name, parentName, constructor) ###

Register a new metatable as a Torch type with the given string `name`. The new metatable is returned.

If the string `parentName` is not `nil` and is a valid Torch type (previously created
by `torch.newmetatable()`) then set the corresponding metatable as a metatable to the returned new
metatable. 

If the given `constructor` function is not `nil`, then assign to the variable `name` the given constructor.
The given `name` might be of the form `package.className`, in which case the `className` will be local to the
specified `package`. In that case, `package` must be a valid and already loaded package.

<a name="torch.factory"/>
### [function] torch.factory(name) ###

Returns the factory function of the Torch class `name`. If the class name is invalid or if the class
has no factory, then returns `nil`.

A Torch class is a class created with [torch.class()](#torch.class) or
[torch.newmetatable()](#torch.newmetatable).

A factory function is able to return a new (empty) object of its corresponding class. This is helpful for
[object serialization](file.md#torch.File.serialization).

<a name="torch.getmetatable"/>
### [table] torch.getmetatable(string) ###

Given a `string`, returns a metatable corresponding to the Torch class described
by `string`. Returns `nil` if the class does not exist.

A Torch class is a class created with [torch.class()](#torch.class) or
[torch.newmetatable()](#torch.newmetatable).

Example:
```lua
> for k,v in pairs(torch.getmetatable("torch.CharStorage")) do print(k,v) end
__index__       function: 0x1a4ba80
__typename      torch.CharStorage
write   function: 0x1a49cc0
__tostring__    function: 0x1a586e0
__newindex__    function: 0x1a4ba40
string  function: 0x1a4d860
__version       1
copy    function: 0x1a49c80
read    function: 0x1a4d840
__len__ function: 0x1a37440
fill    function: 0x1a375c0
resize  function: 0x1a37580
__index table: 0x1a4a080
size    function: 0x1a4ba20
```

<a name="torch.isequal"/>
### [boolean] torch.isequal(object1, object2) ###

If the two objects given as arguments are `Lua` tables (or Torch objects), then returns `true` if and only if the
tables (or Torch objects) have the same address in memory. Returns `false` in any other cases.

A Torch class is a class created with [torch.class()](#TorchClass) or
[torch.newmetatable()](#torch.newmetatable).

<a name="torch.getdefaulttensortype"/>
### [string] torch.getdefaulttensortype() ###

Returns a string representing the default tensor type currently in use
by Torch7.

<a name="torch.getenv"/>
### [table] torch.getenv(function or userdata) ###

Returns the Lua `table` environment of the given `function` or the given
`userdata`.  To know more about environments, please read the documentation
of [lua_setfenv()](http://www.lua.org/manual/5.1/manual.html#lua_setfenv)
and [lua_getfenv()](http://www.lua.org/manual/5.1/manual.html#lua_getfenv).

<a name="torch.version"/>
### [number] torch.version(object) ###

Returns the field ```lua__version``` of a given object. This might
be helpful to handle variations in a class over time.

<a name="torch.pointer"/>
### [number] torch.pointer(object) ###

Returns a unique id (pointer) of the given `object`, which can be a Torch
object, a table, a thread or a function.

This is different from the _class_ id returned by [torch.id()](#torch.id).

<a name="torch.setdefaulttensortype"/>
### torch.setdefaulttensortype([typename]) ###

Sets the default tensor type for all the tensors allocated from this
point on. Valid types are:
  * `ByteTensor`
  * `CharTensor`
  * `ShortTensor`
  * `IntTensor`
  * `FloatTensor`
  * `DoubleTensor`

<a name="torch.setenv"/>
### torch.setenv(function or userdata, table) ###

Assign `table` as the Lua environment of the given `function` or the given
`userdata`.  To know more about environments, please read the documentation
of [lua_setfenv()](http://www.lua.org/manual/5.1/manual.html#lua_setfenv)
and [lua_getfenv()](http://www.lua.org/manual/5.1/manual.html#lua_getfenv).

<a name="torch.setmetatable"/>
### [object] torch.setmetatable(table, classname) ###

Set the metatable of the given `table` to the metatable of the Torch
object named `classname`.  This function has to be used with a lot
of care.

<a name="torch.getconstructortable"/>
### [table] torch.getconstructortable(string) ###

BUGGY
Return the constructor table of the Torch class specified by ''string'.

