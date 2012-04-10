local File = torch.getmetatable('torch.File')

function File:writeBool(value)
   if value then
      self:writeInt(1)
   else
      self:writeInt(0)
   end
end

function File:readBool()
   return (self:readInt() == 1)
end

local TYPE_NIL      = 0
local TYPE_NUMBER   = 1
local TYPE_STRING   = 2
local TYPE_TABLE    = 3
local TYPE_TORCH    = 4
local TYPE_BOOLEAN  = 5
local TYPE_FUNCTION = 6

function File:isWritableObject(object)
   local typename = type(object)
   local typeidx
   if type(object) ~= 'boolean' and not object then
      typeidx = TYPE_NIL
   elseif torch.typename(object) and torch.factory(torch.typename(object)) then
      typeidx = TYPE_TORCH
   elseif typename == 'table' then
      typeidx = TYPE_TABLE
   elseif typename == 'number' then
      typeidx = TYPE_NUMBER
   elseif typename == 'string' then
      typeidx = TYPE_STRING
   elseif typename == 'boolean' then
      typeidx = TYPE_BOOLEAN
   elseif typename == 'function' and pcall(string.dump, object) then
      typeidx = TYPE_FUNCTION
   end
   return typeidx
end

function File:writeObject(object)
   -- we use an environment to keep a record of written objects
   if not torch.getenv(self).writeObjects then
      torch.setenv(self, {writeObjects={}, writeObjectsRef={}, readObjects={}})
   end

   -- if nil object, only write the type and return
   if type(object) ~= 'boolean' and not object then
      self:writeInt(TYPE_NIL)
      return
   end

   -- check the type we are dealing with
   local typeidx = self:isWritableObject(object)
   if not typeidx then
      error(string.format('Unwritable object <%s>', type(object)))
   end
   self:writeInt(typeidx)

   if typeidx == TYPE_NUMBER then
      self:writeDouble(object)
   elseif typeidx == TYPE_BOOLEAN then
      self:writeBool(object)
   elseif typeidx == TYPE_STRING then
      local stringStorage = torch.CharStorage():string(object)
      self:writeInt(#stringStorage)
      self:writeChar(stringStorage)
   elseif typeidx == TYPE_FUNCTION then
      local upvalues = {}
      while true do
         local name,value = debug.getupvalue(object, #upvalues+1)
         if not name then break end
         table.insert(upvalues, value)
      end
      local dumped = string.dump(object)
      local stringStorage = torch.CharStorage():string(dumped)
      self:writeInt(#stringStorage)
      self:writeChar(stringStorage)
      self:writeObject(upvalues)
   elseif typeidx == TYPE_TORCH or typeidx == TYPE_TABLE then
      -- check it exists already (we look at the pointer!)
      local objects = torch.getenv(self).writeObjects
      local objectsRef = torch.getenv(self).writeObjectsRef
      local index = objects[torch.pointer(object)]

      if index then
         -- if already exists, write only its index
         self:writeInt(index)
      else
         -- else write the object itself
         index = objects.nWriteObject or 0
         index = index + 1
         objects[torch.pointer(object)] = index
         objectsRef[object] = index -- we make sure the object is not going to disappear
         self:writeInt(index)
         objects.nWriteObject = index

         if typeidx == TYPE_TORCH then
            local version   = torch.CharStorage():string('V ' .. torch.version(object))
            local className = torch.CharStorage():string(torch.typename(object))
            self:writeInt(#version)
            self:writeChar(version)
            self:writeInt(#className)
            self:writeChar(className)
            if object.write then
               object:write(self)
            elseif type(object) == 'table' then
               local var = {}
               for k,v in pairs(object) do
                  if self:isWritableObject(v) then
                     var[k] = v
                  else
                     print(string.format('$ Warning: cannot write object field <%s>', k))
                  end
               end
               self:writeObject(var)
            else
               error(string.format('<%s> is a non-serializable Torch object', torch.typename(object)))
            end
         else -- it is a table
            local size = 0; for k,v in pairs(object) do size = size + 1 end
            self:writeInt(size)
            for k,v in pairs(object) do
               self:writeObject(k)
               self:writeObject(v)
            end
         end
      end
   else
      error('Unwritable object')
   end
end

function File:readObject()
   -- we use an environment to keep a record of read objects
   if not torch.getenv(self).writeObjects then
      torch.setenv(self, {writeObjects={}, writeObjectsRef={}, readObjects={}})
   end

   -- read the typeidx
   local typeidx = self:readInt()

   -- is it nil?
   if typeidx == TYPE_NIL then
      return nil
   end

   if typeidx == TYPE_NUMBER then
      return self:readDouble()
   elseif typeidx == TYPE_BOOLEAN then
      return self:readBool()
   elseif typeidx == TYPE_STRING then
      local size = self:readInt()
      return self:readChar(size):string()
   elseif typeidx == TYPE_FUNCTION then
      local size = self:readInt()
      local dumped = self:readChar(size):string()
      local func = loadstring(dumped)
      local upvalues = self:readObject()
      for index,upvalue in ipairs(upvalues) do
         debug.setupvalue(func, index, upvalue)
      end
      return func
   elseif typeidx == TYPE_TABLE or typeidx == TYPE_TORCH then
      -- read the index
      local index = self:readInt()

      -- check it is loaded already
      local objects = torch.getenv(self).readObjects
      if objects[index] then
         return objects[index]
      end

      -- otherwise read it
      if typeidx == TYPE_TORCH then
         local version, className, versionNumber
         version = self:readChar(self:readInt()):string()
         versionNumber = tonumber(string.match(version, '^V (.*)$'))
         if not versionNumber then
            className = version
            versionNumber = 0 -- file created before existence of versioning system
         else
            className = self:readChar(self:readInt()):string()
         end
         if not torch.factory(className) then
            error(string.format('unknown Torch class <%s>', tostring(className)))
         end
         local object = torch.factory(className)()
         objects[index] = object
         if object.read then
            object:read(self, versionNumber)
         elseif type(object) == 'table' then
            local var = self:readObject(var)
            for k,v in pairs(var) do
               object[k] = v
            end
         else
            error(string.format('Cannot load object class <%s>', tostring(className)))
         end
         return object
      else -- it is a table
         local size = self:readInt()
         local object = {}
         objects[index] = object
         for i = 1,size do
            local k = self:readObject()
            local v = self:readObject()
            object[k] = v
         end
         return object
      end
   else
      error('unknown object')
   end
end

-- simple helpers to save/load arbitrary objects/tables
function torch.save(filename, object, mode)
   mode = mode or 'binary'
   local file = torch.DiskFile(filename, 'w')
   file[mode](file)
   file:writeObject(object)
   file:close()
end

function torch.load(filename, mode)
   mode = mode or 'binary'
   local file = torch.DiskFile(filename, 'r')
   file[mode](file)
   local object = file:readObject()
   file:close()
   return object
end

-- public API (saveobj/loadobj are safe for global import)
torch.saveobj = torch.save
torch.loadobj = torch.load
