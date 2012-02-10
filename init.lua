
-- We are using paths.require to appease mkl
require "paths"
paths.require "libtorch"

--- package stuff
function torch.packageLuaPath(name)
   if not name then
      local ret = string.match(torch.packageLuaPath('torch'), '(.*)/')
       if not ret then --windows?
           ret = string.match(torch.packageLuaPath('torch'), '(.*)\\')
       end
       return ret 
   end
   for path in string.gmatch(package.path, "(.-);") do
      path = string.gsub(path, "%?", name)
      local f = io.open(path)
      if f then
         f:close()
         local ret = string.match(path, "(.*)/")
         if not ret then --windows?
             ret = string.match(path, "(.*)\\")
         end
         return ret
      end
   end
end

function torch.include(package, file)
   dofile(torch.packageLuaPath(package) .. '/' .. file) 
end

function torch.class(tname, parenttname)

   local function constructor(...)
      local self = {}
      torch.setmetatable(self, tname)
      if self.__init then
         self:__init(...)
      end
      return self
   end
   
   local function factory()
      local self = {}
      torch.setmetatable(self, tname)
      return self
   end

   local mt = torch.newmetatable(tname, parenttname, constructor, nil, factory)
   local mpt
   if parenttname then
      mpt = torch.getmetatable(parenttname)
   end
   return mt, mpt
end

function torch.setdefaulttensortype(typename)
   assert(type(typename) == 'string', 'string expected')
   if torch.getconstructortable(typename) then
      torch.Tensor = torch.getconstructortable(typename)
      torch.Storage = torch.getconstructortable(torch.typename(torch.Tensor(1):storage()))
      torch.__setdefaulttensortype(typename)
   else
      error(string.format("<%s> is not a string describing a torch object", typename))
   end
end

local localinstalldir = paths.concat(os.getenv('HOME'),'.torch','usr')
if paths.dirp(localinstalldir) then
   package.path = package.path .. ';' .. paths.concat(localinstalldir,'share','torch','lua','?','init.lua')
   package.path = package.path .. ';' .. paths.concat(localinstalldir,'share','torch','lua','?.lua')
   package.cpath = package.cpath .. ';' .. paths.concat(localinstalldir,'lib','torch','?.so')
   package.cpath = package.cpath .. ';' .. paths.concat(localinstalldir,'lib','torch','?.dylib')
end

torch.setdefaulttensortype('torch.DoubleTensor')

torch.include('torch', 'Tensor.lua')
torch.include('torch', 'File.lua')
torch.include('torch', 'CmdLine.lua')
torch.include('torch', 'Tester.lua')
torch.include('torch', 'test.lua')
return torch
