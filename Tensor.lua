-- additional methods for Storage
local Storage = {}

-- additional methods for Tensor
local Tensor = {}

-- types
local types = {'Byte', 'Char', 'Short', 'Int', 'Long', 'Float', 'Double'}

-- tostring() functions for Tensor and Storage
local function Storage__printformat(self)
   local intMode = true
   local type = torch.typename(self)
--   if type == 'torch.FloatStorage' or type == 'torch.DoubleStorage' then
      for i=1,self:size() do
         if self[i] ~= math.ceil(self[i]) then
            intMode = false
            break
         end
      end
--   end
   local tensor = torch.DoubleTensor(torch.DoubleStorage(self:size()):copy(self), 1, self:size()):abs()
   local expMin = tensor:min()
   if expMin ~= 0 then
      expMin = math.floor(math.log10(expMin)) + 1
   end
   local expMax = tensor:max()
   if expMax ~= 0 then
      expMax = math.floor(math.log10(expMax)) + 1
   end

   local format
   local scale
   local sz
   if intMode then
      if expMax > 9 then
         format = "%11.4e"
         sz = 11
      else
         format = "%SZd"
         sz = expMax + 1
      end
   else
      if expMax-expMin > 4 then
         format = "%SZ.4e"
         sz = 11
         if math.abs(expMax) > 99 or math.abs(expMin) > 99 then
            sz = sz + 1
         end
      else
         if expMax > 5 or expMax < 0 then
            format = "%SZ.4f"
            sz = 7
            scale = math.pow(10, expMax-1)
         else
            format = "%SZ.4f"
            if expMax == 0 then
               sz = 7
            else
               sz = expMax+6
            end
         end
      end
   end
   format = string.gsub(format, 'SZ', sz)
   if scale == 1 then
      scale = nil
   end
   return format, scale, sz
end

function Storage.__tostring__(self)
   local strt = {'\n'}
   local format,scale = Storage__printformat(self)
   if format:sub(2,4) == 'nan' then format = '%f' end
   if scale then
      table.insert(strt, string.format('%g', scale) .. ' *\n')
      for i = 1,self:size() do
         table.insert(strt, string.format(format, self[i]/scale) .. '\n')
      end
   else
      for i = 1,self:size() do
         table.insert(strt, string.format(format, self[i]) .. '\n')
      end
   end
   table.insert(strt, '[' .. torch.typename(self) .. ' of size ' .. self:size() .. ']\n')
   str = table.concat(strt)
   return str
end

for _,type in ipairs(types) do
   local metatable = torch.getmetatable('torch.' .. type .. 'Storage')
   for funcname, func in pairs(Storage) do
      rawset(metatable, funcname, func)
   end
end

local function Tensor__printMatrix(self, indent)
   local format,scale,sz = Storage__printformat(self:storage())
   if format:sub(2,4) == 'nan' then format = '%f' end
--   print('format = ' .. format)
   scale = scale or 1
   indent = indent or ''
   local strt = {indent}
   local nColumnPerLine = math.floor((80-#indent)/(sz+1))
--   print('sz = ' .. sz .. ' and nColumnPerLine = ' .. nColumnPerLine)
   local firstColumn = 1
   local lastColumn = -1
   while firstColumn <= self:size(2) do
      if firstColumn + nColumnPerLine - 1 <= self:size(2) then
         lastColumn = firstColumn + nColumnPerLine - 1
      else
         lastColumn = self:size(2)
      end
      if nColumnPerLine < self:size(2) then
         if firstColumn ~= 1 then
            table.insert(strt, '\n')
         end
         table.insert(strt, 'Columns ' .. firstColumn .. ' to ' .. lastColumn .. '\n' .. indent)
      end
      if scale ~= 1 then
         table.insert(strt, string.format('%g', scale) .. ' *\n ' .. indent)
      end
      for l=1,self:size(1) do
         local row = self:select(1, l)
         for c=firstColumn,lastColumn do
            table.insert(strt, string.format(format, row[c]/scale))
            if c == lastColumn then
               table.insert(strt, '\n')
               if l~=self:size(1) then
                  if scale ~= 1 then
                     table.insert(strt, indent .. ' ')
                  else
                     table.insert(strt, indent)
                  end
               end
            else
               table.insert(strt, ' ')
            end
         end
      end
      firstColumn = lastColumn + 1
   end
   local str = table.concat(strt)
   return str
end

local function Tensor__printTensor(self)
   local counter = torch.LongStorage(self:nDimension()-2)
   local strt = {''}
   local finished
   counter:fill(1)
   counter[1] = 0
   while true do
      for i=1,self:nDimension()-2 do
         counter[i] = counter[i] + 1
         if counter[i] > self:size(i) then
            if i == self:nDimension()-2 then
               finished = true
               break
            end
            counter[i] = 1
         else
            break
         end
      end
      if finished then
         break
      end
--      print(counter)
      if #strt > 1 then
         table.insert(strt, '\n')
      end
      table.insert(strt, '(')
      local tensor = self
      for i=1,self:nDimension()-2 do
         tensor = tensor:select(1, counter[i])
         table.insert(strt, counter[i] .. ',')
      end
      table.insert(strt, '.,.) = \n')
      table.insert(strt, Tensor__printMatrix(tensor, ' '))
   end
   local str = table.concat(strt)
   return str
end

function Tensor.__tostring__(self)
   local str = '\n'
   local strt = {''}
   if self:nDimension() == 0 then
      table.insert(strt, '[' .. torch.typename(self) .. ' with no dimension]\n')
   else
      local tensor = torch.DoubleTensor():resize(self:size()):copy(self)
      if tensor:nDimension() == 1 then
         local format,scale,sz = Storage__printformat(tensor:storage())
	 if format:sub(2,4) == 'nan' then format = '%f' end
         if scale then
            table.insert(strt, string.format('%g', scale) .. ' *\n')
            for i = 1,tensor:size(1) do
               table.insert(strt, string.format(format, tensor[i]/scale) .. '\n')
            end
         else
            for i = 1,tensor:size(1) do
               table.insert(strt, string.format(format, tensor[i]) .. '\n')
            end
         end
         table.insert(strt, '[' .. torch.typename(self) .. ' of dimension ' .. tensor:size(1) .. ']\n')
      elseif tensor:nDimension() == 2 then
         table.insert(strt, Tensor__printMatrix(tensor))
         table.insert(strt, '[' .. torch.typename(self) .. ' of dimension ' .. tensor:size(1) .. 'x' .. tensor:size(2) .. ']\n')
      else
         table.insert(strt, Tensor__printTensor(tensor))
         table.insert(strt, '[' .. torch.typename(self) .. ' of dimension ')
         for i=1,tensor:nDimension() do
            table.insert(strt, tensor:size(i))
            if i ~= tensor:nDimension() then
               table.insert(strt, 'x')
            end
         end
         table.insert(strt, ']\n')
      end
   end
   local str = table.concat(strt)
   return str
end

function Tensor.type(self,type)
   local current = torch.typename(self)
   if not type then return current end
   if type ~= current then
      local new = torch.getmetatable(type).new()
      if self:nElement() > 0 then
         new:resize(self:size()):copy(self)
      end
      return new
   else
      return self
   end
end

function Tensor.typeAs(self,tensor)
   return self:type(tensor:type())
end

function Tensor.byte(self)
   return self:type('torch.ByteTensor')
end

function Tensor.char(self)
   return self:type('torch.CharTensor')
end

function Tensor.short(self)
   return self:type('torch.ShortTensor')
end

function Tensor.int(self)
   return self:type('torch.IntTensor')
end

function Tensor.long(self)
   return self:type('torch.LongTensor')
end

function Tensor.float(self)
   return self:type('torch.FloatTensor')
end

function Tensor.double(self)
   return self:type('torch.DoubleTensor')
end

function Tensor.real(self)
   return self:type(torch.getdefaulttensortype())
end

function Tensor.expand(tensor,...)
   -- get sizes
   local sizes = {...}

   -- check type
   local size
   if torch.typename(sizes[1]) and torch.typename(sizes[1])=='torch.LongStorage' then
      size = sizes[1]
   else
      size = torch.LongStorage(#sizes)
      for i,s in ipairs(sizes) do
         size[i] = s
      end
   end

   -- get dimensions
   local tensor_dim = tensor:dim()
   local tensor_stride = tensor:stride()
   local tensor_size = tensor:size()

   -- check nb of dimensions
   if #size ~= tensor:dim() then
      error('the number of dimensions provided must equal tensor:dim()')
   end

   -- create a new geometry for tensor:
   for i = 1,tensor_dim do
      if tensor_size[i] == 1 then
         tensor_size[i] = size[i]
         tensor_stride[i] = 0
      elseif tensor_size[i] ~= size[i] then
         error('incorrect size: only supporting singleton expansion (size=1)')
      end
   end

   -- create new view, with singleton expansion:
   tensor = torch.Tensor(tensor:storage(), tensor:storageOffset(),
                         tensor_size, tensor_stride)
   return tensor
end
torch.expand = Tensor.expand

function Tensor.expandAs(tensor,template)
   return tensor:expand(template:size())
end
torch.expandAs = Tensor.expandAs

for _,type in ipairs(types) do
   local metatable = torch.getmetatable('torch.' .. type .. 'Tensor')
   for funcname, func in pairs(Tensor) do
      rawset(metatable, funcname, func)
   end
end
