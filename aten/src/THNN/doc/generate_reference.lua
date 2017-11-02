--[[
  This script regenerates api_reference.md based on comments placed in THNN.h.
]]--

local header = [[
# API docs

This document only describes a THNN API. For a thorough review of all modules present here please refer to [nn's docs](http://github.com/torch/nn/tree/master/doc).

### Note on function names

Please remember, that because C doesn't support function overloading, functions taking different tensor types have different names. So e.g. for an Abs module, there are actually two updateOutput functions:

* `void THNN_FloatAbs_updateOutput(...)`
* `void THNN_DoubleAbs_updateOutput(...)`

In these docs such function will be referred to as `void THNN_Abs_updateOutput(...)`, and it's up to developer to add a type prefix. `real` is an alias for that type.

### Argument types

Some arguments have additional tags placed in square brackets:
* **[OUT]** - This is the output argument. It will be reshaped if needed.
* **[OPTIONAL]** - This argument is optional and can be safely set to NULL
* **[BUFFER]** - A buffer. `updateGradInput` and `accGradParameters` should get the same buffers that were used in `updateOutput` call.
* **[MODIFIED]** - Some functions accept an `inplace` flag. If set to true, this argument might be modified (in addition to the output).

## Module list

These are all modules implemented in THNN:

]]

local hfile = io.open('../generic/THNN.h', 'r')
local lines = hfile:read('*a'):split('\n')
hfile:close()

-- Parse input
local declarations = {}
local current_declaration
local declaration_module
for i,line in ipairs(lines) do
   if line:sub(1, 6) == 'TH_API' then
     current_declaration = ''
     declaration_module = line:match('THNN_%((.+)_.+%)')
   end

   if current_declaration then
      current_declaration = current_declaration .. line .. '\n'
   end

   if line:match('%);') then
     current_declaration = current_declaration:sub(1, -2) -- remove a trailing newline
     declarations[declaration_module] = declarations[declaration_module] or {}
     table.insert(declarations[declaration_module], current_declaration)
     current_declaration = nil
     declaration_module = nil
   end
end
declarations["unfolded"] = nil

-- Sort modules
modules = {}
for k,_ in pairs(declarations) do table.insert(modules, k) end
table.sort(modules)

-- Create an index
local outfile = io.open('api_reference.md', 'w')
outfile:write(header)
for i, name in ipairs(modules) do
    outfile:write(string.format('* [%s](#%s)\n', name, name:lower()))
end
outfile:write('\n')

-- Write proper docs
for i,name in ipairs(modules) do
    outfile:write('## ' .. name ..'\n')

    for i,declaration in ipairs(declarations[name]) do

        -- Write source code
        outfile:write('```C' .. '\n')
        local declaration_lines = declaration:split('\n')
        for i, line in ipairs(declaration_lines) do
            if i == 1 then
                line = line:gsub('TH_API ', ''):gsub('%(', ''):gsub('%)', '') .. '(' -- remove macro junk
            else
                line = line:gsub('%s*//.*$', '') -- remove the comment
            end
            outfile:write(line .. '\n')
        end
        outfile:write('```' .. '\n')

        -- Describe arguments
        table.remove(declaration_lines, 1)
        for i,line in ipairs(declaration_lines) do
            local param, comment = line:match('^%s*(.*),%s*// (.*)$')
            if param == nil then param, comment = line:match('^%s*(.*)%);%s*// (.*)$') end

            if param ~= nil then
                comment = comment:gsub('%[', '%*%*%['):gsub('%]', '%]%*%*') -- use bold font for tags
                outfile:write(string.format('`%s` - %s\n<br/>\n', param, comment))
            end
        end
    end
end
outfile:close()
