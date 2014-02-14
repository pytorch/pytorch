local wrap = require 'cwrap'
local types = wrap.types

types.Tensor = {

   helpname = function(arg)
                 if arg.dim then
                    return string.format("Tensor~%dD", arg.dim)
                 else
                    return "Tensor"
                 end
            end,

   declare = function(arg)
                local txt = {}
                table.insert(txt, string.format("THTensor *arg%d = NULL;", arg.i))
                if arg.returned then
                   table.insert(txt, string.format("int arg%d_idx = 0;", arg.i));
                end
                return table.concat(txt, '\n')
           end,
   
   check = function(arg, idx)
              if arg.dim then
                 return string.format("(arg%d = luaT_toudata(L, %d, torch_Tensor)) && (arg%d->nDimension == %d)", arg.i, idx, arg.i, arg.dim)
              else
                 return string.format("(arg%d = luaT_toudata(L, %d, torch_Tensor))", arg.i, idx)
              end
         end,

   read = function(arg, idx)
             if arg.returned then
                return string.format("arg%d_idx = %d;", arg.i, idx)
             end
          end,

   init = function(arg)
             if type(arg.default) == 'boolean' then
                return string.format('arg%d = THTensor_(new)();', arg.i)
             elseif type(arg.default) == 'number' then
                return string.format('arg%d = %s;', arg.i, arg.args[arg.default]:carg())
             else
                error('unknown default tensor type value')
             end
          end,
   
   carg = function(arg)
             return string.format('arg%d', arg.i)
          end,

   creturn = function(arg)
                return string.format('arg%d', arg.i)
             end,
   
   precall = function(arg)
                local txt = {}
                if arg.default and arg.returned then
                   table.insert(txt, string.format('if(arg%d_idx)', arg.i)) -- means it was passed as arg
                   table.insert(txt, string.format('lua_pushvalue(L, arg%d_idx);', arg.i))
                   table.insert(txt, string.format('else'))
                   if type(arg.default) == 'boolean' then -- boolean: we did a new()
                      table.insert(txt, string.format('luaT_pushudata(L, arg%d, torch_Tensor);', arg.i))
                   else  -- otherwise: point on default tensor --> retain
                      table.insert(txt, string.format('{'))
                      table.insert(txt, string.format('THTensor_(retain)(arg%d);', arg.i)) -- so we need a retain
                      table.insert(txt, string.format('luaT_pushudata(L, arg%d, torch_Tensor);', arg.i))
                      table.insert(txt, string.format('}'))
                   end
                elseif arg.default then
                   -- we would have to deallocate the beast later if we did a new
                   -- unlikely anyways, so i do not support it for now
                   if type(arg.default) == 'boolean' then
                      error('a tensor cannot be optional if not returned')
                   end
                elseif arg.returned then
                   table.insert(txt, string.format('lua_pushvalue(L, arg%d_idx);', arg.i))
                end
                return table.concat(txt, '\n')
             end,

   postcall = function(arg)
                 local txt = {}
                 if arg.creturned then
                    -- this next line is actually debatable
                    table.insert(txt, string.format('THTensor_(retain)(arg%d);', arg.i))
                    table.insert(txt, string.format('luaT_pushudata(L, arg%d, torch_Tensor);', arg.i))
                 end
                 return table.concat(txt, '\n')
              end
}

types.IndexTensor = {

   helpname = function(arg)
               return "LongTensor"
            end,

   declare = function(arg)
                local txt = {}
                table.insert(txt, string.format("THLongTensor *arg%d = NULL;", arg.i))
                if arg.returned then
                   table.insert(txt, string.format("int arg%d_idx = 0;", arg.i));
                end
                return table.concat(txt, '\n')
           end,
   
   check = function(arg, idx)
              return string.format('(arg%d = luaT_toudata(L, %d, "torch.LongTensor"))', arg.i, idx)
           end,

   read = function(arg, idx)
             local txt = {}
             if not arg.noreadadd then
                table.insert(txt, string.format("THLongTensor_add(arg%d, arg%d, -1);", arg.i, arg.i));
             end
             if arg.returned then
                table.insert(txt, string.format("arg%d_idx = %d;", arg.i, idx))
             end
             return table.concat(txt, '\n')
          end,

   init = function(arg)
             return string.format('arg%d = THLongTensor_new();', arg.i)
          end,
   
   carg = function(arg)
             return string.format('arg%d', arg.i)
          end,

   creturn = function(arg)
                return string.format('arg%d', arg.i)
             end,
   
   precall = function(arg)
                local txt = {}
                if arg.default and arg.returned then
                   table.insert(txt, string.format('if(arg%d_idx)', arg.i)) -- means it was passed as arg
                   table.insert(txt, string.format('lua_pushvalue(L, arg%d_idx);', arg.i))
                   table.insert(txt, string.format('else')) -- means we did a new()
                   table.insert(txt, string.format('luaT_pushudata(L, arg%d, "torch.LongTensor");', arg.i))
                elseif arg.default then
                   error('a tensor cannot be optional if not returned')
                elseif arg.returned then
                   table.insert(txt, string.format('lua_pushvalue(L, arg%d_idx);', arg.i))
                end
                return table.concat(txt, '\n')
             end,

   postcall = function(arg)
                 local txt = {}
                 if arg.creturned or arg.returned then
                    table.insert(txt, string.format("THLongTensor_add(arg%d, arg%d, 1);", arg.i, arg.i));
                 end
                 if arg.creturned then
                    -- this next line is actually debatable
                    table.insert(txt, string.format('THLongTensor_retain(arg%d);', arg.i))
                    table.insert(txt, string.format('luaT_pushudata(L, arg%d, "torch.LongTensor");', arg.i))
                 end
                 return table.concat(txt, '\n')
              end
}

for _,typename in ipairs({"ByteTensor", "CharTensor", "ShortTensor", "IntTensor", "LongTensor",
                          "FloatTensor", "DoubleTensor"}) do

   types[typename] = {

      helpname = function(arg)
                    if arg.dim then
                       return string.format('%s~%dD', typename, arg.dim)
                    else
                       return typename
                    end
                 end,
      
      declare = function(arg)
                   local txt = {}
                   table.insert(txt, string.format("TH%s *arg%d = NULL;", typename, arg.i))
                   if arg.returned then
                      table.insert(txt, string.format("int arg%d_idx = 0;", arg.i));
                   end
                   return table.concat(txt, '\n')
                end,
      
      check = function(arg, idx)
                 if arg.dim then
                    return string.format('(arg%d = luaT_toudata(L, %d, "torch.%s")) && (arg%d->nDimension == %d)', arg.i, idx, typename, arg.i, arg.dim)
                 else
                    return string.format('(arg%d = luaT_toudata(L, %d, "torch.%s"))', arg.i, idx, typename)
                 end
              end,

      read = function(arg, idx)
                if arg.returned then
                   return string.format("arg%d_idx = %d;", arg.i, idx)
                end
             end,
      
      init = function(arg)
                if type(arg.default) == 'boolean' then
                   return string.format('arg%d = TH%s_new();', arg.i, typename)
                elseif type(arg.default) == 'number' then
                   return string.format('arg%d = %s;', arg.i, arg.args[arg.default]:carg())
                else
                   error('unknown default tensor type value')
                end
             end,

      carg = function(arg)
                return string.format('arg%d', arg.i)
             end,

      creturn = function(arg)
                   return string.format('arg%d', arg.i)
             end,
      
      precall = function(arg)
                   local txt = {}
                   if arg.default and arg.returned then
                      table.insert(txt, string.format('if(arg%d_idx)', arg.i)) -- means it was passed as arg
                      table.insert(txt, string.format('lua_pushvalue(L, arg%d_idx);', arg.i))
                      table.insert(txt, string.format('else'))
                      if type(arg.default) == 'boolean' then -- boolean: we did a new()
                         table.insert(txt, string.format('luaT_pushudata(L, arg%d, "torch.%s");', arg.i, typename))
                      else  -- otherwise: point on default tensor --> retain
                         table.insert(txt, string.format('{'))
                         table.insert(txt, string.format('TH%s_retain(arg%d);', typename, arg.i)) -- so we need a retain
                         table.insert(txt, string.format('luaT_pushudata(L, arg%d, "torch.%s");', arg.i, typename))
                         table.insert(txt, string.format('}'))
                      end
                   elseif arg.default then
                      -- we would have to deallocate the beast later if we did a new
                      -- unlikely anyways, so i do not support it for now
                      if type(arg.default) == 'boolean' then
                         error('a tensor cannot be optional if not returned')
                      end
                   elseif arg.returned then
                      table.insert(txt, string.format('lua_pushvalue(L, arg%d_idx);', arg.i))
                   end
                   return table.concat(txt, '\n')
                end,

      postcall = function(arg)
                    local txt = {}
                    if arg.creturned then
                       -- this next line is actually debatable
                       table.insert(txt, string.format('TH%s_retain(arg%d);', typename, arg.i))
                       table.insert(txt, string.format('luaT_pushudata(L, arg%d, "torch.%s");', arg.i, typename))
                    end
                    return table.concat(txt, '\n')
                 end
   }
end

types.LongArg = {

   vararg = true,

   helpname = function(arg)
               return "(LongStorage | dim1 [dim2...])"
            end,

   declare = function(arg)
              return string.format("THLongStorage *arg%d = NULL;", arg.i)
           end,

   init = function(arg)
             if arg.default then
                error('LongArg cannot have a default value')
             end
          end,
   
   check = function(arg, idx)
            return string.format("torch_islongargs(L, %d)", idx)
         end,

   read = function(arg, idx)
             return string.format("arg%d = torch_checklongargs(L, %d);", arg.i, idx)
          end,
   
   carg = function(arg, idx)
             return string.format('arg%d', arg.i)
          end,

   creturn = function(arg, idx)
                return string.format('arg%d', arg.i)
             end,
   
   precall = function(arg)
                local txt = {}
                if arg.returned then
                   table.insert(txt, string.format('luaT_pushudata(L, arg%d, "torch.LongStorage");', arg.i))
                end
                return table.concat(txt, '\n')
             end,

   postcall = function(arg)
                 local txt = {}
                 if arg.creturned then
                    -- this next line is actually debatable
                    table.insert(txt, string.format('THLongStorage_retain(arg%d);', arg.i))
                    table.insert(txt, string.format('luaT_pushudata(L, arg%d, "torch.LongStorage");', arg.i))
                 end
                 if not arg.returned and not arg.creturned then
                    table.insert(txt, string.format('THLongStorage_free(arg%d);', arg.i))
                 end
                 return table.concat(txt, '\n')
              end   
}

types.charoption = {
   
   helpname = function(arg)
                 if arg.values then
                    return "(" .. table.concat(arg.values, '|') .. ")"
                 end
              end,

   declare = function(arg)
                local txt = {}
                table.insert(txt, string.format("const char *arg%d = NULL;", arg.i))
                if arg.default then
                   table.insert(txt, string.format("char arg%d_default = '%s';", arg.i, arg.default))
                end
                return table.concat(txt, '\n')
           end,

   init = function(arg)
             return string.format("arg%d = &arg%d_default;", arg.i, arg.i)
          end,
   
   check = function(arg, idx)
              local txt = {}
              local txtv = {}
              table.insert(txt, string.format('(arg%d = lua_tostring(L, %d)) && (', arg.i, idx))
              for _,value in ipairs(arg.values) do
                 table.insert(txtv, string.format("*arg%d == '%s'", arg.i, value))
              end
              table.insert(txt, table.concat(txtv, ' || '))
              table.insert(txt, ')')              
              return table.concat(txt, '')
         end,

   read = function(arg, idx)
          end,
   
   carg = function(arg, idx)
             return string.format('arg%d', arg.i)
          end,

   creturn = function(arg, idx)
             end,
   
   precall = function(arg)
             end,

   postcall = function(arg)
              end   
}
