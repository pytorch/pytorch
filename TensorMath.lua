local interface = wrap.CInterface.new()
local method = wrap.CInterface.new()

interface:print([[
#include "TH.h"
#include "luaT.h"
#include "utils.h"
                ]])

-- special argument specific to torch package
local argtypes = {}
argtypes.LongArg = {

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

argtypes.charoption = {
   
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

-- both interface & method support these new arguments
for k,v in pairs(argtypes) do
   interface.argtypes[k] = v
   method.argtypes[k] = v
end
   
-- also specific to torch: we generate a 'dispatch' function
-- first we create a helper function
-- note that it let the "torch" table on the stack
interface:print([[
static const void* torch_istensortype(lua_State *L, const char *tname)
{
  if(!tname)
    return NULL;

  if(!luaT_pushmetatable(L, tname))
    return NULL;

  lua_pushstring(L, "torch");
  lua_rawget(L, -2);
  if(lua_istable(L, -1))
    return tname;
  else
  {
    lua_pop(L, 2);
    return NULL;
  }

  return NULL;
}
]])

interface.dispatchregistry = {}
function interface:wrap(name, ...)
   -- usual stuff
   wrap.CInterface.wrap(self, name, ...)

   -- dispatch function
   if not interface.dispatchregistry[name] then
      interface.dispatchregistry[name] = true
      table.insert(interface.dispatchregistry, {name=name, wrapname=string.format("torch_%s", name)})

      interface:print(string.gsub([[
static int torch_NAME(lua_State *L)
{
  int narg = lua_gettop(L);
  const void *tname;
  if(narg >= 1 && (tname = torch_istensortype(L, luaT_typename(L, 1)))) /* first argument is tensor? */
  {
  }
  else if(narg >= 2 && (tname = torch_istensortype(L, luaT_typename(L, 2)))) /* second? */
  {
  }
  else if(narg >= 1 && lua_isstring(L, narg)
	  && (tname = torch_istensortype(L, lua_tostring(L, narg)))) /* do we have a valid tensor type string then? */
  {
    lua_remove(L, -2);
  }
  else if(!(tname = torch_istensortype(L, torch_getdefaulttensortype(L))))
    luaL_error(L, "internal error: the default tensor type does not seem to be an actual tensor");
  
  lua_pushstring(L, "NAME");
  lua_rawget(L, -2);
  if(lua_isfunction(L, -1))
  {
    lua_insert(L, 1);
    lua_pop(L, 2); /* the two tables we put on the stack above */
    lua_call(L, lua_gettop(L)-1, LUA_MULTRET);
  }
  else
    return luaL_error(L, "%s does not implement the torch.NAME() function", tname);

  return lua_gettop(L);
}
]], 'NAME', name))
  end
end

function interface:dispatchregister(name)
   local txt = self.txt
   table.insert(txt, string.format('static const struct luaL_Reg %s [] = {', name))
   for _,reg in ipairs(self.dispatchregistry) do
      table.insert(txt, string.format('{"%s", %s},', reg.name, reg.wrapname))
   end
   table.insert(txt, '{NULL, NULL}')
   table.insert(txt, '};')
   table.insert(txt, '')   
   self.dispatchregistry = {}
end

interface:print('/* WARNING: autogenerated file */')
interface:print('')

local function wrap(...)
   local args = {...}

   -- interface
   interface:wrap(...)

   -- method: we override things possibly in method table field
   for _,x in ipairs(args) do
      if type(x) == 'table' then -- ok, now we have a list of args
         for _, arg in ipairs(x) do
            if arg.method then
               for k,v in pairs(arg.method) do
                  if v == 'nil' then -- special case, we erase the field
                     arg[k] = nil
                  else
                     arg[k] = v
                  end
               end
            end
         end
      end
   end
   method:wrap(unpack(args))
end

local reals = {ByteTensor='unsigned char',
               CharTensor='char',
               ShortTensor='short',
               IntTensor='int',
               LongTensor='long',
               FloatTensor='float',
               DoubleTensor='double'}

local accreals = {ByteTensor='long',
               CharTensor='long',
               ShortTensor='long',
               IntTensor='long',
               LongTensor='long',
               FloatTensor='double',
               DoubleTensor='double'}

for _,Tensor in ipairs({"ByteTensor", "CharTensor",
                        "ShortTensor", "IntTensor", "LongTensor",
                        "FloatTensor", "DoubleTensor"}) do

   local real = reals[Tensor]
   local accreal = accreals[Tensor]

   function interface.luaname2wrapname(self, name)
      return string.format('torch_%s_%s', Tensor, name)
   end

   function method.luaname2wrapname(self, name)
      return string.format('m_torch_%s_%s', Tensor, name)
   end

   local function cname(name)
      return string.format('TH%s_%s', Tensor, name)
   end

   local function lastdim(argn)
      return function(arg)
                return string.format("TH%s_nDimension(%s)", Tensor, arg.args[argn]:carg())
             end
   end
   
   wrap("zero",
        cname("zero"),
        {{name=Tensor, returned=true}})
   
   wrap("fill",
        cname("fill"),
        {{name=Tensor, returned=true},
         {name=real}})

   wrap("zeros",
        cname("zeros"),
        {{name=Tensor, default=true, returned=true, method={default='nil'}},
         {name="LongArg"}})

   wrap("ones",
        cname("ones"),
        {{name=Tensor, default=true, returned=true, method={default='nil'}},
         {name="LongArg"}})

   wrap("reshape",
        cname("reshape"),
        {{name=Tensor, default=true, returned=true},
         {name=Tensor},
         {name="LongArg"}})

   wrap("dot",
        cname("dot"),
        {{name=Tensor},
         {name=Tensor},
         {name=accreal, creturned=true}})
   
   wrap("add",
        cname("add"),
        {{name=Tensor, default=true, returned=true, method={default='nil'}},
         {name=Tensor, method={default=1}},
         {name=real}},
        cname("cadd"),
        {{name=Tensor, default=true, returned=true, method={default='nil'}},
         {name=Tensor, method={default=1}},
         {name=real, default=1},
         {name=Tensor}})
   
   wrap("mul",
        cname("mul"),
        {{name=Tensor, default=true, returned=true, method={default='nil'}},
         {name=Tensor, method={default=1}},
         {name=real}})

   wrap("div",
        cname("div"),
        {{name=Tensor, default=true, returned=true, method={default='nil'}},
         {name=Tensor, method={default=1}},
         {name=real}})

   wrap("cmul",
        cname("cmul"),
        {{name=Tensor, default=true, returned=true, method={default='nil'}},
         {name=Tensor, method={default=1}},
         {name=Tensor}})

   wrap("cdiv",
        cname("cdiv"),
        {{name=Tensor, default=true, returned=true, method={default='nil'}},
         {name=Tensor, method={default=1}},
         {name=Tensor}})

   wrap("addcmul",
        cname("addcmul"),
        {{name=Tensor, default=true, returned=true, method={default='nil'}},
         {name=Tensor, method={default=1}},
         {name=real, default=1},
         {name=Tensor},
         {name=Tensor}})

   wrap("addcdiv",
        cname("addcdiv"),
        {{name=Tensor, default=true, returned=true, method={default='nil'}},
         {name=Tensor, method={default=1}},
         {name=real, default=1},
         {name=Tensor},
         {name=Tensor}})

   wrap("mv",
        cname("addmv"),
        {{name=Tensor, default=true, returned=true, method={default='nil'},
          init=function(arg)
                  return table.concat(
                     {
                        arg.__metatable.init(arg),
                        string.format("TH%s_resize1d(%s, %s->size[0]);", Tensor, arg:carg(), arg.args[5]:carg())
                     }, '\n')
               end,
          precall=function(arg)
                     return table.concat(
                        {
                           string.format("TH%s_zero(%s);", Tensor, arg:carg()),
                           arg.__metatable.precall(arg)
                        }, '\n')
                  end
       },
         {name=real, default=1, invisible=true},
         {name=Tensor, default=1, invisible=true},
         {name=real, default=1, invisible=true},
         {name=Tensor, dim=2},
         {name=Tensor, dim=1}}
     )

   wrap("mm",
        cname("addmm"),
        {{name=Tensor, default=true, returned=true, method={default='nil'},
          init=function(arg)
                  return table.concat(
                     {
                        arg.__metatable.init(arg),
                        string.format("TH%s_resize2d(%s, %s->size[0], %s->size[1]);", Tensor, arg:carg(), arg.args[5]:carg(), arg.args[6]:carg())
                     }, '\n')
               end,
          precall=function(arg)
                     return table.concat(
                        {
                           string.format("TH%s_zero(%s);", Tensor, arg:carg()),
                           arg.__metatable.precall(arg)
                        }, '\n')
                  end
       },
         {name=real, default=1, invisible=true},
         {name=Tensor, default=1, invisible=true},
         {name=real, default=1, invisible=true},
         {name=Tensor, dim=2},
         {name=Tensor, dim=2}}
     )

   wrap("ger",
        cname("addr"),
        {{name=Tensor, default=true, returned=true, method={default='nil'},
          init=function(arg)
                  return table.concat(
                     {
                        arg.__metatable.init(arg),
                        string.format("TH%s_resize2d(%s, %s->size[0], %s->size[0]);", Tensor, arg:carg(), arg.args[5]:carg(), arg.args[6]:carg())
                     }, '\n')
               end,
          precall=function(arg)
                     return table.concat(
                        {
                           string.format("TH%s_zero(%s);", Tensor, arg:carg()),
                           arg.__metatable.precall(arg)
                        }, '\n')
                  end
       },
        {name=real, default=1, invisible=true},
        {name=Tensor, default=1, invisible=true},
        {name=real, default=1, invisible=true},
        {name=Tensor, dim=1},
        {name=Tensor, dim=1}}
     )

   for _,f in ipairs({
                        {name="addmv", dim1=1, dim2=2, dim3=1},
                        {name="addmm", dim1=2, dim2=2, dim3=2},
                        {name="addr",  dim1=2, dim2=1, dim3=1},
                     }
                  ) do

      interface:wrap(f.name,
                     cname(f.name),
                     {{name=Tensor, default=true, returned=true},
                      {name=real, default=1},
                      {name=Tensor, dim=f.dim1},
                      {name=real, default=1},
                      {name=Tensor, dim=f.dim2},
                      {name=Tensor, dim=f.dim3}})

      -- there is an ambiguity here, hence the more complicated setup
      method:wrap(f.name,
                  cname(f.name),
                  {{name=Tensor, returned=true, dim=f.dim1},
                   {name=real, default=1, invisible=true},
                   {name=Tensor, default=1, dim=f.dim1},
                   {name=real, default=1},
                   {name=Tensor, dim=f.dim2},
                   {name=Tensor, dim=f.dim3}},
                  cname(f.name),
                  {{name=Tensor, returned=true, dim=f.dim1},
                   {name=real},
                   {name=Tensor, default=1, dim=f.dim1},
                   {name=real},
                   {name=Tensor, dim=f.dim2},
                   {name=Tensor, dim=f.dim3}})
   end

   wrap("numel",
        cname("numel"),
        {{name=Tensor},
         {name=real, creturned=true}})

   for _,name in ipairs({"prod", "cumsum", "cumprod"}) do
      wrap(name,
           cname(name),
           {{name=Tensor, default=true, returned=true},
            {name=Tensor},
            {name="index"}})
   end

   wrap("sum",
        cname("sumall"),
        {{name=Tensor},
         {name=accreal, creturned=true}},
        cname("sum"),
        {{name=Tensor, default=true, returned=true},
         {name=Tensor},
         {name="index"}})
   
   for _,name in ipairs({"min", "max"}) do
      wrap(name,
           cname(name .. "all"),
           {{name=Tensor},
            {name=real, creturned=true}},
           cname(name),
           {{name=Tensor, default=true, returned=true},
            {name="IndexTensor", default=true, returned=true},
            {name=Tensor},
            {name="index"}})
   end

   wrap("trace",
        cname("trace"),
        {{name=Tensor},
         {name=accreal, creturned=true}})
   
   wrap("cross",
        cname("cross"),
        {{name=Tensor, default=true, returned=true},
         {name=Tensor},
         {name=Tensor},
         {name="index", default=0}})
   
   wrap("diag",
        cname("diag"),
        {{name=Tensor, default=true, returned=true},
         {name=Tensor},
         {name="long", default=0}})
   
   wrap("eye",
        cname("eye"),
        {{name=Tensor, default=true, returned=true, method={default='nil'}},
         {name="long"},
         {name="long", default=0}})
   
   wrap("range",
        cname("range"),
        {{name=Tensor, default=true, returned=true, method={default='nil'}},
         {name=real},
         {name=real},
         {name=real, default=1}})

   wrap("randperm",
        cname("randperm"),
        {{name=Tensor, default=true, returned=true, method={default='nil'},
          postcall=function(arg)
                      return table.concat(
                         {
                            arg.__metatable.postcall(arg),
                            string.format("TH%s_add(%s, %s, 1);", Tensor, arg:carg(), arg:carg())
                         }, '\n')
                   end},
         {name="long"}})

   wrap("sort",
        cname("sort"),
        {{name=Tensor, default=true, returned=true},
         {name="IndexTensor", default=true, returned=true},
         {name=Tensor},
         {name="index", default=lastdim(3)},
         {name="boolean", default=0}})
   
   wrap("tril",
        cname("tril"),
        {{name=Tensor, default=true, returned=true},
         {name=Tensor},
         {name="int", default=0}})

   wrap("triu",
        cname("triu"),
        {{name=Tensor, default=true, returned=true},
         {name=Tensor},
         {name="int", default=0}})

   wrap("cat",
        cname("cat"),
        {{name=Tensor, default=true, returned=true},
         {name=Tensor},
         {name=Tensor},
         {name="index", default=lastdim(2)}})
   
   if Tensor == 'ByteTensor' then -- we declare this only once
      interface:print(
         [[
static int THRandom_random2__(long a, long b)
{
  THArgCheck(b >= a, 2, "upper bound must be larger than lower bound");
  return((THRandom_random() % (b+1-a)) + a);
}
         
static int THRandom_random1__(long b)
{
  THArgCheck(b > 0, 1, "upper bound must be strictly positive");
  return(THRandom_random() % b + 1);
}
         ]])
   end

   interface:print(string.gsub(
                      [[
static void THTensor_random2__(THTensor *self, long a, long b)
{
  THArgCheck(b >= a, 2, "upper bound must be larger than lower bound");
  TH_TENSOR_APPLY(real, self, *self_data = ((THRandom_random() % (b+1-a)) + a);)
}

static void THTensor_random1__(THTensor *self, long b)
{
  THArgCheck(b > 0, 1, "upper bound must be strictly positive");
  TH_TENSOR_APPLY(real, self, *self_data = (THRandom_random() % b + 1);)
}
]], 'Tensor', Tensor):gsub('real', real))

   wrap('random',
        'THRandom_random2__',
        {{name='long'},
         {name='long'},
         {name='long', creturned=true}},
        'THRandom_random1__',
        {{name='long'},
         {name='long', creturned=true}},
        'THRandom_random',
        {{name='long', creturned=true}},
        cname("random2__"),
        {{name=Tensor, returned=true},
         {name='long'},
         {name='long'}},
        cname("random1__"),
        {{name=Tensor, returned=true},
         {name='long'}},
        cname("random"),
        {{name=Tensor, returned=true}})

   for _,f in ipairs({{name='geometric'},
                      {name='bernoulli', a=0.5}}) do
      
      wrap(f.name,
           string.format("THRandom_%s", f.name),
           {{name="double", default=f.a},
            {name="double", creturned=true}},
           cname(f.name),
           {{name=Tensor, returned=true},
            {name=real, default=f.a}})
   end

   wrap("squeeze",
        cname("squeeze"),
        {{name=Tensor, default=true, returned=true, postcall=function(arg)
                                                                local txt = {}
                                                                if arg.returned then
                                                                   table.insert(txt, string.format('if(arg%d->nDimension == 1 && arg%d->size[0] == 1)', arg.i, arg.i)) -- number
                                                                   table.insert(txt, string.format('lua_pushnumber(L, (lua_Number)(*TH%s_data(arg%d)));', Tensor, arg.i))
                                                                end
                                                                return table.concat(txt, '\n')
                                                             end},
         {name=Tensor}},
        cname("squeeze1d"),
        {{name=Tensor, default=true, returned=true, postcall=function(arg)
                                                                local txt = {}
                                                                if arg.returned then
                                                                   table.insert(txt, string.format('if(arg%d->nDimension == 1 && arg%d->size[0] == 1)', arg.i, arg.i)) -- number
                                                                   table.insert(txt, string.format('lua_pushnumber(L, (lua_Number)(*TH%s_data(arg%d)));', Tensor, arg.i))
                                                                end
                                                                return table.concat(txt, '\n')
                                                             end},
         {name=Tensor},
         {name="index"}})

   wrap("sign",
        cname("sign"),
        {{name=Tensor, default=true, returned=true, method={default='nil'}},
         {name=Tensor, method={default=1}}})

   wrap("conv2",
        cname("conv2Dmul"),
        {{name=Tensor, default=true, returned=true},
         {name=real, default=0, invisible=true},
         {name=real, default=1, invisible=true},
         {name=Tensor, dim=2},
         {name=Tensor, dim=2},
         {name=real, default=1, invisible=true},
         {name=real, default=1, invisible=true},
         {name='charoption', values={'V', 'F'}, default='V'},
         {name='charoption', default="C", invisible=true}},
        cname("conv2Dcmul"),
        {{name=Tensor, default=true, returned=true},
         {name=real, default=0, invisible=true},
         {name=real, default=1, invisible=true},
         {name=Tensor, dim=3},
         {name=Tensor, dim=3},
         {name=real, default=1, invisible=true},
         {name=real, default=1, invisible=true},
         {name='charoption', values={'V', 'F'}, default='V'},
         {name='charoption', default="C", invisible=true}},
        cname("conv2Dmv"),
        {{name=Tensor, default=true, returned=true},
         {name=real, default=0, invisible=true},
         {name=real, default=1, invisible=true},
         {name=Tensor, dim=3},
         {name=Tensor, dim=4},
         {name=real, default=1, invisible=true},
         {name=real, default=1, invisible=true},
         {name='charoption', values={'V', 'F'}, default='V'},
         {name='charoption', default="C", invisible=true}}
     )

   wrap("xcorr2",
        cname("conv2Dmul"),
        {{name=Tensor, default=true, returned=true},
         {name=real, default=0, invisible=true},
         {name=real, default=1, invisible=true},
         {name=Tensor, dim=2},
         {name=Tensor, dim=2},
         {name=real, default=1, invisible=true},
         {name=real, default=1, invisible=true},
         {name='charoption', values={'V', 'F'}, default='V'},
         {name='charoption', default="X", invisible=true}},
        cname("conv2Dcmul"),
        {{name=Tensor, default=true, returned=true},
         {name=real, default=0, invisible=true},
         {name=real, default=1, invisible=true},
         {name=Tensor, dim=3},
         {name=Tensor, dim=3},
         {name=real, default=1, invisible=true},
         {name=real, default=1, invisible=true},
         {name='charoption', values={'V', 'F'}, default='V'},
         {name='charoption', default="X", invisible=true}},
        cname("conv2Dmv"),
        {{name=Tensor, default=true, returned=true},
         {name=real, default=0, invisible=true},
         {name=real, default=1, invisible=true},
         {name=Tensor, dim=3},
         {name=Tensor, dim=4},
         {name=real, default=1, invisible=true},
         {name=real, default=1, invisible=true},
         {name='charoption', values={'V', 'F'}, default='V'},
         {name='charoption', default="X", invisible=true}}
     )

   wrap("conv3",
        cname("conv3Dmul"),
        {{name=Tensor, default=true, returned=true},
         {name=real, default=0, invisible=true},
         {name=real, default=1, invisible=true},
         {name=Tensor, dim=3},
         {name=Tensor, dim=3},
         {name=real, default=1, invisible=true},
         {name=real, default=1, invisible=true},
         {name=real, default=1, invisible=true},
         {name='charoption', values={'V', 'F'}, default='V'},
         {name='charoption', default="C", invisible=true}},
        cname("conv3Dcmul"),
        {{name=Tensor, default=true, returned=true},
         {name=real, default=0, invisible=true},
         {name=real, default=1, invisible=true},
         {name=Tensor, dim=4},
         {name=Tensor, dim=4},
         {name=real, default=1, invisible=true},
         {name=real, default=1, invisible=true},
         {name=real, default=1, invisible=true},
         {name='charoption', values={'V', 'F'}, default='V'},
         {name='charoption', default="C", invisible=true}},
        cname("conv3Dmv"),
        {{name=Tensor, default=true, returned=true},
         {name=real, default=0, invisible=true},
         {name=real, default=1, invisible=true},
         {name=Tensor, dim=4},
         {name=Tensor, dim=5},
         {name=real, default=1, invisible=true},
         {name=real, default=1, invisible=true},
         {name=real, default=1, invisible=true},
         {name='charoption', values={'V', 'F'}, default='V'},
         {name='charoption', default="C", invisible=true}}
     )

   wrap("xcorr3",
        cname("conv3Dmul"),
        {{name=Tensor, default=true, returned=true},
         {name=real, default=0, invisible=true},
         {name=real, default=1, invisible=true},
         {name=Tensor, dim=3},
         {name=Tensor, dim=3},
         {name=real, default=1, invisible=true},
         {name=real, default=1, invisible=true},
         {name=real, default=1, invisible=true},
         {name='charoption', values={'V', 'F'}, default='V'},
         {name='charoption', default="X", invisible=true}},
        cname("conv3Dcmul"),
        {{name=Tensor, default=true, returned=true},
         {name=real, default=0, invisible=true},
         {name=real, default=1, invisible=true},
         {name=Tensor, dim=4},
         {name=Tensor, dim=4},
         {name=real, default=1, invisible=true},
         {name=real, default=1, invisible=true},
         {name=real, default=1, invisible=true},
         {name='charoption', values={'V', 'F'}, default='V'},
         {name='charoption', default="X", invisible=true}},
        cname("conv3Dmv"),
        {{name=Tensor, default=true, returned=true},
         {name=real, default=0, invisible=true},
         {name=real, default=1, invisible=true},
         {name=Tensor, dim=4},
         {name=Tensor, dim=5},
         {name=real, default=1, invisible=true},
         {name=real, default=1, invisible=true},
         {name=real, default=1, invisible=true},
         {name='charoption', values={'V', 'F'}, default='V'},
         {name='charoption', default="X", invisible=true}}
     )

   for _,name in pairs({'lt','gt','le','ge','eq','ne'}) do
      wrap(name,
           cname(name .. 'Value'),
           {{name='ByteTensor',default=true, returned=true},
            {name=Tensor},
            {name=real}},
           cname(name .. 'Tensor'),
           {{name='ByteTensor',default=true, returned=true},
            {name=Tensor},
            {name=Tensor}})
   end

   if Tensor == 'FloatTensor' or Tensor == 'DoubleTensor' then

      wrap("mean",
           cname("meanall"),
           {{name=Tensor},
            {name=accreal, creturned=true}},
           cname("mean"),
           {{name=Tensor, default=true, returned=true},
            {name=Tensor},
            {name="index"}})
      
      for _,name in ipairs({"var", "std"}) do
         wrap(name,
              cname(name .. "all"),
              {{name=Tensor},
               {name=accreal, creturned=true}},
              cname(name),
              {{name=Tensor, default=true, returned=true},
               {name=Tensor},
               {name="index"},
               {name="boolean", default=false}})
      end
      wrap("histc",
           cname("histc"),
           {{name=Tensor, default=true, returned=true},
            {name=Tensor},
            {name="long",default=100},
            {name="double",default=0},
            {name="double",default=0}})

      wrap("norm",
           cname("normall"),
           {{name=Tensor},
            {name=real, default=2},
            {name=accreal, creturned=true}},
           cname("norm"),
           {{name=Tensor, default=true, returned=true},
            {name=Tensor},
            {name=real},
            {name="index"}})
      
      wrap("dist",
           cname("dist"),
           {{name=Tensor},
            {name=Tensor},
            {name=real, default=2},
            {name=accreal, creturned=true}})
      
      wrap("linspace",
           cname("linspace"),
           {{name=Tensor, default=true, returned=true, method={default='nil'}},
            {name=real},
            {name=real},
            {name="long", default=100}})

      wrap("logspace",
           cname("logspace"),
           {{name=Tensor, default=true, returned=true, method={default='nil'}},
            {name=real},
            {name=real},
            {name="long", default=100}})
      
      for _,name in ipairs({"log", "log1p", "exp",
                            "cos", "acos", "cosh",
                            "sin", "asin", "sinh",
                            "tan", "atan", "tanh",
                            "sqrt",
                            "ceil", "floor"}) do
                            --"abs"}) do

         wrap(name, 
              cname(name),
              {{name=Tensor, default=true, returned=true, method={default='nil'}},
               {name=Tensor, method={default=1}}},
              name,
              {{name=real},
               {name=real, creturned=true}})
         
      end
         wrap("abs",
              cname("abs"),
              {{name=Tensor, default=true, returned=true, method={default='nil'}},
               {name=Tensor, method={default=1}}},
              "fabs",
              {{name=real},
               {name=real, creturned=true}})

      wrap("atan2",
           cname("atan2"),
           {{name=Tensor, default=true, returned=true, method={default='nil'}},
            {name=Tensor, method={default=1}},
            {name=Tensor}},
           "atan2",
           {{name=real},
            {name=real},
            {name=real, creturned=true}}
            )

      wrap("pow",
           cname("pow"),
           {{name=Tensor, default=true, returned=true, method={default='nil'}},
            {name=Tensor, method={default=1}},
            {name=real}},
           "pow",
           {{name=real},
            {name=real},
            {name=real, creturned=true}})

      wrap("rand",
           cname("rand"),
           {{name=Tensor, default=true, returned=true, method={default='nil'}},
            {name="LongArg"}})

      wrap("randn",
           cname("randn"),
           {{name=Tensor, default=true, returned=true, method={default='nil'}},
            {name="LongArg"}})
      
      for _,f in ipairs({{name='uniform', a=0, b=1},
                         {name='normal', a=0, b=1},
                         {name='cauchy', a=0, b=1},
                         {name='logNormal', a=1, b=2}}) do
         
         wrap(f.name,
              string.format("THRandom_%s", f.name),
              {{name="double", default=f.a},
               {name="double", default=f.b},
               {name="double", creturned=true}},
              cname(f.name),
              {{name=Tensor, returned=true},
               {name=real, default=f.a},
               {name=real, default=f.b}})
      end

      for _,f in ipairs({{name='exponential'}}) do
         
         wrap(f.name,
              string.format("THRandom_%s", f.name),
              {{name="double", default=f.a},
               {name="double", creturned=true}},
              cname(f.name),
              {{name=Tensor, returned=true},
               {name=real, default=f.a}})
      end
      
      for _,name in ipairs({"gesv","gels"}) do
         interface:wrap(name,
                        cname(name),
                        {{name=Tensor, returned=true},
                         {name=Tensor, returned=true},
                         {name=Tensor},
                         {name=Tensor}},
                        cname(name),
                        {{name=Tensor, default=true, returned=true, invisible=true},
                         {name=Tensor, default=true, returned=true, invisible=true},
                         {name=Tensor},
                         {name=Tensor}}
                     )
      end

      interface:wrap("symeig",
                     cname("syev"),
                     {{name=Tensor, returned=true},
                      {name=Tensor, returned=true},
                      {name=Tensor},
                      {name='charoption', values={'N', 'V'}, default='N'},
                      {name='charoption', values={'U', 'L'}, default='U'}},
                     cname("syev"),
                     {{name=Tensor, default=true, returned=true, invisible=true},
                      {name=Tensor, default=true, returned=true, invisible=true},
                      {name=Tensor},
                      {name='charoption', values={'N', 'V'}, default='N'},
                      {name='charoption', values={'U', 'L'}, default='U'}}
                  )
      interface:wrap("eig",
                     cname("geev"),
                     {{name=Tensor, returned=true},
                      {name=Tensor, returned=true},
                      {name=Tensor},
                      {name='charoption', values={'N', 'V'}, default='N'}},
                     cname("geev"),
                     {{name=Tensor, default=true, returned=true, invisible=true},
                      {name=Tensor, default=true, returned=true, invisible=true},
                      {name=Tensor},
                      {name='charoption', values={'N', 'V'}, default='N'}}
                  )

      interface:wrap("svd",
                     cname("gesvd"),
                     {{name=Tensor, returned=true},
                      {name=Tensor, returned=true},
                      {name=Tensor, returned=true},
                      {name=Tensor},
                      {name='charoption', values={'A', 'S'}, default='S'}},
                     cname("gesvd"),
                     {{name=Tensor, default=true, returned=true, invisible=true},
                      {name=Tensor, default=true, returned=true, invisible=true},
                      {name=Tensor, default=true, returned=true, invisible=true},
                      {name=Tensor},
                      {name='charoption', values={'A', 'S'}, default='S'}}
                  )
      interface:wrap("inverse",
                     cname("getri"),
                     {{name=Tensor, returned=true},
                      {name=Tensor}},
                     cname("getri"),
                     {{name=Tensor, default=true, returned=true, invisible=true},
                      {name=Tensor}}
                  )
      
   end

   method:register(string.format("m_torch_%sMath__", Tensor))
   interface:print(method:tostring())
   method:clearhistory()
   interface:register(string.format("torch_%sMath__", Tensor))

   interface:print(string.gsub([[
static void torch_TensorMath_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.Tensor");

  /* register methods */
  luaL_register(L, NULL, m_torch_TensorMath__);

  /* register functions into the "torch" field of the tensor metaclass */
  lua_pushstring(L, "torch");
  lua_newtable(L);
  luaL_register(L, NULL, torch_TensorMath__);
  lua_rawset(L, -3);
  lua_pop(L, 1);
}
]], 'Tensor', Tensor))
end

interface:dispatchregister("torch_TensorMath__")

interface:print([[
void torch_TensorMath_init(lua_State *L)
{
  torch_ByteTensorMath_init(L);
  torch_CharTensorMath_init(L);
  torch_ShortTensorMath_init(L);
  torch_IntTensorMath_init(L);
  torch_LongTensorMath_init(L);
  torch_FloatTensorMath_init(L);
  torch_DoubleTensorMath_init(L);
  luaL_register(L, NULL, torch_TensorMath__);
}
]])

if arg[1] then
   interface:tofile(arg[1])
else
   print(interface:tostring())
end
