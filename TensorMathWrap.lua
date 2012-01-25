-- 
-- require 'wrap'
---

local interface = wrap.CInterface.new()

interface:print([[
#include "TH.h"
#include "luaT.h"
#include "utils.h"

static const void* torch_ByteTensor_id;
static const void* torch_CharTensor_id;
static const void* torch_ShortTensor_id;
static const void* torch_IntTensor_id;
static const void* torch_LongTensor_id;
static const void* torch_FloatTensor_id;
static const void* torch_DoubleTensor_id;

static const void* torch_LongStorage_id;
                ]])

-- special argument specific to torch package
interface.argtypes.LongArg = {

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
                   table.insert(txt, string.format('luaT_pushudata(L, arg%d, torch_LongStorage_id);', arg.i))
                end
                return table.concat(txt, '\n')
             end,

   postcall = function(arg)
                 local txt = {}
                 if arg.creturned then
                    -- this next line is actually debatable
                    table.insert(txt, string.format('THLongStorage_retain(arg%d);', arg.i))
                    table.insert(txt, string.format('luaT_pushudata(L, arg%d, torch_LongStorage_id);', arg.i))
                 end
                 if not arg.returned and not arg.creturned then
                    table.insert(txt, string.format('THLongStorage_free(arg%d);', arg.i))
                 end
                 return table.concat(txt, '\n')
              end   
}

interface.argtypes.charoption = {

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

-- also specific to torch: we generate a 'dispatch' function
-- first we create a helper function
interface:print([[
static const void* torch_istensorid(lua_State *L, const void *id)
{
  if(!id)
    return NULL;

  luaT_pushmetaclass(L, id);
  lua_pushstring(L, "torch");
  lua_rawget(L, -2);
  if(lua_istable(L, -1))
    return id;
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
  const void *id;

  if(narg < 1 || !(id = torch_istensorid(L, luaT_id(L, 1)))) /* first argument is tensor? */
  {
    if(narg < 2 || !(id = torch_istensorid(L, luaT_id(L, 2)))) /* second? */
    {
      if(lua_isstring(L, -1) && (id = torch_istensorid(L, luaT_typename2id(L, lua_tostring(L, -1))))) /* do we have a valid string then? */
        lua_pop(L, 1);
      else if(!(id = torch_istensorid(L, torch_getdefaulttensorid())))
        luaL_error(L, "internal error: the default tensor type does not seem to be an actual tensor");
    }
  }
  
  lua_pushstring(L, "NAME");
  lua_rawget(L, -2);
  if(lua_isfunction(L, -1))
  {
    lua_insert(L, 1);
    lua_pop(L, 2); /* the two tables we put on the stack above */
    lua_call(L, lua_gettop(L)-1, LUA_MULTRET);
  }
  else
    return luaL_error(L, "%s does not implement the torch.NAME() function", luaT_id2typename(L, id));

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

local reals = {ByteTensor='unsigned char',
               CharTensor='char',
               ShortTensor='short',
               IntTensor='int',
               LongTensor='long',
               FloatTensor='float',
               DoubleTensor='double'}

for _,Tensor in ipairs({"ByteTensor", "CharTensor",
                        "ShortTensor", "IntTensor", "LongTensor",
                        "FloatTensor", "DoubleTensor"}) do

   local real = reals[Tensor]

   function interface.luaname2wrapname(self, name)
      return string.format('torch_%s_%s', Tensor, name)
   end

   local function cname(name)
      return string.format('TH%s_%s', Tensor, name)
   end

   local function lastdim(argn)
      return function(arg)
                return string.format("TH%s_nDimension(%s)", Tensor, arg.args[argn]:carg())
             end
   end
   
   interface:wrap("zero",
                  cname("zero"),
                  {{name=Tensor, returned=true}})

   interface:wrap("fill",
                  cname("fill"),
                  {{name=Tensor, returned=true},
                   {name=real}})

   interface:wrap("zeros",
                  cname("zeros"),
                  {{name=Tensor, default=true, returned=true},
                   {name="LongArg"}})

   interface:wrap("ones",
                  cname("ones"),
                  {{name=Tensor, default=true, returned=true},
                   {name="LongArg"}})

   interface:wrap("reshape",
                  cname("reshape"),
                  {{name=Tensor, default=true, returned=true},
                   {name=Tensor},
                   {name="LongArg"}})

   interface:wrap("dot",
                  cname("dot"),
                  {{name=Tensor},
                   {name=Tensor},
                   {name=real, creturned=true}})

   for _,name in ipairs({"minall", "maxall", "sumall"}) do
      interface:wrap(name,
                     cname(name),
                     {{name=Tensor},            
                      {name=real, creturned=true}})
   end

   interface:wrap("add",
                  cname("add"),
                  {{name=Tensor, default=true, returned=true},
                   {name=Tensor},
                   {name=real}},
                  cname("cadd"),
                  {{name=Tensor, default=true, returned=true},
                   {name=Tensor},
                   {name=real, default=1},
                   {name=Tensor}})

   interface:wrap("mul",
                  cname("mul"),
                  {{name=Tensor, default=true, returned=true},
                   {name=Tensor},
                   {name=real}})

   interface:wrap("div",
                  cname("div"),
                  {{name=Tensor, default=true, returned=true},
                   {name=Tensor},
                   {name=real}})

   interface:wrap("cmul",
                  cname("cmul"),
                  {{name=Tensor, default=true, returned=true},
                   {name=Tensor},
                   {name=Tensor}})

   interface:wrap("cdiv",
                  cname("cdiv"),
                  {{name=Tensor, default=true, returned=true},
                   {name=Tensor},
                   {name=Tensor}})

   interface:wrap("addcmul",
                  cname("addcmul"),
                  {{name=Tensor, default=true, returned=true},
                   {name=Tensor},
                   {name=real, default=1},
                   {name=Tensor},
                   {name=Tensor}})

   interface:wrap("addcdiv",
                  cname("addcdiv"),
                  {{name=Tensor, default=true, returned=true},
                   {name=Tensor},
                   {name=real, default=1},
                   {name=Tensor},
                   {name=Tensor}})

   for _,name in ipairs({"addmv", "addmm", "addr"}) do
      interface:wrap(name,
                     cname(name),
                     {{name=Tensor, default=true, returned=true},
                      {name=real, default=1},
                      {name=Tensor},
                      {name=real, default=1},
                      {name=Tensor},
                      {name=Tensor}})
   end

   interface:wrap("numel",
                  cname("numel"),
                  {{name=Tensor},
                   {name=real, creturned=true}})

   for _,name in ipairs({"sum", "prod", "cumsum", "cumprod"}) do
      interface:wrap(name,
                     cname(name),
                     {{name=Tensor, default=true, returned=true},
                      {name=Tensor},
                      {name="index", default=lastdim(2)}})
   end

   interface:wrap("min",
                  cname("min"),
                  {{name=Tensor, default=true, returned=true},
                   {name="IndexTensor", default=true, returned=true},
                   {name=Tensor},
                   {name="index", default=lastdim(3)}})

   interface:wrap("max",
                  cname("max"),
                  {{name=Tensor, default=true, returned=true},
                   {name="IndexTensor", default=true, returned=true},
                   {name=Tensor},
                   {name="index", default=lastdim(3)}})

   interface:wrap("trace",
                  cname("trace"),
                  {{name=Tensor},
                   {name=real, creturned=true}})

   interface:wrap("cross",
                  cname("cross"),
                  {{name=Tensor, default=true, returned=true},
                   {name=Tensor},
                   {name=Tensor},
                   {name="index", default=0}})

   interface:wrap("diag",
                  cname("diag"),
                  {{name=Tensor, default=true, returned=true},
                   {name=Tensor},
                   {name="long", default=0}})

   interface:wrap("eye",
                  cname("eye"),
                  {{name=Tensor, default=true, returned=true},
                   {name="long"},
                   {name="long", default=0}})

   interface:wrap("range",
                  cname("range"),
                  {{name=Tensor, default=true, returned=true},
                   {name=real},
                   {name=real},
                   {name=real, default=1}})

   interface:wrap("randperm",
                  cname("randperm"),
                  {{name=Tensor, default=true, returned=true, userpostcall=function(arg)
                                                                              return string.format("TH%s_add(%s, %s, 1);", Tensor, arg:carg(), arg:carg())
                                                                           end},
                   {name="long"}})

   interface:wrap("sort",
                  cname("sort"),
                  {{name=Tensor, default=true, returned=true},
                   {name="IndexTensor", default=true, returned=true},
                   {name=Tensor},
                   {name="index", default=lastdim(3)},
                   {name="boolean", default=0}})


   interface:wrap("tril",
                  cname("tril"),
                  {{name=Tensor, default=true, returned=true},
                   {name=Tensor},
                   {name="int", default=0}})

   interface:wrap("triu",
                  cname("triu"),
                  {{name=Tensor, default=true, returned=true},
                   {name=Tensor},
                   {name="int", default=0}})

   interface:wrap("cat",
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

   interface:wrap('random',
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
                  {{name=Tensor},
                   {name='long'},
                   {name='long'}},
                  cname("random1__"),
                  {{name=Tensor},
                   {name='long'}},
                  cname("random"),
                  {{name=Tensor}})

   for _,f in ipairs({{name='geometric'},
                      {name='bernoulli', a=0.5}}) do
      
      interface:wrap(f.name,
                     string.format("THRandom_%s", f.name),
                     {{name="double", default=f.a},
                      {name="double", creturned=true}},
                     cname(f.name),
                     {{name=Tensor, returned=true},
                      {name=real, default=f.a}})
   end

   interface:wrap("squeeze",
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

   interface:wrap("sign",
		  cname("sign"),
		  {{name=Tensor, default=true, returned=true},
		   {name=Tensor}})

   interface:wrap("conv2",
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

   interface:wrap("xcorr2",
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

   interface:wrap("conv3",
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

   interface:wrap("xcorr3",
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

   if Tensor == 'FloatTensor' or Tensor == 'DoubleTensor' then

      interface:wrap("mean",
                     cname("mean"),
                     {{name=Tensor, default=true, returned=true},
                      {name=Tensor},
                      {name="index", default=lastdim(2)}})

      interface:wrap("std",
                     cname("std"),
                     {{name=Tensor, default=true, returned=true},
                      {name=Tensor},
                      {name="index", default=lastdim(2)},
                      {name="boolean", default=false}})

      interface:wrap("var",
                     cname("var"),
                     {{name=Tensor, default=true, returned=true},
                      {name=Tensor},
                      {name="index", default=lastdim(2)},
                      {name="boolean", default=false}})

      interface:wrap("norm",
                     cname("norm"),
                     {{name=Tensor},
                      {name=real, default=2},
                      {name=real, creturned=true}})

      interface:wrap("dist",
                     cname("dist"),
                     {{name=Tensor},
                      {name=Tensor},
                      {name=real, default=2},
                      {name=real, creturned=true}})

      for _,name in ipairs({"meanall", "varall", "stdall"}) do
         interface:wrap(name,
                        cname(name),
                        {{name=Tensor},
                         {name=real, creturned=true}})
      end

      interface:wrap("linspace",
                     cname("linspace"),
                     {{name=Tensor, default=true, returned=true},
                      {name=real},
                      {name=real},
                      {name="long", default=100}})

      interface:wrap("logspace",
                     cname("logspace"),
                     {{name=Tensor, default=true, returned=true},
                      {name=real},
                      {name=real},
                      {name="long", default=100}})

      for _,name in ipairs({"log", "log1p", "exp",
                            "cos", "acos", "cosh",
                            "sin", "asin", "sinh",
                            "tan", "atan", "tanh",
                            "sqrt",
                            "ceil", "floor",
                            "abs"}) do

         interface:wrap(name,
                        cname(name),
                        {{name=Tensor, default=true, returned=true},
                         {name=Tensor}},
                        name,
                        {{name=real},
                         {name=real, creturned=true}})
         
      end

      interface:wrap("pow",
                     cname("pow"),
                     {{name=Tensor, default=true, returned=true},
                      {name=Tensor},
                      {name=real}},
                     "pow",
                     {{name=real},
                      {name=real},
                      {name=real, creturned=true}})

      interface:wrap("rand",
                     cname("rand"),
                     {{name=Tensor, default=true, returned=true},
                      {name="LongArg"}})

      interface:wrap("randn",
                     cname("randn"),
                     {{name=Tensor, default=true, returned=true},
                      {name="LongArg"}})

      for _,f in ipairs({{name='uniform', a=0, b=1},
                         {name='normal', a=0, b=1},
                         {name='cauchy', a=0, b=1},
                         {name='logNormal', a=1, b=2}}) do
         
         interface:wrap(f.name,
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
         
         interface:wrap(f.name,
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

      interface:wrap("eig",
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
      
   end

   interface:register(string.format("torch_%sMath__", Tensor))

   interface:print(string.gsub([[
static void torch_TensorMath_init(lua_State *L)
{
  torch_Tensor_id = luaT_checktypename2id(L, "torch.Tensor");
  torch_LongStorage_id = luaT_checktypename2id(L, "torch.LongStorage");

  /* register everything into the "torch" field of the tensor metaclass */
  luaT_pushmetaclass(L, torch_Tensor_id);
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
   interface:tostdio()
end
