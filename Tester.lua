local Tester = torch.class('torch.Tester')

function Tester:__init()
   self.errors = {}
   self.tests = {}
   self.testnames = {}
   self.curtestname = ''
end


function Tester:assert_sub (condition, message)
   self.countasserts = self.countasserts + 1
   if not condition then
      local ss = debug.traceback('tester',2)
      --print(ss)
      ss = ss:match('[^\n]+\n[^\n]+\n([^\n]+\n[^\n]+)\n')
      self.errors[#self.errors+1] = self.curtestname .. '\n' .. message .. '\n' .. ss .. '\n'
   end
end

function Tester:assert (condition, message)
   self:assert_sub(condition,string.format('%s\n%s  condition=%s',message,' BOOL violation ', tostring(condition)))
end

function Tester:assertlt (val, condition, message)
   self:assert_sub(val<condition,string.format('%s\n%s  val=%s, condition=%s',message,' LT(<) violation ', tostring(val), tostring(condition)))
end

function Tester:assertgt (val, condition, message)
   self:assert_sub(val>condition,string.format('%s\n%s  val=%s, condition=%s',message,' GT(>) violation ', tostring(val), tostring(condition)))
end

function Tester:assertle (val, condition, message)
   self:assert_sub(val<=condition,string.format('%s\n%s  val=%s, condition=%s',message,' LE(<=) violation ', tostring(val), tostring(condition)))
end

function Tester:assertge (val, condition, message)
   self:assert_sub(val>=condition,string.format('%s\n%s  val=%s, condition=%s',message,' GE(>=) violation ', tostring(val), tostring(condition)))
end

function Tester:asserteq (val, condition, message)
   self:assert_sub(val==condition,string.format('%s\n%s  val=%s, condition=%s',message,' EQ(==) violation ', tostring(val), tostring(condition)))
end

function Tester:assertne (val, condition, message)
   self:assert_sub(val~=condition,string.format('%s\n%s  val=%s, condition=%s',message,' NE(~=) violation ', tostring(val), tostring(condition)))
end

function Tester:assertTensorEq(ta, tb, condition, message)
   local diff = ta-tb
   local err = diff:abs():max()
   self:assert_sub(err<condition,string.format('%s\n%s  val=%s, condition=%s',message,' TensorEQ(~=) violation ', tostring(err), tostring(condition)))
end

function Tester:assertTableEq(ta, condition, message)
   self:assert_sub(unpack(ta) == unpack(condition), string.format('%s\n%s val=%s, condition=%s',message,' TableEQ(~=) violation ', tostring(err), tostring(condition)))
end

function Tester:pcall(f)
   local nerr = #self.errors
   local res = f()
--    local stat, result = pcall(f)
--    if not stat then
--       result = result .. debug.traceback()
--    end
--    return stat, result, stat and (nerr == #self.errors)
   return true, res, nerr == #self.errors
end

function Tester:report(tests)
   if not tests then
      tests = self.tests
   end
   print('Completed ' .. self.countasserts .. ' asserts in ' .. #tests .. ' tests with ' .. #self.errors .. ' errors')
   print()
   print(string.rep('-',80))
   for i,v in ipairs(self.errors) do
      print(v)
      print(string.rep('-',80))
   end
end

function Tester:run(run_tests)
   local tests, testnames
   self.countasserts = 0
   tests = self.tests
   testnames = self.testnames
   if type(run_tests) == 'string' then
      run_tests = {run_tests}
   end
   if type(run_tests) == 'table' then
      tests = {}
      testnames = {}
      for i,fun in ipairs(self.tests) do
         for j,name in ipairs(run_tests) do
            if self.testnames[i] == name then
               tests[#tests+1] = self.tests[i]
               testnames[#testnames+1] = self.testnames[i]
            end
         end
      end
   end

   print('Running ' .. #tests .. ' tests')
   local statstr = string.rep('_',#tests)
   local pstr = ''
   io.write(statstr .. '\r')
   for i,v in ipairs(tests) do
      self.curtestname = testnames[i]
      
      --clear
      io.write('\r' .. string.rep(' ', pstr:len()))
      io.flush()
      --write
      pstr = statstr:sub(1,i-1) .. '|' .. statstr:sub(i+1) .. '  ==> ' .. self.curtestname
      io.write('\r' .. pstr)
      io.flush()
      
      local stat, message, pass = self:pcall(v)
      
      if pass then
         --io.write(string.format('\b_'))
         statstr = statstr:sub(1,i-1) .. '_' .. statstr:sub(i+1)
      else
         statstr = statstr:sub(1,i-1) .. '*' .. statstr:sub(i+1)
         --io.write(string.format('\b*'))
      end
      
      if not stat then
         print()
         print('Function call failed: Test No ' .. i .. ' ' .. testnames[i])
         print(message)
      end
      collectgarbage()
   end
   --clear
   io.write('\r' .. string.rep(' ', pstr:len()))
   io.flush()
   -- write finish
   pstr = statstr .. '  ==> Done '
   io.write('\r' .. pstr)
   io.flush()
   print()
   print()
   self:report(tests)
end

function Tester:add(f,name)
   name = name or 'unknown'
   if type(f) == "table" then
      for i,v in pairs(f) do
         self:add(v,i)
      end
   elseif type(f) == "function" then
      self.tests[#self.tests+1] = f
      self.testnames[#self.tests] = name
   else
      error('Tester:add(f) expects a function or a table of functions')
   end
end
