for _,tensortype in ipairs({'ByteTensor',
                      'CharTensor',
                      'ShortTensor',
                      'IntTensor',
                      'LongTensor',
                      'FloatTensor',
                      'DoubleTensor'}) do

   for _,func in ipairs({'add',
                         'mul',
                         'div',
                         'cmul',
                         'cdiv',
                         'addcmul',
                         'addcdiv',
                         'log',
                         'log1p',
                         'exp',
                         'cos',
                         'acos',
                         'cosh',
                         'sin',
                         'asin',
                         'sinh',
                         'tan',
                         'atan',
                         'tanh',
                         'pow',
                         'sqrt',
                         'ceil',
                         'floor',
                         'abs',
			 'sign'
                      }) do

      local torchfunc = torch[tensortype].torch[func]
      torch[tensortype][func] = function(self, ...)
                             return torchfunc(self, self, ...)
                          end      
   end

   for _,func in ipairs({'addmv',
                         'addmm',
                         'addr'}) do
      
      local torchfunc = torch[tensortype].torch[func]
      torch[tensortype][func] = function(self, next1, next2, ...)
                                   if type(next1) == 'number' and type(next2) == 'number' then
                                      return torchfunc(self, next1, self, next2, ...)
                                   elseif type(next1) == 'number' then
                                      return torchfunc(self, self, next1, next2, ...)                                      
                                   else
                                      return torchfunc(self, self, next1, next2, ...)
                                   end
                          end      
   end

   for _,func in ipairs({'zero',
                         'fill',
                         'dot',
                         'minall',
                         'maxall',
                         'sumall',                         
                         'numel',
                         'max',
                         'min',
                         'sum',
                         'prod',
                         'cumsum',
                         'cumprod',
                         'trace',
                         'cross',
                         'zeros',
                         'ones',
                         'diag',
                         'eye',
                         'range',
                         'randperm',
                         'reshape',
                         'sort',
                         'tril',
                         'triu',
                         '_histc',
                         'cat',
                         'mean',
                         'std',
                         'var',
                         'norm',
                         'dist',
                         'meanall',
                         'varall',
                         'stdall',
                         'linspace',
                         'logspace',
                         'rand',
                         'randn',
                         'random',
                         'uniform',
                         'normal',
                         'cauchy',
                         'logNormal',
                         'exponential',
                         'geometric',
                         'bernoulli',
                         'squeeze'
                      }) do

      torch[tensortype][func] = torch[tensortype].torch[func]
   end
end
