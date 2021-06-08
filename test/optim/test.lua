local cjson = require 'cjson'
require 'optim'

function rosenbrock(t)
    x, y = t[1], t[2]
    return (1 - x) ^ 2 + 100 * (y - x^2)^2
end

function drosenbrock(t)
    x, y = t[1], t[2]
    return torch.DoubleTensor({-400 * x * (y - x^2) - 2 * (1 - x), 200 * x * (y - x^2)})
end

local fd = io.open('tests.json', 'r')
local tests = cjson.decode(fd:read('*a'))
fd:close()

for i, test in ipairs(tests) do
    print(test.algorithm)
    algorithm = optim[test.algorithm]
    for i, config in ipairs(test.config) do
        print('================================================================================')
        params = torch.DoubleTensor({1.5, 1.5})
        for i = 1, 100 do
            function closure(x)
                return rosenbrock(x), drosenbrock(x)
            end
            algorithm(closure, params, config)
            print(string.format('%.8f\t%.8f', params[1], params[2]))
        end
    end
end
