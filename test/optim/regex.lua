assert(arg[1])
funcs = {
    'resizeAs', 'add', 'zero', 'mul', 'div', 'abs',
    'addcmul', 'addcdiv', 'copy', 'sqrt', 'fill',
    {'cmul', 'mul'},
    {'cdiv', 'div'},
}
for _, val in pairs(funcs) do
    local name, newname
    if type(val) == 'table' then
        name = val[1]
        newname = val[2]
    else
        name = val
        newname = val .. '_'
    end

    command = "sed -i -r "
        .. "'/torch\\." .. name .. "\\(/b; " -- short-circuits
        .. "s/([a-zA-Z]*)\\." .. name .. "\\(" -- substitution
        .. "/"
        .. "\\1\\." .. newname .. "\\(/g' " .. arg[1]
    print(command)
    os.execute(command)
    command = "sed -i 's/math\\." .. newname
        .. "/math\\." .. name .. "/' " .. arg[1]
    print(command)
    os.execute(command)
end

funcs = {
    {'torch\.cmul', 'torch\.mul'},
    {'torch\.cdiv', 'torch\.div'},
}
for _, val in pairs(funcs) do
    command = "sed -i 's/" .. val[1] .. "/" .. val[2] .. "/' " .. arg[1]
    print(command)
    os.execute(command)
end
