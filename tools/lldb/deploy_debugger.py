import lldb  # type: ignore[import]


# load into lldb instance with:
#   command script import tools/lldb/deploy_debugger.py

target = lldb.debugger.GetSelectedTarget()
bp = target.BreakpointCreateByRegex("__deploy_register_code")
bp.SetScriptCallbackBody(
    """\
process = frame.thread.GetProcess()
target = process.target
symbol_addr = frame.module.FindSymbol("__deploy_module_info").GetStartAddress()
info_addr = symbol_addr.GetLoadAddress(target)
e = lldb.SBError()
ptr_size = 8
str_addr = process.ReadPointerFromMemory(info_addr, e)
file_addr = process.ReadPointerFromMemory(info_addr + ptr_size, e)
file_size = process.ReadPointerFromMemory(info_addr + 2*ptr_size, e)
load_bias = process.ReadPointerFromMemory(info_addr + 3*ptr_size, e)
name = process.ReadCStringFromMemory(str_addr, 512, e)
r = process.ReadMemory(file_addr, file_size, e)
from tempfile import NamedTemporaryFile
from pathlib import Path
stem = Path(name).stem
with NamedTemporaryFile(prefix=stem, suffix='.so', delete=False) as tf:
    tf.write(r)
    print("torch_deploy registering debug information for ", tf.name)
    cmd1 = f"target modules add {tf.name}"
    # print(cmd1)
    lldb.debugger.HandleCommand(cmd1)
    cmd2 = f"target modules load -f {tf.name} -s {hex(load_bias)}"
    # print(cmd2)
    lldb.debugger.HandleCommand(cmd2)

return False
"""
)
