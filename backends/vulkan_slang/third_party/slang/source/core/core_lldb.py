"""
This python script provides LLDB formatters for Slang core types.
To use it, add the following line to your ~/.lldbinit file:
command script import /path/to/source/core/core_lldb.py
"""

import lldb  # type: ignore[import]

# Set to True to enable the logger
ENABLE_LOGGING = True


# log to the LLDB formatter stream
def log(msg):
    if ENABLE_LOGGING:
        lldb.formatters.Logger.Logger() >> msg


def make_string(F, L):
    strval = ""
    G = F.uint8
    for X in range(L):
        V = G[X]
        if V == 0:
            break
        strval = strval + chr(V % 256)
    return '"' + strval + '"'


# Return the pointer to the data in a Slang::RefPtr
def get_ref_pointer(valobj):
    return valobj.GetNonSyntheticValue().GetChildMemberWithName("pointer")


# Check if a pointer is nullptr
def is_nullptr(valobj):
    return valobj.GetValueAsUnsigned(0) == 0


# Slang::String summary
def String_summary(valobj, dict):
    buffer_ptr = get_ref_pointer(valobj.GetChildMemberWithName("m_buffer"))
    if is_nullptr(buffer_ptr):
        return '""'
    buffer = buffer_ptr.Dereference()
    length = buffer.GetChildMemberWithName("length").GetValueAsUnsigned(0)
    data = buffer_ptr.GetPointeeData(1, length)
    return make_string(data, length)


# Slang::UnownedStringSlice summary
def UnownedStringSlice_summary(valobj, dict):
    begin = valobj.GetChildMemberWithName("m_begin")
    end = valobj.GetChildMemberWithName("m_end")
    length = end.GetValueAsUnsigned(0) - begin.GetValueAsUnsigned(0)
    if length <= 0:
        return '""'
    data = begin.GetPointeeData(0, length)
    return make_string(data, length)


# Slang::RefPtr synthetic provider
class RefPtr_synthetic:
    def __init__(self, valobj, dict):
        self.valobj = valobj

    def has_children(self):
        return True

    def num_children(self):
        return len(self.children)

    def get_child_index(self, name):
        for index in range(self.num_children()):
            if self.children[index].GetName() == name:
                return index
        return -1

    def get_child_at_index(self, index):
        if index >= 0 and index < self.num_children():
            return self.children[index]
        else:
            return None

    def update(self):
        self.pointer = self.valobj.GetNonSyntheticValue().GetChildMemberWithName(
            "pointer"
        )
        self.children = []
        if not is_nullptr(self.pointer):
            self.children = self.pointer.Dereference().children


# Slang::RefPtr summary
def RefPtr_summary(valobj, dict):
    pointer = valobj.GetNonSyntheticValue().GetChildMemberWithName("pointer")
    if is_nullptr(pointer):
        return "nullptr"
    pointee = pointer.Dereference()
    refcount = pointee.GetChildMemberWithName("referenceCount").GetValueAsUnsigned()
    return str(pointer.GetValue()) + " refcount=" + str(refcount)


# Slang::ComPtr synthetic provider
class ComPtr_synthetic:
    def __init__(self, valobj, dict):
        self.valobj = valobj

    def has_children(self):
        return len(self.children) > 0

    def num_children(self):
        return len(self.children)

    def get_child_index(self, name):
        for index in range(self.num_children()):
            if self.children[index].GetName() == name:
                return index
        return -1

    def get_child_at_index(self, index):
        if index >= 0 and index < self.num_children():
            return self.children[index]
        else:
            return None

    def update(self):
        self.pointer = self.valobj.GetChildMemberWithName("m_ptr")
        self.children = []
        if not is_nullptr(self.pointer):
            self.children = self.pointer.Dereference().children


# Slang::ComPtr summary
def ComPtr_summary(valobj, dict):
    pointer = valobj.GetNonSyntheticValue().GetChildMemberWithName("m_ptr")
    if is_nullptr(pointer):
        return "nullptr"
    return str(pointer.GetValue())


# Slang::Array synthetic provider
class Array_synthetic:
    def __init__(self, valobj, dict):
        self.valobj = valobj

    def has_children(self):
        return True

    def num_children(self):
        return self.count.GetValueAsUnsigned(0)

    def get_child_index(self, name):
        return int(name.lstrip("[").rstrip("]"))

    def get_child_at_index(self, index):
        if index >= 0 and index < self.num_children():
            offset = index * self.data_size
            return self.buffer.CreateChildAtOffset(
                "[" + str(index) + "]", offset, self.data_type
            )
        else:
            return None

    def update(self):
        self.count = self.valobj.GetChildMemberWithName("m_count")
        self.buffer = self.valobj.GetChildMemberWithName("m_buffer")
        self.data_type = self.buffer.GetType().GetArrayElementType()
        self.data_size = self.data_type.GetByteSize()


# Slang::List synthetic provider
class List_synthetic:
    def __init__(self, valobj, dict):
        self.valobj = valobj

    def has_children(self):
        return True

    def num_children(self):
        return self.count.GetValueAsUnsigned(0)

    def get_child_index(self, name):
        return int(name.lstrip("[").rstrip("]"))

    def get_child_at_index(self, index):
        if index >= 0 and index < self.num_children():
            offset = index * self.data_size
            return self.buffer.CreateChildAtOffset(
                "[" + str(index) + "]", offset, self.data_type
            )
        else:
            return None

    def update(self):
        self.count = self.valobj.GetChildMemberWithName("m_count")
        self.buffer = self.valobj.GetChildMemberWithName("m_buffer")
        self.data_type = self.buffer.GetType().GetPointeeType()
        self.data_size = self.data_type.GetByteSize()


# Slang::ShortList synthetic provider
class ShortList_synthetic:
    def __init__(self, valobj, dict):
        self.valobj = valobj

    def has_children(self):
        return True

    def num_children(self):
        return self.count.GetValueAsUnsigned(0)

    def get_child_index(self, name):
        return int(name.lstrip("[").rstrip("]"))

    def get_child_at_index(self, index):
        if index >= 0 and index < self.short_count:
            offset = index * self.data_size
            return self.short_buffer.CreateChildAtOffset(
                "[" + str(index) + "]", offset, self.data_type
            )
        elif index >= self.short_count and index < self.num_children():
            offset = (index - self.short_count) * self.data_size
            return self.buffer.CreateChildAtOffset(
                "[" + str(index) + "]", offset, self.data_type
            )
        else:
            return None

    def update(self):
        self.count = self.valobj.GetChildMemberWithName("m_count")
        self.buffer = self.valobj.GetChildMemberWithName("m_buffer")
        self.short_buffer = self.valobj.GetChildMemberWithName("m_shortBuffer")
        self.short_count = self.short_buffer.GetNumChildren()
        self.data_type = self.buffer.GetType().GetPointeeType()
        self.data_size = self.data_type.GetByteSize()


def __lldb_init_module(debugger, internal_dict):
    if ENABLE_LOGGING:
        lldb.formatters.Logger._lldb_formatters_debug_level = 2

    commands = [
        # Slang::String
        "type summary add Slang::String -F core_lldb.String_summary -w slang",
        # Slang::UnownedStringSlice
        "type summary add Slang::UnownedStringSlice -F core_lldb.UnownedStringSlice_summary -w slang",
        # Slang::RefPtr
        'type synthetic add -x "^Slang::RefPtr<.+>$" -l core_lldb.RefPtr_synthetic -w slang',
        'type summary add -x "^Slang::RefPtr<.+>$" -F core_lldb.RefPtr_summary -w slang',
        # Slang::ComPtr
        'type synthetic add -x "^Slang::ComPtr<.+>$" -l core_lldb.ComPtr_synthetic -w slang',
        'type summary add -x "^Slang::ComPtr<.+>$" -F core_lldb.ComPtr_summary -w slang',
        # Slang::Array
        'type synthetic add -x "^Slang::Array<.+>$" -l core_lldb.Array_synthetic -w slang',
        'type summary add --x "^Slang::Array<.+>$" --summary-string "size=${svar%#}" -w slang',
        # Slang::List
        'type synthetic add -x "^Slang::List<.+>$" -l core_lldb.List_synthetic -w slang',
        'type summary add --expand -x "^Slang::List<.+>$" --summary-string "size=${var.m_count} capacity=${var.m_capacity}" -w slang',
        # Slang::ShortList
        'type synthetic add -x "^Slang::ShortList<.+>$" -l core_lldb.ShortList_synthetic -w slang',
        'type summary add --expand -x "^Slang::ShortList<.+>$" --summary-string "size=${var.m_count} capacity=${var.m_capacity}" -w slang',
        # Enable slang category
        "type category enable slang",
    ]

    for c in commands:
        debugger.HandleCommand(c)
