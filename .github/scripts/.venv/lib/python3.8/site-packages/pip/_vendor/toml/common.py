# content after the \
escapes = ['0', 'b', 'f', 'n', 'r', 't', '"']
# What it should be replaced by
escapedchars = ['\0', '\b', '\f', '\n', '\r', '\t', '\"']
# Used for substitution
escape_to_escapedchars = dict(zip(_escapes, _escapedchars))
