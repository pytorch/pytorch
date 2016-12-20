' creating configuration storage and initializing with default values
Set cfg = CreateObject("Scripting.Dictionary")

' reading ini file
for i = 0 to ubound(ini_strings)
    s = trim(ini_strings(i))

    ' skipping empty strings and comments
    if mid(s, 1, 1) <> "#" and len(s) > 0 then
      ' obtaining key and value
      parts = split(s, "=", -1, 1)

      if ubound(parts)+1 = 2 then
        parts(0) = trim(parts(0))
        parts(1) = trim(parts(1))

        ' reading configuration and filenames
        select case lcase(parts(0))
          case "uncompressed""_postfix" cfg.item("uncompressed""_postfix") = parts(1)
          case "f"
                    options = split(parts(1), "|", -1, 1)
                    if ubound(options)+1 = 2 then
                      ' 0: filename,  1: options
                      ff.add trim(options(0)), trim(options(1))
                    end if
        end select
      end if
    end if
next