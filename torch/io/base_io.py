class BaseIO():
    def __init__(self):
        pass

    def _readline(self, read):
        # line terminator is always b'\n' for binary files
        line = bytearray()
        while True:
            char = read(1)
            if char == b'\n':
                line += b'\n'
                break

            if char == b'':
                break

            line += char

        return line