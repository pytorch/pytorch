import collections


Entry = collections.namedtuple('Entry', 'version, hash')


def update_hash(seed, value):
    # Good old boost::hash_combine
    # https://www.boost.org/doc/libs/1_35_0/doc/html/boost/hash_combine_id241013.html
    return seed ^ (hash(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2))


def hash_source_files(hash_value, source_files):
    for filename in source_files:
        with open(filename) as file:
            hash_value = update_hash(hash_value, file.read())
    return hash_value


def hash_build_arguments(hash_value, build_arguments):
    for group in build_arguments:
        if group:
            for argument in group:
                hash_value = update_hash(hash_value, argument)
    return hash_value


class ExtensionVersioner(object):
    def __init__(self):
        self.entries = {}

    def get_version(self, name):
        entry = self.entries.get(name)
        return None if entry is None else entry.version

    def bump_version_if_changed(self,
                                name,
                                source_files,
                                build_arguments,
                                build_directory,
                                with_cuda):
        hash_value = 0
        hash_value = hash_source_files(hash_value, source_files)
        hash_value = hash_build_arguments(hash_value, build_arguments)
        hash_value = update_hash(hash_value, build_directory)
        hash_value = update_hash(hash_value, with_cuda)

        entry = self.entries.get(name)
        if entry is None:
            self.entries[name] = entry = Entry(0, hash_value)
        elif hash_value != entry.hash:
            self.entries[name] = entry = Entry(entry.version + 1, hash_value)

        return entry.version
