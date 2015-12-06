# Run get_protobuf.sh to update the protocol buffer to the most recent version.

rm -rf protobuf
wget https://github.com/google/protobuf/archive/master.zip
unzip master.zip
mv protobuf-master/src/google/protobuf ./

# Do not overwrite arena.h for now. This may be buggy if the upstream arena.h
# is changed, but it is a temporary solution before the PR
#     https://github.com/google/protobuf/pull/1025
# is merged.
git checkout protobuf/arena.h

rm master.zip
rm -rf protobuf-master