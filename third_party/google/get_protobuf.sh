# Run get_protobuf.sh to update the protocol buffer to the most recent version.

rm -rf protobuf
wget https://github.com/google/protobuf/archive/master.zip
unzip master.zip
mv protobuf-master/src/google/protobuf ./
rm master.zip
rm -rf protobuf-master