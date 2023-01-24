apt-get update
apt-get install -y sudo wget libboost-dev libboost-test-dev libboost-program-options-dev libboost-filesystem-dev libboost-thread-dev libevent-dev automake libtool flex bison pkg-config g++ libssl-dev
wget https://www-us.apache.org/dist/thrift/0.12.0/thrift-0.12.0.tar.gz
tar -xvf thrift-0.12.0.tar.gz
cd thrift-0.12.0
for file in ./compiler/cpp/Makefile*; do
  sed -i 's/\-Werror//' $file
done
./bootstrap.sh
./configure --without-php --without-java --without-python --without-nodejs --without-go --without-ruby
sudo make
sudo make install
cd ..
rm thrift-0.12.0.tar.gz
