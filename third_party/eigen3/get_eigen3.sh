# Run get_eigen3.sh to update the protocol buffer to the most recent version.
# Current Eigen version:
#    Eigen 3.3 alpha 1

rm -rf Eigen
rm -rf unsupported

wget http://bitbucket.org/eigen/eigen/get/3.3-alpha1.tar.gz
tar xzvf *.tar.gz
mv eigen-eigen-*/Eigen .
mv eigen-eigen-*/unsupported .
rm -rf eigen-eigen*
rm *.tar.gz
