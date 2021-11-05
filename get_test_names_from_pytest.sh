# TODO: figure out why "pytest" is not 
# /fsx/users/rvarm1/conda/envs/pytorch/bin/pytest in bash script.
# Does bash script run in conda env automatically pick up the conda pytest? 


# List pytest tests and write them to a file
WORLD_SIZE=2 BACKEND=gloo /fsx/users/rvarm1/conda/envs/pytorch/bin/pytest test/distributed/test_distributed_spawn.py -v --collect-only > dist_tests
# Grab an array for all the tests, they are in the format like
# <TestCaseFunction test_*>
all_lines=$(cat dist_tests | grep "<TestCaseFunction*")
stringarray=($all_lines)

TEST_PREFIX='test_'

# Iterate through the array and print out the real test names
# that we can invoke pytest -k with.
for cand_test in "${stringarray[@]}"
do
    if [[ "$cand_test" == *"$TEST_PREFIX"* ]]; then
        # Remove the last >. For some reason putting
        # cand_test after sed doesnt work, but the echo does.
        test_name=$(echo $cand_test | sed "s/.$//")
        # TODO - we have to provide the name such aas test_mod.py::TestClass::test_method otherwise pytest -k test_name will run tests such as
        # test_name, test_name_foo, etc together.
#        test_name=$(sed -e "s/.$//" "$cand_test")
        echo $test_name
        export BACKEND=gloo
        export WORLD_SIZE=2
        pytest test/distributed/test_distributed_spawn.py -v -k $test_name
    fi
done
