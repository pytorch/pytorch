/*
    Copyright (c) 2005-2018 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.




*/

#if __TBB_TEST_USE_WSUGGEST_OVERRIDE
// __TBB_override may not be used in the tested header file
#pragma GCC diagnostic ignored "-Wsuggest-override"
#undef __TBB_TEST_USE_WSUGGEST_OVERRIDE
#endif

#include "harness_defs.h" // for suppress_unused_warning

#if TBB_USE_EXCEPTIONS
#include "harness_assert.h"
#include "../../examples/common/utility/utility.h"
#include <sstream>

namespace implementation_unit_tests {
    namespace argument_dest_test_suite{
        void test_type_impl_parse_and_store_simple_parse(){
            int a=0;
            utility::internal::type_impl<int> a_("","",a);
            a_.parse_and_store("9");
            ASSERT(a==9,"");
        }
        void test_default_value_of_is_matched(){
            //Testing for result of is_matched() for arguments not yet tried to be parsed.
            //I.e. values were set up by argument::constructor.
            using  utility::internal::argument;
            int i;
            argument b("","",i);
            ASSERT(!b.is_matched(),"");

            argument c = b;
            ASSERT(!c.is_matched(),"");

            argument d = b;
            d = c;
            ASSERT(!d.is_matched(),"");
        }
    }
    //TODO: test cases  for argument type management
    namespace compile_only{
        //TODO: enhance these to actually do checks  by a negative test, or (if possible)
        //by a positive test that at compile time selects between two alternatives,
        //depending on whether operators exist or not (yes, SFINAE :)) -
        //as non_pod class does provide the operators, and test  do not check that compiler
        //will reject types which don't have those.
        using utility::cli_argument_pack;
        void arg_chain(){
            cli_argument_pack p;
            int size=0;
            p.arg(size,"size","size");
        }
        namespace tc_helper{
            struct non_pod{
                std::string s;
                friend std::ostream& operator<<(std::ostream& o, non_pod){ return o;}
                friend std::istream& operator>>(std::istream& i, non_pod){ return i;}
            };
        }
        void non_pod_dest_type(){
            cli_argument_pack p;
            tc_helper::non_pod np;
            p.arg(np,"","");
        }
    }
    namespace cli_argument_pack_suite{
        void test_copy_assign(){
            using utility::cli_argument_pack;
            int i=9;
            std::stringstream expected_output; using std::endl;
            expected_output
                 << " Program usage is:" << endl
                 << " the_binary_name [i=value]"
                 << endl << endl
                 << " where:" << endl
                 << " i - i desc (9)" << endl
            ;
            cli_argument_pack copy(cli_argument_pack().arg(i,"i","i desc"));
            ASSERT(copy.usage_string("the_binary_name") == expected_output.str(),"usage string is not as expected");
            cli_argument_pack assignee; assignee = copy;
            ASSERT(assignee.usage_string("the_binary_name") == expected_output.str(),"Copying of cli_argument_pack breaks generation of usage string?");
        }
    }
}

#include <utility>
namespace high_level_api_tests {
    using utility::cli_argument_pack;
    using utility::internal::array_length;

    static const char * wrong_exception = "wrong exception thrown";
    static const char * wrong_exception_description = "caught exception has wrong description";
    void test_parse_basic(){
        char const* argv[]={"some.exe","1","a"};
        cli_argument_pack p;
        int i=0; char a=' ';
        p.positional_arg(i,"int","").positional_arg(a,"char","");
        p.parse(array_length(argv),argv);
        ASSERT(i==1,"");
        ASSERT(a=='a',"");
    }
    //helper function for test of named flag parsing
    template<typename T, size_t N>
    bool parse_silent_flag( T(& argv)[N]){
        cli_argument_pack p;
        bool silent=false;
        p.arg(silent,"silent","is extra info needed");
        p.parse(array_length(argv),argv);
        return  silent;
    }
    void test_named_flags_success(){
        char const* argv[]={"some.exe","silent"};
        ASSERT(true == parse_silent_flag(argv),"");
    }

    void test_named_flags_failure(){
        try {
            char const* argv[]={"some.exe","1"};
            parse_silent_flag(argv);
            ASSERT(false,"exception was expected due to invalid argument, but not caught");
        }
        catch(std::invalid_argument& e){
            ASSERT(e.what()==std::string("unknown parameter starting at:'1'"),wrong_exception_description);
        }
        catch(...){ASSERT(false,wrong_exception);}
    }

    //helper function for test of named flag parsing
    template<typename T, size_t N>
    std::pair<bool,int> parse_silent_flag_and_int( T(& argv)[N]){
        cli_argument_pack p;
        bool silent=false;
        int i=125;
        p
            .arg(silent,"silent","is extra info needed")
            .positional_arg(i,"int","");
        p.parse(array_length(argv),argv);
        return  std::make_pair(silent,i);
    }

    void test_named_flags_failure_and_other_arg(){
        char const* argv[]={"some.exe","1"};
        ASSERT(std::make_pair(false,1) == parse_silent_flag_and_int(argv),"");
    }

    void test_named_flags_and_other_arg(){
        char const* argv[]={"some.exe","silent","7"};
        ASSERT(std::make_pair(true,7) == parse_silent_flag_and_int(argv),"");
    }

    void test_named_flags_and_other_arg_different_order(){
        char const* argv[]={"some.exe","7","silent"};
        ASSERT(std::make_pair(true,7) == parse_silent_flag_and_int(argv),"");
    }

    void test_flags_only_others_default(){
        char const* argv[]={"some.exe","silent"};
        ASSERT(std::make_pair(true,125) == parse_silent_flag_and_int(argv),"");
    }

    namespace parameters_validation_test_suite{
        namespace test_validation_function_called_helpers{
            struct validator{
                static bool called;
                static bool accept(const int & ){
                    called = true;
                    return true;
                }
            };
            bool validator::called =false;
        }
        void test_validation_function_called(){
            using test_validation_function_called_helpers::validator;

            char const* argv[]={"some.exe","7"};
            cli_argument_pack p;
            int size =0;
            p.positional_arg(size,"size","",validator::accept);
            p.parse(array_length(argv),argv);
            ASSERT((validator::called),"validation function has not been called");
        }
        void test_validation_failed(){
            struct validator{
                static bool reject(const int &){
                    return false;
                }
            };
            char const* argv[]={"some.exe","7"};
            cli_argument_pack p;
            int size =0;
            p.positional_arg(size,"size","",validator::reject);
            try {
                p.parse(array_length(argv),argv);
                ASSERT((false),"An exception was expected due to failed argument validation, "
                        "but no exception thrown");
            }
            catch(std::invalid_argument& e){
                std::string error_msg("'7' is invalid value for argument 'size'");
                ASSERT(e.what()==error_msg , wrong_exception_description);
            }
            catch(...){ASSERT((false),wrong_exception);}
        }
    }
    namespace error_handling {
        void test_wrong_input(){
            char const* argv[]={"some.exe","silent"};
            cli_argument_pack p;
            int size =0;
            p.positional_arg(size,"size","");
            try{
                p.parse(array_length(argv),argv);
                ASSERT(false,"An exception was expected due to wrong input, but no exception thrown");
            }
            catch(std::invalid_argument & e){
                std::string error_msg("'silent' is incorrect input for argument 'size' (error converting string 'silent')");
                ASSERT(e.what()==error_msg, wrong_exception_description);
            }
            catch(...){ASSERT(false,wrong_exception);}
        }
        void test_duplicate_arg_names(){
            cli_argument_pack p;
            int a=0;
            p.arg(a,"a","");
            try{
                int dup_a=0;
                p.arg(dup_a,"a","");
                ASSERT(false, "An exception was expected due adding duplicate parameter name, but not thrown");
            }
            catch(std::invalid_argument& e){
                ASSERT(e.what()==std::string("argument with name: 'a' already registered"),wrong_exception_description);
            }
            catch(...){ASSERT(false,wrong_exception);}
        }
        void test_duplicate_positional_arg_names(){
            cli_argument_pack p;
            int a=0;
            p.positional_arg(a,"a","");
            try{
                int dup_a=0;
                p.positional_arg(dup_a,"a","");
                ASSERT(false, "An exception was expected due adding duplicate parameter name, but not thrown");
            }
            catch(std::invalid_argument& e){
                ASSERT(e.what()==std::string("argument with name: 'a' already registered"),wrong_exception_description);
            }
            catch(...){ASSERT(false,wrong_exception);}
        }
    }
    namespace usage_string {
        void test_one_arg(){
            cli_argument_pack p;
            int size =9;
            p.arg(size,"size","size of problem domain");
            std::string const binary_name = "binary.exe";
            std::stringstream expected_output;
            using std::endl;
            expected_output << " Program usage is:" << endl
                 << " " << binary_name << " [size=value]"
                 << endl << endl
                 << " where:" << endl
                 << " size - size of problem domain (9)" << endl
            ;
            std::string usage= p.usage_string(binary_name);
            ASSERT(usage==expected_output.str(),"");
        }
        void test_named_and_postional_args(){
            cli_argument_pack p;
            int size =9;
            int length =8;
            int stride = 7;
            p
                .arg(size,"size","")
                .positional_arg(length,"length","")
                .positional_arg(stride,"stride","");
            std::string const binary_name = "binary.exe";
            std::stringstream expected_output;
            using std::endl;
            expected_output << " Program usage is:" << endl
                 << " " << binary_name << " [size=value] [length=value] [stride=value] [length [stride]]"
                 << endl << endl
                 << " where:" << endl
                 << " size -  (9)" << endl
                 << " length -  (8)" << endl
                 << " stride -  (7)" << endl
            ;
            std::string usage= p.usage_string(binary_name);
            ASSERT(usage==expected_output.str(),"");
        }
        void test_bool_flag(){
            bool flag=false;
            cli_argument_pack p;
            p.arg(flag,"flag","");
            std::string const binary_name = "binary.exe";
            std::stringstream expected_output;
            using std::endl;
            expected_output << " Program usage is:" << endl
                 << " " << binary_name << " [flag]"
                 << endl << endl
                 << " where:" << endl
                 << " flag -  (0)" << endl
            ;
            std::string usage= p.usage_string(binary_name);
            ASSERT(usage==expected_output.str(),"");

        }

    }
    namespace name_positional_syntax {
        void test_basic(){
            cli_argument_pack p;
            int size =0;
            int time = 0;
            p
                .positional_arg(size,"size","")
                .positional_arg(time,"time","");
            char const* argv[]={"some.exe","1","2"};
            p.parse(array_length(argv),argv);
            ASSERT(size==1,"");
            ASSERT(time==2,"");
        }
        void test_positional_args_explicitly_named(){
            const char* no_or_wrong_exception_error_msg = "exception was expected but not thrown, or wrong exception caught";
            //TODO: Similar functionality is used all over the test. Generalize this helper further, and use as wide within the test as possible?
            struct failed_with_exception{
                static bool _(cli_argument_pack & p, std::size_t argc, char const* argv[]){
                    try{
                        p.parse(argc,argv);
                        return false;
                    }
                    catch(std::exception &){
                        return true;
                    }
                    catch(...){
                        return false;
                    }
                }
            };
            {
                cli_argument_pack p;
                int a,b,c,d;
                p
                    .positional_arg(a,"a","")
                    .positional_arg(b,"b","")
                    .positional_arg(c,"c","")
                    .positional_arg(d,"d","");
                char const* argv[]={"some.exe","a=7","0","1","2","4"};
                ASSERT(failed_with_exception::_(p,array_length(argv),argv),no_or_wrong_exception_error_msg);
            }
            {
                cli_argument_pack p;
                int a,b,c,d;
                p
                    .positional_arg(a,"a","")
                    .positional_arg(b,"b","")
                    .positional_arg(c,"c","")
                    .positional_arg(d,"d","");
                char const* argv[]={"some.exe","a=7","0","1","2"};
                ASSERT(failed_with_exception::_(p,array_length(argv),argv),no_or_wrong_exception_error_msg);
            }
            {
                cli_argument_pack p;
                int a=-1,b=-1,c = -1,d=-1;
                p
                    .positional_arg(a,"a","")
                    .positional_arg(b,"b","")
                    .positional_arg(c,"c","")
                    .positional_arg(d,"d","");
                char const* argv[]={"some.exe","0","1","d=7",};
                ASSERT(!failed_with_exception::_(p,array_length(argv),argv),"unexpected exception");
                ASSERT(a==0,""); ASSERT(b==1,""); ASSERT(c==-1,"");ASSERT(d==7,"");
            }
        }
    }
    namespace name_value_syntax {
        void test_basic(){
            cli_argument_pack p;
            int size =0;
            p.arg(size,"size","size of problem domain");
            char const* argv[]={"some.exe","size=7"};
            p.parse(array_length(argv),argv);
            ASSERT(size==7,"");
        }

        void test_relaxed_order(){
            cli_argument_pack p;
            int size =0;
            int time=0;
            p
                .arg(size,"size","")
                .arg(time,"time","");
            char const* argv[]={"some.exe","time=1","size=2"};
            p.parse(array_length(argv),argv);
            ASSERT(size==2,"");
            ASSERT(time==1,"");
        }

    }
    namespace number_of_argument_value{
        void test_only_single_values_allowed(){
            cli_argument_pack p;
            int a=0;
            p.arg(a,"a","");
            const char* argv[] = {"","a=7","a=8"};
            try {
                p.parse(array_length(argv),argv);
                ASSERT(false,"exception was expected due to duplicated values provided in input, but not thrown");
            }
            catch(std::invalid_argument& e){
                //TODO: use patterns (regexp ?) to generate /validate exception descriptions
                ASSERT(e.what() == std::string("several values specified for: 'a' argument"),wrong_exception_description);
            }
            catch(...){ASSERT(false,wrong_exception);}
        }
    }
    namespace thread_range_tests{
        using utility::thread_number_range;
        using utility::internal::thread_range_step;
        using utility::internal::step_function_multiply;
        using utility::internal::step_function_plus;
        using utility::internal::step_function_power2_ladder;

        int auto_value(){
            return 100;
        }
        bool operator ==(thread_range_step const& left, utility::internal::thread_range_step const& right){
            return (left.step_function == right.step_function)
                   && (left.step_function_argument == right.step_function_argument)
            ;
        }

        bool operator ==(thread_number_range const& left, thread_number_range const& right){
            return (left.auto_number_of_threads==right.auto_number_of_threads)
                    && (left.first == right.first)
                    && (left.last == right.last)
                    && (left.step == right.step)
            ;
        }

        void constructor_default_values(){
            thread_number_range r(auto_value);
            const int default_num_threads = auto_value();
            ASSERT((r.first==1)&&(r.last==default_num_threads),"");
        }
        void validation(){
            try{
                thread_number_range range(auto_value,12,6);
                Harness::suppress_unused_warning(range);
                ASSERT(false,"exception was expected due to invalid range specified, but not thrown");
            }
            catch(std::invalid_argument& e){
                ASSERT(e.what() == std::string("decreasing sequence not allowed"), wrong_exception_description);
            }
            catch(...){ASSERT(false,wrong_exception);}
        }

        thread_number_range thread_number_range_from_string(std::string const& string_to_parse){
            thread_number_range r(auto_value,0,0);
            std::stringstream str(string_to_parse); str>>r;
            return r;
        }
        static const char* thread_range_parse_failed = "error parsing thread range string";
        void post_process_single_value(){
            ASSERT(thread_number_range_from_string("auto") ==
                    thread_number_range(auto_value,auto_value(),auto_value())
                  ,thread_range_parse_failed
            );
        }
        void post_process_pair_value(){
            ASSERT(thread_number_range_from_string("1:auto") ==
                    thread_number_range(auto_value,1,auto_value())
                  ,thread_range_parse_failed
            );

            ASSERT(thread_number_range_from_string("auto:auto") ==
                    thread_number_range(auto_value,auto_value(),auto_value())
                  ,thread_range_parse_failed
            );
        }

        void post_process_troika_value_with_plus_step(){
            ASSERT(thread_number_range_from_string("1:auto:+2") ==
                    thread_number_range(auto_value,1,auto_value(),thread_range_step(step_function_plus,2))
                  ,thread_range_parse_failed
            );
        }

        void post_process_troika_value_with_multiply_step(){
            ASSERT(thread_number_range_from_string("1:auto:*2.6") ==
                    thread_number_range(auto_value,1,auto_value(),thread_range_step(step_function_multiply,2.6))
                  ,thread_range_parse_failed
            );
        }

        void post_process_troika_value_with_ladder_step(){
            try{
                thread_number_range range = thread_number_range_from_string("1:16:#3");
                Harness::suppress_unused_warning(range);
                ASSERT(false,"exception was expected due to invalid range specified, but not thrown");
            }
            catch(std::invalid_argument& e){
                ASSERT(e.what() == std::string("the argument of # should be a power of 2"), wrong_exception_description);
            }
            catch(...){ASSERT(false,wrong_exception);}

            ASSERT(thread_number_range_from_string("1:32:#4") ==
                    thread_number_range(auto_value,1,32,thread_range_step(step_function_power2_ladder,4))
                  ,thread_range_parse_failed
            );
        }

        void test_print_content(){
            std::stringstream str;
            str<<thread_number_range(auto_value,1,8,thread_range_step(step_function_multiply,2));
            ASSERT(str.str() == "1:8:*2","Unexpected string");
        }
    }
}

void run_implementation_unit_tests(){
    using namespace implementation_unit_tests;
    argument_dest_test_suite::test_type_impl_parse_and_store_simple_parse();
    argument_dest_test_suite::test_default_value_of_is_matched();

    cli_argument_pack_suite::test_copy_assign();
}
void run_high_level_api_tests(){
    using namespace  high_level_api_tests;

    test_parse_basic();
    test_named_flags_success();
    test_named_flags_failure();
    test_named_flags_failure_and_other_arg();
    test_named_flags_and_other_arg();
    test_flags_only_others_default();
    test_named_flags_and_other_arg_different_order();

    usage_string::test_one_arg();
    usage_string::test_named_and_postional_args();
    usage_string::test_bool_flag();

    parameters_validation_test_suite::test_validation_function_called();
    parameters_validation_test_suite::test_validation_failed();

    name_value_syntax::test_basic();
    name_value_syntax::test_relaxed_order();

    number_of_argument_value::test_only_single_values_allowed();

    name_positional_syntax::test_basic();
    name_positional_syntax::test_positional_args_explicitly_named();

    error_handling::test_wrong_input();
    error_handling::test_duplicate_arg_names();
    error_handling::test_duplicate_positional_arg_names();

    thread_range_tests::constructor_default_values();
    thread_range_tests::validation();
    thread_range_tests::post_process_single_value();
    thread_range_tests::post_process_pair_value();
    thread_range_tests::post_process_troika_value_with_plus_step();
    thread_range_tests::post_process_troika_value_with_multiply_step();
    thread_range_tests::post_process_troika_value_with_ladder_step();
    thread_range_tests::test_print_content();
}
#endif // TBB_USE_EXCEPTIONS

#include "harness.h"
int TestMain(){
#if TBB_USE_EXCEPTIONS
    Harness::suppress_unused_warning(utility::thread_number_range_desc);
    try{
        run_implementation_unit_tests();
        run_high_level_api_tests();
    }catch(std::exception& e){
        //something went wrong , dump any possible details
        std::stringstream str; str<< "run time error: " << e.what()<<std::endl;
        ASSERT(false,str.str().c_str());
    }
    return Harness::Done;
#else
    REPORT("Known issue: the test cannot work with exceptions disabled\n");
    return Harness::Done;
#endif
}
