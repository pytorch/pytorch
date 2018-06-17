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

#ifndef UTILITY_H_
#define UTILITY_H_

#if __TBB_MIC_OFFLOAD
#pragma offload_attribute (push,target(mic))
#include <exception>
#include <cstdio>
#pragma offload_attribute (pop)
#endif // __TBB_MIC_OFFLOAD

#include <utility>
#include <string>
#include <cstring>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <sstream>
#include <numeric>
#include <stdexcept>
#include <memory>
#include <cassert>
#include <iostream>
#include <cstdlib>
// TBB headers should not be used, as some examples may need to be built without TBB.

namespace utility{
    namespace internal{

#if (_MSC_VER >= 1600 || __cplusplus >= 201103L || __GXX_EXPERIMENTAL_CXX0X__) \
    && (_CPPLIB_VER || _LIBCPP_VERSION || __GLIBCXX__ && _UNIQUE_PTR_H ) \
    && (!__INTEL_COMPILER || __INTEL_COMPILER >= 1200 )
    // std::unique_ptr is available, and compiler can use it
    #define smart_ptr std::unique_ptr
    using std::swap;
#else
    #if __INTEL_COMPILER && __GXX_EXPERIMENTAL_CXX0X__
    // std::unique_ptr is unavailable, so suppress std::auto_prt<> deprecation warning
    #pragma warning(disable: 1478)
    #endif
    #define smart_ptr std::auto_ptr
    // in some C++ libraries, std::swap does not work with std::auto_ptr
    template<typename T>
    void swap( std::auto_ptr<T>& ptr1, std::auto_ptr<T>& ptr2 ) {
        std::auto_ptr<T> tmp; tmp = ptr2; ptr2 = ptr1; ptr1 = tmp;
    }
#endif

        //TODO: add tcs
        template<class dest_type>
        dest_type& string_to(std::string const& s, dest_type& result){
            std::stringstream stream(s);
            stream>>result;
            if ((!stream)||(stream.fail())){
                throw std::invalid_argument("error converting string '"+std::string(s)+"'");
            }
            return result;
        }

        template<class dest_type>
        dest_type string_to(std::string const& s){
            dest_type result;
            return string_to(s,result);
        }

        template<typename>
        struct is_bool          { static bool value(){return false;}};
        template<>
        struct is_bool<bool>    { static bool value(){return true;}};

        class type_base {
            type_base& operator=(const type_base&);
            public:
            const std::string name;
            const std::string description;

            type_base (std::string a_name, std::string a_description) : name(a_name), description(a_description) {}
            virtual void parse_and_store(const std::string & s) = 0;
            virtual std::string value() const = 0;
            virtual smart_ptr<type_base> clone() const = 0;
            virtual ~type_base(){}
        };
        template <typename type>
        class type_impl : public type_base {
        private:
            type_impl& operator=(const type_impl&);
            typedef bool(*validating_function_type)(const type&);
        private:
            type & target;
            validating_function_type validating_function;
        public:
            type_impl(std::string a_name, std::string a_description, type & a_target, validating_function_type a_validating_function = NULL)
                : type_base (a_name,a_description), target(a_target),validating_function(a_validating_function)
            {};
            void parse_and_store (const std::string & s) /*override*/ {
                try{
                    const bool is_bool = internal::is_bool<type>::value();
                    if (is_bool && s.empty()){
                        //to avoid directly assigning true
                        //(as it will impose additional layer of indirection)
                        //so, simply pass it as string
                        internal::string_to("1",target);
                    }else {
                        internal::string_to(s,target);
                    }
                }catch(std::invalid_argument& e){
                    std::stringstream str;
                    str <<"'"<<s<<"' is incorrect input for argument '"<<name<<"'"
                        <<" ("<<e.what()<<")";
                    throw std::invalid_argument(str.str());
                }
                if (validating_function){
                    if (!((validating_function)(target))){
                        std::stringstream str;
                        str <<"'"<<target<<"' is invalid value for argument '"<<name<<"'";
                        throw std::invalid_argument(str.str());
                    }
                }
            }
            template <typename t>
            static bool is_null_c_str(t&){return false;}
            static bool is_null_c_str(char* s){return s==NULL;}
            std::string value() const /*override*/ {
                std::stringstream str;
                if (!is_null_c_str(target))
                    str<<target;
                return str.str();
            }
            smart_ptr<type_base> clone() const /*override*/ {
                return smart_ptr<type_base>(new type_impl(*this));
            }
        };

        class argument{
        private:
            smart_ptr<type_base> p_type;
            bool matched_;
        public:
            argument(argument const& other)
                : p_type(other.p_type.get() ? (other.p_type->clone()).release() : NULL)
                 ,matched_(other.matched_)
            {}
            argument& operator=(argument a){
                this->swap(a);
                return *this;
            }
            void swap(argument& other){
                internal::swap(p_type, other.p_type);
                std::swap(matched_,other.matched_);
            }
            template<class type>
            argument(std::string a_name, std::string a_description, type& dest, bool(*a_validating_function)(const type&)= NULL)
                :p_type(new type_impl<type>(a_name,a_description,dest,a_validating_function))
                 ,matched_(false)
            {}
            std::string value()const{
                return p_type->value();
            }
            std::string name()const{
                return p_type->name;
            }
            std::string description() const{
                return p_type->description;
            }
            void parse_and_store(const std::string & s){
                p_type->parse_and_store(s);
                matched_=true;
            }
            bool is_matched() const{return matched_;}
        };
    } // namespace internal

    class cli_argument_pack{
        typedef std::map<std::string,internal::argument> args_map_type;
        typedef std::vector<std::string> args_display_order_type;
        typedef std::vector<std::string> positional_arg_names_type;
    private:
        args_map_type args_map;
        args_display_order_type args_display_order;
        positional_arg_names_type positional_arg_names;
        std::set<std::string> bool_args_names;
    private:
        void add_arg(internal::argument const& a){
            std::pair<args_map_type::iterator, bool> result = args_map.insert(std::make_pair(a.name(),a));
            if (!result.second){
                throw std::invalid_argument("argument with name: '"+a.name()+"' already registered");
            }
            args_display_order.push_back(a.name());
        }
    public:
        template<typename type>
        cli_argument_pack& arg(type& dest,std::string const& name, std::string const& description, bool(*validate)(const type &)= NULL){
            internal::argument a(name,description,dest,validate);
            add_arg(a);
            if (internal::is_bool<type>::value()){
                bool_args_names.insert(name);
            }
            return *this;
        }

        //Positional means that argument name can be omitted in actual CL
        //only key to match values for parameters with
        template<typename type>
        cli_argument_pack& positional_arg(type& dest,std::string const& name, std::string const& description, bool(*validate)(const type &)= NULL){
            internal::argument a(name,description,dest,validate);
            add_arg(a);
            if (internal::is_bool<type>::value()){
                bool_args_names.insert(name);
            }
            positional_arg_names.push_back(name);
            return *this;
        }

        void parse(std::size_t argc, char const* argv[]){
            {
                std::size_t current_positional_index=0;
                for (std::size_t j=1;j<argc;j++){
                    internal::argument* pa = NULL;
                    std::string argument_value;

                    const char * const begin=argv[j];
                    const char * const end=begin+std::strlen(argv[j]);

                    const char * const assign_sign = std::find(begin,end,'=');

                    struct throw_unknown_parameter{ static void _(std::string const& location){
                        throw std::invalid_argument(std::string("unknown parameter starting at:'")+location+"'");
                    }};
                    //first try to interpret it like parameter=value string
                    if (assign_sign!=end){
                        std::string name_found = std::string(begin,assign_sign);
                        args_map_type::iterator it = args_map.find(name_found );

                        if(it!=args_map.end()){
                            pa= &((*it).second);
                            argument_value = std::string(assign_sign+1,end);
                        }else {
                            throw_unknown_parameter::_(argv[j]);
                        }
                    }
                    //then see is it a named flag
                    else{
                        args_map_type::iterator it = args_map.find(argv[j] );
                        if(it!=args_map.end()){
                            pa= &((*it).second);
                            argument_value = "";
                        }
                        //then try it as positional argument without name specified
                        else if (current_positional_index < positional_arg_names.size()){
                            std::stringstream str(argv[j]);
                            args_map_type::iterator found_positional_arg = args_map.find(positional_arg_names.at(current_positional_index));
                            //TODO: probably use of smarter assert would help here
                            assert(found_positional_arg!=args_map.end()/*&&"positional_arg_names and args_map are out of sync"*/);
                            if (found_positional_arg==args_map.end()){
                                throw std::logic_error("positional_arg_names and args_map are out of sync");
                            }
                            pa= &((*found_positional_arg).second);
                            argument_value = argv[j];

                            current_positional_index++;
                        }else {
                            //TODO: add tc to check
                            throw_unknown_parameter::_(argv[j]);
                        }
                    }
                    assert(pa);
                    if (pa->is_matched()){
                        throw std::invalid_argument(std::string("several values specified for: '")+pa->name()+"' argument");
                    }
                    pa->parse_and_store(argument_value);
                }
            }
        }
        std::string usage_string(const std::string& binary_name)const{
            std::string command_line_params;
            std::string summary_description;

            for (args_display_order_type::const_iterator it = args_display_order.begin();it!=args_display_order.end();++it){
                const bool is_bool = (0!=bool_args_names.count((*it)));
                args_map_type::const_iterator argument_it = args_map.find(*it);
                //TODO: probably use of smarter assert would help here
                assert(argument_it!=args_map.end()/*&&"args_display_order and args_map are out of sync"*/);
                if (argument_it==args_map.end()){
                    throw std::logic_error("args_display_order and args_map are out of sync");
                }
                const internal::argument & a = (*argument_it).second;
                command_line_params +=" [" + a.name() + (is_bool ?"":"=value")+ "]";
                summary_description +=" " + a.name() + " - " + a.description() +" ("+a.value() +")" + "\n";
            }

            std::string positional_arg_cl;
            for (positional_arg_names_type::const_iterator it = positional_arg_names.begin();it!=positional_arg_names.end();++it){
                positional_arg_cl +=" ["+(*it);
            }
            for (std::size_t i=0;i<positional_arg_names.size();++i){
                positional_arg_cl+="]";
            }
            command_line_params+=positional_arg_cl;
            std::stringstream str;
            using std::endl;
            str << " Program usage is:" << endl
                 << " " << binary_name << command_line_params
                 << endl << endl
                 << " where:" << endl
                 << summary_description
            ;
            return str.str();
        }
    }; // class cli_argument_pack

    namespace internal {
        template<typename T>
        bool is_power_of_2( T val ) {
            size_t intval = size_t(val);
            return (intval&(intval-1)) == size_t(0);
        }
        int step_function_plus(int previous, double step){
            return static_cast<int>(previous+step);
        }
        int step_function_multiply(int previous, double multiply){
            return static_cast<int>(previous*multiply);
        }
        // "Power-of-2 ladder": nsteps is the desired number of steps between any subsequent powers of 2.
        // The actual step is the quotient of the nearest smaller power of 2 divided by that number (but at least 1).
        // E.g., '1:32:#4' means 1,2,3,4,5,6,7,8,10,12,14,16,20,24,28,32
        int step_function_power2_ladder(int previous, double nsteps){
            int steps = int(nsteps);
            assert( is_power_of_2(steps) );  // must be a power of 2
            // The actual step is 1 until the value is twice as big as nsteps
            if( previous < 2*steps )
                return previous+1;
            // calculate the previous power of 2
            int prev_power2 = previous/2;                 // start with half the given value
            int rshift = 1;                               // and with the shift of 1;
            while( int shifted = prev_power2>>rshift ) {  // shift the value right; while the result is non-zero,
                prev_power2 |= shifted;                   //   add the bits set in 'shifted';
                rshift <<= 1;                             //   double the shift, as twice as many top bits are set;
            }                                             // repeat.
            ++prev_power2; // all low bits set; now it's just one less than the desired power of 2
            assert( is_power_of_2(prev_power2) );
            assert( (prev_power2<=previous)&&(2*prev_power2>previous) );
            // The actual step value is the previous power of 2 divided by steps
            return previous + (prev_power2/steps);
        }
        typedef int (* step_function_ptr_type)(int,double);

        struct step_function_descriptor  {
            char mnemonic;
            step_function_ptr_type function;
        public:
            step_function_descriptor(char a_mnemonic, step_function_ptr_type a_function) : mnemonic(a_mnemonic), function(a_function) {}
        private:
            void operator=(step_function_descriptor  const&);
        };
        step_function_descriptor step_function_descriptors[] = {
                step_function_descriptor('*',step_function_multiply),
                step_function_descriptor('+',step_function_plus),
                step_function_descriptor('#',step_function_power2_ladder)
        };

        template<typename T, size_t N>
        inline size_t array_length(const T(&)[N])
        {
           return N;
        }

        struct thread_range_step {
            step_function_ptr_type step_function;
            double step_function_argument;

            thread_range_step ( step_function_ptr_type step_function_, double step_function_argument_)
                :step_function(step_function_),step_function_argument(step_function_argument_)
            {
                if (!step_function_)
                    throw std::invalid_argument("step_function for thread range step should not be NULL");
            }
            int operator()(int previous)const {
                assert(0<=previous); // test 0<=first and loop discipline
                const int ret = step_function(previous,step_function_argument);
                assert(previous<ret);
                return ret;
            }
            friend std::istream& operator>>(std::istream& input_stream, thread_range_step& step){
                char function_char;
                double function_argument;
                input_stream >> function_char >> function_argument;
                size_t i = 0;
                while ((i<array_length(step_function_descriptors)) && (step_function_descriptors[i].mnemonic!=function_char)) ++i;
                if (i >= array_length(step_function_descriptors)){
                    throw std::invalid_argument("unknown step function mnemonic: "+std::string(1,function_char));
                } else if ((function_char=='#') && !is_power_of_2(function_argument)) {
                    throw std::invalid_argument("the argument of # should be a power of 2");
                }
                step.step_function = step_function_descriptors[i].function;
                step.step_function_argument = function_argument;
                return input_stream;
            }
        };
    } // namespace internal

    struct thread_number_range{
        int (*auto_number_of_threads)();
        int first; // 0<=first (0 can be used as a special value)
        int last;  // first<=last

        internal::thread_range_step step;

        thread_number_range( int (*auto_number_of_threads_)(),int low_=1, int high_=-1
                , internal::thread_range_step step_ =  internal::thread_range_step(internal::step_function_power2_ladder,4)
        )
            : auto_number_of_threads(auto_number_of_threads_), first(low_), last((high_>-1) ? high_ : auto_number_of_threads_())
              ,step(step_)
        {
            if (first<0) {
                throw std::invalid_argument("negative value not allowed");
            }
            if (first>last) {
                throw std::invalid_argument("decreasing sequence not allowed");
            }
        }
        friend std::istream& operator>>(std::istream& i, thread_number_range& range){
            try{
                std::string s;
                i>>s;
                struct string_to_number_of_threads{
                    int auto_value;
                    string_to_number_of_threads(int auto_value_):auto_value(auto_value_){}
                    int operator()(const std::string & value)const{
                        return (value=="auto")? auto_value : internal::string_to<int>(value);
                    }
                };
                string_to_number_of_threads string_to_number_of_threads(range.auto_number_of_threads());
                int low, high;
                std::size_t colon = s.find(':');
                if ( colon == std::string::npos ){
                    low = high = string_to_number_of_threads(s);
                } else {
                    //it is a range
                    std::size_t second_colon = s.find(':',colon+1);

                    low  = string_to_number_of_threads(std::string(s, 0, colon)); //not copying the colon
                    high = string_to_number_of_threads(std::string(s, colon+1, second_colon - (colon+1))); //not copying the colons
                    if (second_colon != std::string::npos){
                        internal::string_to(std::string(s,second_colon + 1),range.step);
                    }
                }
                range = thread_number_range(range.auto_number_of_threads,low,high,range.step);
            }catch(std::invalid_argument&){
                i.setstate(std::ios::failbit);
                throw;
            }
            return i;
        }
        friend std::ostream& operator<<(std::ostream& o, thread_number_range const& range){
            using namespace internal;
            size_t i = 0;
            for (; i < array_length(step_function_descriptors) && step_function_descriptors[i].function != range.step.step_function; ++i ) {}
            if (i >= array_length(step_function_descriptors)){
                throw std::invalid_argument("unknown step function for thread range");
            }
            o<<range.first<<":"<<range.last<<":"<<step_function_descriptors[i].mnemonic<<range.step.step_function_argument;
            return o;
        }
    }; // struct thread_number_range
    //TODO: fix unused warning here
    //TODO: update the thread range description in the .html files
    static const char* thread_number_range_desc="number of threads to use; a range of the form low[:high[:(+|*|#)step]],"
                                                "\n\twhere low and optional high are non-negative integers or 'auto' for the default choice,"
                                                "\n\tand optional step expression specifies how thread numbers are chosen within the range."
                                                "\n\tSee examples/common/index.html for detailed description."
   ;

    inline void report_elapsed_time(double seconds){
        std::cout<<"elapsed time : "<<seconds<<" seconds"<<std::endl;
    }

    inline void report_skipped(){
        std::cout<<"skip"<<std::endl;
    }

    inline void parse_cli_arguments(int argc, const char* argv[], utility::cli_argument_pack cli_pack){
        bool show_help = false;
        cli_pack.arg(show_help,"-h","show this message");

        bool invalid_input=false;
        try {
            cli_pack.parse(argc,argv);
        }catch(std::exception& e){
            std::cerr
                    <<"error occurred while parsing command line."<<std::endl
                    <<"error text: "<<e.what()<<std::endl
                    <<std::flush;
            invalid_input =true;
        }
        if (show_help || invalid_input){
            std::cout<<cli_pack.usage_string(argv[0])<<std::flush;
            std::exit(0);
        }

    }
    inline void parse_cli_arguments(int argc, char* argv[], utility::cli_argument_pack cli_pack){
         parse_cli_arguments(argc, const_cast<const char**>(argv), cli_pack);
    }
}

#endif /* UTILITY_H_ */
