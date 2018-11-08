# Copyright 2014 Google Inc. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from getopt import getopt
import os
import re
#import types

TERM_BOLD_START = "\033[1m"
TERM_BOLD_END = "\033[0m"

class Option:
    def __init__(self, letter, name, desc, parser, set_once, default, excuses, requires, save):
        assert not name is None
        self.letter = letter
        self.name = name
        self.desc = desc
        self.parser = parser
        self.set_once = set_once
        self.default = default
        self.excuses = excuses
        self.requires = requires
        self.save = save
        
        self.value = None
        self.value_given = False
        self.prefixed_letter = min(2, len(letter)) * '-' + letter
        
    def set_value(self, value, parse=True):
        try:
            self.value = self.parser.parse(value) if parse else value
            self.value_given = True
#            print self.name, self.value
        except OptionException, e:
            raise OptionException("Unable to parse option %s (%s): %s" % (self.prefixed_letter, self.desc, e))
        
    def set_default(self):
        if not self.default is None:
            self.value = self.default
    
    def eval_expr_default(self, env):
        try:
            if isinstance(self.default, OptionExpression) and not self.value_given:
                self.value = self.default.evaluate(env)
                if not self.parser.is_type(self.value):
                    raise OptionException("expression result %s is not of right type (%s)" % (self.value, self.parser.get_type_str()))
        except Exception, e:
            raise OptionException("Unable to set default value for option %s (%s): %s" % (self.prefixed_letter, self.desc, e))
            
    def get_str_value(self, get_default_str=False):
        val = self.value
        if get_default_str: val = self.default
        if val is None: return ""
        if isinstance(val, OptionExpression):
            return val.expr
        return self.parser.to_string(val)

class OptionsParser:
    """An option parsing class. All options without default values are mandatory, unless a excuses
    option (usually a load file) is given.
    Does not support options without arguments."""
    SORT_LETTER = 1
    SORT_DESC = 2
    SORT_EXPR_LAST = 3
    EXCUSE_ALL = "all"
    def __init__(self):
        self.options = {}
        
    def add_option(self, letter, name, parser, desc, set_once=False, default=None, excuses=[], requires=[], save=True):
        """
        The letter parameter is the actual parameter that the user will have to supply on the command line.
        The name parameter is some name to be given to this option and must be a valid python variable name.
        
        An explanation of the "default" parameter:
        The default value, if specified, should have the same type as the option.
        You can also specify an expression as the default value. In this case, the default value of the parameter
        will be the output of the expression. The expression may assume all other option names
        as local variables. For example, you can define the hidden bias
        learning rate to be 10 times the weight learning rate by setting this default:
        
        default=OptionExpression("eps_w * 10") (assuming an option named eps_w exists).
        
        However, it is up to you to make sure you do not make any circular expression definitions.
        
        Note that the order in which the options are parsed is arbitrary.
        In particular, expression default values that depend on other expression default values
        will often raise errors (depending on the order in which they happen to be parsed).
        Therefore it is best not to make the default value of one variable depend on the value
        of another if the other variable's default value is itself an expression.
        
        An explanation of the "excuses" parameter:
        All options are mandatory, but certain options can exclude other options from being mandatory.
        For example, if the excuses parameter for option "load_file" is ["num_hid", "num_vis"],
        then the options num_hid and num_vis are not mandatory as long as load_file is specified.
        Use the special flag EXCUSE_ALL to allow an option to make all other options optional.
        """
        
        assert name not in self.options
        self.options[name] = Option(letter, name, desc, parser, set_once, default, excuses, requires, save)
    
    def set_value(self, name, value, parse=True):
        self.options[name].set_value(value, parse=parse)
    
    def get_value(self, name):
        return self.options[name].value
        
    def delete_option(self, name):
        if name in self.options:
            del self.options[name]
            
    def parse(self, eval_expr_defaults=False):
        """Parses the options in sys.argv based on the options added to this parser. The
        default behavior is to leave any expression default options as OptionExpression objects.
        Set eval_expr_defaults=True to circumvent this."""
        short_opt_str = ''.join(["%s:" % self.options[name].letter for name in self.options if len(self.options[name].letter) == 1])
        long_opts = ["%s=" % self.options[name].letter for name in self.options if len(self.options[name].letter) > 1]
        (go, ga) = getopt(sys.argv[1:], short_opt_str, longopts=long_opts)
        dic = dict(go)
        
        for o in self.get_options_list(sort_order=self.SORT_EXPR_LAST):
            if o.prefixed_letter in dic:  
                o.set_value(dic[o.prefixed_letter])
            else:
                # check if excused or has default
                excused = max([o2.prefixed_letter in dic for o2 in self.options.values() if o2.excuses == self.EXCUSE_ALL or o.name in o2.excuses])
                if not excused and o.default is None:
                    raise OptionMissingException("Option %s (%s) not supplied" % (o.prefixed_letter, o.desc))
                o.set_default()
            # check requirements
            if o.prefixed_letter in dic:
                for o2 in self.get_options_list(sort_order=self.SORT_LETTER):
                    if o2.name in o.requires and o2.prefixed_letter not in dic:
                        raise OptionMissingException("Option %s (%s) requires option %s (%s)" % (o.prefixed_letter, o.desc,
                                                                                                 o2.prefixed_letter, o2.desc))
        if eval_expr_defaults:
            self.eval_expr_defaults()
        return self.options
    
    def merge_from(self, op2):
        """Merges the options in op2 into this instance, but does not overwrite
        this instances's SET options with op2's default values."""
        for name, o in self.options.iteritems():
            if name in op2.options and ((op2.options[name].value_given and op2.options[name].value != self.options[name].value) or not op2.options[name].save):
                if op2.options[name].set_once:
                    raise OptionException("Option %s (%s) cannot be changed" % (op2.options[name].prefixed_letter, op2.options[name].desc))
                self.options[name] = op2.options[name]
        for name in op2.options:
            if name not in self.options:
                self.options[name] = op2.options[name]
    
    def eval_expr_defaults(self):
        env = dict([(name, o.value) for name, o in self.options.iteritems()])
        for o in self.options.values():
            o.eval_expr_default(env)
            
    def all_values_given(self):
        return max([o.value_given for o in self.options.values() if o.default is not None])
    
    def get_options_list(self, sort_order=SORT_LETTER):
        """ Returns the list of Option objects in this OptionParser,
        sorted as specified"""
        
        cmp = lambda x, y: (x.desc < y.desc and -1 or 1)
        if sort_order == self.SORT_LETTER:
            cmp = lambda x, y: (x.letter < y.letter and -1 or 1)
        elif sort_order == self.SORT_EXPR_LAST:
            cmp = lambda x, y: (type(x.default) == OptionExpression and 1 or -1)
        return sorted(self.options.values(), cmp=cmp)
    
    def print_usage(self, print_constraints=False):
        print "%s usage:" % os.path.basename(sys.argv[0])
        opslist = self.get_options_list()

        usage_strings = []
        num_def = 0
        for o in opslist:
            excs = ' '
            if o.default is None:
                excs = ', '.join(sorted([o2.prefixed_letter for o2 in self.options.values() if o2.excuses == self.EXCUSE_ALL or o.name in o2.excuses]))
            reqs = ', '.join(sorted([o2.prefixed_letter for o2 in self.options.values() if o2.name in o.requires]))
            usg = (OptionsParser._bold(o.prefixed_letter) + " <%s>" % o.parser.get_type_str(), o.desc, ("[%s]" % o.get_str_value(get_default_str=True)) if not o.default is None else None, excs, reqs)
            if o.default is None:
                usage_strings += [usg]
            else:
                usage_strings.insert(num_def, usg)
                num_def += 1
                
        col_widths = [self._longest_value(usage_strings, key=lambda x:x[i]) for i in range(len(usage_strings[0]) - 1)]

        col_names = ["    Option", "Description", "Default"]
        if print_constraints:
            col_names += ["Excused by", "Requires"]
        for i, s in enumerate(col_names):
            print self._bold(s.ljust(col_widths[i])),

        print ""
        for l, d, de, ex, req in usage_strings:
            if de is None:
                de = ' '
                print ("     %s  -" % l.ljust(col_widths[0])), d.ljust(col_widths[1]), de.ljust(col_widths[2]),
            else:
                print ("    [%s] -" % l.ljust(col_widths[0])), d.ljust(col_widths[1]), de.ljust(col_widths[2]),
            if print_constraints:
                print ex.ljust(col_widths[3]), req
            else:
                print ""
                
    def print_values(self):
        longest_desc = self._longest_value(self.options.values(), key=lambda x:x.desc)
        longest_def_value = self._longest_value([v for v in self.options.values() if not v.value_given and not v.default is None],
                                                 key=lambda x:x.get_str_value())
        for o in self.get_options_list(sort_order=self.SORT_DESC):
            print "%s: %s %s" % (o.desc.ljust(longest_desc), o.get_str_value().ljust(longest_def_value), (not o.value_given and not o.default is None) and "[DEFAULT]" or "")
    
    @staticmethod
    def _longest_value(values, key=lambda x:x):
        mylen = lambda x: 0 if x is None else len(x)
        return mylen(key(max(values, key=lambda x:mylen(key(x)))))

    @staticmethod
    def _bold(str):
        return TERM_BOLD_START + str + TERM_BOLD_END

class OptionException(Exception):
    pass
                
class OptionMissingException(OptionException):
    pass

class OptionParser:
    @staticmethod
    def parse(value):
        return str(value)
       
    @staticmethod
    def to_string(value):
        return str(value)
    
    @staticmethod
    def get_type_str():
        pass
    
class IntegerOptionParser(OptionParser):
    @staticmethod
    def parse(value):
        try:
            return int(value)
        except:
            raise OptionException("argument is not an integer")
    
    @staticmethod
    def get_type_str():
        return "int"
    
    @staticmethod
    def is_type(value):
        return type(value) == int
    
class BooleanOptionParser(OptionParser):
    @staticmethod
    def parse(value):
        try:
            v = int(value)
            if not v in (0,1):
                raise OptionException
            return v
        except:
            raise OptionException("argument is not a boolean")
    
    @staticmethod
    def get_type_str():
        return "0/1"
    
    @staticmethod
    def is_type(value):
        return type(value) == int and value in (0, 1)
        
class StringOptionParser(OptionParser):       
    @staticmethod
    def get_type_str():
        return "string"
    
    @staticmethod
    def is_type(value):
        return type(value) == str
    
class FloatOptionParser(OptionParser):
    @staticmethod
    def parse(value):
        try:
            return float(value)
        except:
            raise OptionException("argument is not a float")
    
    @staticmethod
    def to_string(value):
        return "%.6g" % value
    
    @staticmethod
    def get_type_str():
        return "float"
    
    @staticmethod
    def is_type(value):
        return type(value) == float
    
class RangeOptionParser(OptionParser):
    @staticmethod
    def parse(value):
        m = re.match("^(\d+)\-(\d+)$", value)
        try:
            if m: return range(int(m.group(1)), int(m.group(2)) + 1)
            return [int(value)]
        except:
            raise OptionException("argument is neither an integer nor a range")
    
    @staticmethod
    def to_string(value):
        return "%d-%d" % (value[0], value[-1])
    
    @staticmethod
    def get_type_str():
        return "int[-int]"
    
    @staticmethod
    def is_type(value):
        return type(value) == list
    
class ListOptionParser(OptionParser):
    """
    A parser that parses a delimited list of items. If the "parsers"
    argument is a list of parsers, then the list of items must have the form and length
    specified by that list. 
    
    Example:
    ListOptionParser([FloatOptionParser, IntegerOptionParser])
    
    would parse "0.5,3" but not "0.5,3,0.6" or "0.5" or "3,0.5".
    
    If the "parsers" argument is another parser, then the list of items may be of
    arbitrary length, but each item must be parseable by the given parser.
    
    Example:
    ListOptionParser(FloatOptionParser)
    
    would parse "0.5" and "0.5,0.3" and "0.5,0.3,0.6", etc.
    """
    def __init__(self, parsers, sepchar=','):
        self.parsers = parsers
        self.sepchar = sepchar
        
    def parse(self, value):
        values = value.split(self.sepchar)
        if type(self.parsers) == list and len(values) != len(self.parsers):
            raise OptionException("requires %d arguments, given %d" % (len(self.parsers), len(values)))
        
        try:
            if type(self.parsers) == list:
                return [p.parse(v) for p, v in zip(self.parsers, values)]
            return [self.parsers.parse(v) for v in values]
        except:
            raise OptionException("argument is not of the form %s" % self.get_type_str())
    
    def to_string(self, value):
        if type(self.parsers) == list:
            return self.sepchar.join([p.to_string(v) for p, v in zip(self.parsers, value)])
        return self.sepchar.join([self.parsers.to_string(v) for v in value])
    
    def get_type_str(self):
        if type(self.parsers) == list:
            return self.sepchar.join([p.get_type_str() for p in self.parsers])
        return "%s%s..." % (self.parsers.get_type_str(), self.sepchar)
    
    @staticmethod
    def is_type(value):
        return type(value) == list
    
class OptionExpression:
    """
    This allows you to specify option values in terms of other option values.
    Example:
    op.add_option("eps-w", "eps_w", ListOptionParser(FloatOptionParser), "Weight learning rates for each layer")
    op.add_option("eps-b", "eps_b", ListOptionParser(FloatOptionParser), "Bias learning rates for each layer", default=OptionExpression("[o * 10 for o in eps_w]"))
    
    This says: the default bias learning rate for each layer is 10
    times the weight learning rate for that layer.
    """
    def __init__(self, expr):
        self.expr = expr
    
    def evaluate(self, options):
        locals().update(options)
        try:
            return eval(self.expr)
        except Exception, e:
            raise OptionException("expression '%s': unable to parse: %s" % (self.expr, e))
