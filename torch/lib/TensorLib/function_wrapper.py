from code_template import CodeTemplate
import yaml
# what has to be done to add a Operation ...
# 1. add virtual dispatch declaration to Type.h
# 2. add virtual override to TypeDerived.h
# 3. add override definition to TypeDerived.cpp
# 4. add non-virtual declaration to Type.h
# 5. add non-virtual declaration to Type.cpp

class NYIError(Exception):
    """Indicates we don't support this declaration yet"""
    def __init__(self,reason):
        self.reason = reason

def create_generic(top_env, declarations):
    def process_option(option):
        if option['cname'] != 'neg':
            raise NYIError("all not implemented")
        print(yaml.dump(option))

    for declaration in declarations:
        for option in declaration['options']:
            try:
                process_option(option)
            except NYIError as e:
                option['skip'] = True

def create_derived(env,declarations):
    declarations = []
    definitions = []
    def process_option(option):
        pass

    for declaration in declarations:
        for option in declaration['options']:
            if option.get('skip',False):
                try:
                    process_option(option)
                except NYIError as e:
                    pass
    return declarations,definitions
