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

from math import exp
import sys
import ConfigParser as cfg
import os
import numpy as n
import numpy.random as nr
from math import ceil, floor
from collections import OrderedDict
from os import linesep as NL
from python_util.options import OptionsParser
import re

class LayerParsingError(Exception):
    pass

# A neuron that doesn't take parameters
class NeuronParser:
    def __init__(self, type, func_str, uses_acts=True, uses_inputs=True):
        self.type = type
        self.func_str = func_str
        self.uses_acts = uses_acts  
        self.uses_inputs = uses_inputs
        
    def parse(self, type):
        if type == self.type:
            return {'type': self.type,
                    'params': {},
                    'usesActs': self.uses_acts,
                    'usesInputs': self.uses_inputs}
        return None
    
# A neuron that takes parameters
class ParamNeuronParser(NeuronParser):
    neuron_regex = re.compile(r'^\s*(\w+)\s*\[\s*(\w+(\s*,\w+)*)\s*\]\s*$')
    def __init__(self, type, func_str, uses_acts=True, uses_inputs=True):
        NeuronParser.__init__(self, type, func_str, uses_acts, uses_inputs)
        m = self.neuron_regex.match(type)
        self.base_type = m.group(1)
        self.param_names = m.group(2).split(',')
        assert len(set(self.param_names)) == len(self.param_names)
        
    def parse(self, type):
        m = re.match(r'^%s\s*\[([\d,\.\s\-]*)\]\s*$' % self.base_type, type)
        if m:
            try:
                param_vals = [float(v.strip()) for v in m.group(1).split(',')]
                if len(param_vals) == len(self.param_names):
                    return {'type': self.base_type,
                            'params': dict(zip(self.param_names, param_vals)),
                            'usesActs': self.uses_acts,
                            'usesInputs': self.uses_inputs}
            except TypeError:
                pass
        return None

class AbsTanhNeuronParser(ParamNeuronParser):
    def __init__(self):
        ParamNeuronParser.__init__(self, 'abstanh[a,b]', 'f(x) = a * |tanh(b * x)|')
        
    def parse(self, type):
        dic = ParamNeuronParser.parse(self, type)
        # Make b positive, since abs(tanh(bx)) = abs(tanh(-bx)) and the C++ code
        # assumes b is positive.
        if dic:
            dic['params']['b'] = abs(dic['params']['b'])
        return dic

class ParamParser:
    lrs_regex = re.compile(r'^\s*(\w+)\s*(?:\[\s*(\w+(\s*;\w+)*)\s*\])?\s*$')
    param_converters = {'i': int,
                        'f': float}
    def __init__(self, type):
        m = self.lrs_regex.match(type)
        self.base_type = m.group(1)
        param_names_with_type = m.group(2).split(';') if m.group(2) is not None else []
        self.param_names = [p[1:] for p in param_names_with_type]
        self.param_types = [self.param_converters[p[0]] for p in param_names_with_type]
        self.param_regex_inner = ";".join([('\s*%s\s*=\s*[^;,\s=]+\s*' % p) for p in self.param_names])
        self.regex_str = ('^%s\s*(?:\[(%s)\])?\s*$') % (self.base_type, self.param_regex_inner)
        assert len(set(self.param_names)) == len(self.param_names)
    
    def parse(self, type):
        m = re.match(self.regex_str, type, flags=re.IGNORECASE)
        if m:
            try:
                param_vals = [ptype(v.split('=')[1].strip()) for ptype,v in zip(self.param_types, m.group(1).split(';'))] if m.group(1) is not None else []
                if len(param_vals) == len(self.param_names):
                    return {'type': self.base_type,
                            'params': dict(zip(self.param_names, param_vals))}
            except TypeError:
                pass
        return None

# Subclass that throws more convnet-specific exceptions than the default
class MyConfigParser(cfg.SafeConfigParser):
    def safe_get(self, section, option, f=cfg.SafeConfigParser.get, typestr=None, default=None):
        try:
            return f(self, section, option)
        except cfg.NoOptionError, e:
            if default is not None:
                return default
            raise LayerParsingError("Layer '%s': required parameter '%s' missing" % (section, option))
        except ValueError, e:
            if typestr is None:
                raise e
            raise LayerParsingError("Layer '%s': parameter '%s' must be %s" % (section, option, typestr))
        
    def safe_get_list(self, section, option, f=str, typestr='strings', default=None):
        v = self.safe_get(section, option, default=default)
        if type(v) == list:
            return v
        try:
            return [f(x.strip()) for x in v.split(',')]
        except:
            raise LayerParsingError("Layer '%s': parameter '%s' must be ','-delimited list of %s" % (section, option, typestr))
        
    def safe_get_int(self, section, option, default=None):
        return self.safe_get(section, option, f=cfg.SafeConfigParser.getint, typestr='int', default=default)
        
    def safe_get_float(self, section, option, default=None):
        return self.safe_get(section, option, f=cfg.SafeConfigParser.getfloat, typestr='float', default=default)
    
    def safe_get_bool(self, section, option, default=None):
        return self.safe_get(section, option, f=cfg.SafeConfigParser.getboolean, typestr='bool', default=default)
    
    def safe_get_float_list(self, section, option, default=None):
        return self.safe_get_list(section, option, float, typestr='floats', default=default)
    
    def safe_get_int_list(self, section, option, default=None):
        return self.safe_get_list(section, option, int, typestr='ints', default=default)
    
    def safe_get_bool_list(self, section, option, default=None):
        return self.safe_get_list(section, option, lambda x: x.lower() in ('true', '1'), typestr='bools', default=default)

# A class that implements part of the interface of MyConfigParser
class FakeConfigParser(object):
    def __init__(self, dic):
        self.dic = dic

    def safe_get(self, section, option, default=None):
        if option in self.dic:
            return self.dic[option]
        return default
    
    def safe_get_int(self, section, option, default=None):
        return int(self.safe_get(section, option, default))
    
    def safe_get_int_list(self, section, option, default=None):
        return list(self.safe_get(section, option, default))

class LayerParser:
    def __init__(self):
        self.dic = {}
        self.set_defaults()
        
    # Post-processing step -- this is called after all layers have been initialized
    def optimize(self, layers):
        self.dic['actsTarget'] = -1
        self.dic['actsGradTarget'] = -1
        if len(set(len(l['gpu']) for l in layers.values() if 'inputs' in l and self.dic['name'] in l['inputs'])) > 1:
#            print set(len(l['gpu']) for l in layers.values())
            raise LayerParsingError("Layer '%s': all next layers must have equal number of replicas." % (self.dic['name']))
    
    def parse_params(self, vals, parsers, param_name, human_name, num_params=1):
        dic, name = self.dic, self.dic['name']
        
#        print vals
        if len(vals) != num_params and len(vals) != 1:
            raise LayerParsingError("Layer '%s': expected list of length %d for %s but got list of length %d."% (name, num_params, param_name, len(vals)))
        parsed = []
#        print vals
        for v in vals:
            for p in parsers:
                parsedv = p.parse(v)
                if parsedv: 
                    parsed += [parsedv]
                    break
        if len(parsed) == 1 and num_params > 1:
            parsed = parsed * num_params
        if len(parsed) == num_params:
            return parsed
#        print parsed, vals
        raise LayerParsingError("Layer '%s': unable to parse %s %s=%s." % (name, human_name, param_name, ",".join(vals)))
    
    # Add parameters from layer parameter file
    def add_params(self, mcp):
        pass
#        self.dic['conserveMem'] = mcp.convnet.op.get_value('conserve_mem') if mcp.convnet is not None else 0
    
    def init(self, dic):
        self.dic = dic
        return self
    
    def set_defaults(self):
        self.dic['outputs'] = 0
        self.dic['parser'] = self
        self.dic['requiresParams'] = False
        # Does this layer use its own activity matrix
        # for some purpose other than computing its output?
        # Usually, this will only be true for layers that require their
        # own activity matrix for gradient computations. For example, layers
        # with logistic units must compute the gradient y * (1 - y), where y is 
        # the activity matrix.
        # 
        # Layers that do not not use their own activity matrix should advertise
        # this, since this will enable memory-saving matrix re-use optimizations.
        #
        # The default value of this property is True, for safety purposes.
        # If a layer advertises that it does not use its own activity matrix when
        # in fact it does, bad things will happen.
        self.dic['usesActs'] = True
        
        # Does this layer use the activity matrices of its input layers
        # for some purpose other than computing its output?
        #
        # Again true by default for safety
        self.dic['usesInputs'] = True
        
        # Force this layer to use its own activity gradient matrix,
        # instead of borrowing one from one of its inputs.
        # 
        # This should be true for layers where the mapping from output
        # gradient to input gradient is non-elementwise.
        self.dic['forceOwnActs'] = True
        
        # Does this layer need the gradient at all?
        # Should only be true for layers with parameters (weights).
        self.dic['gradConsumer'] = False
        
        # The gpu indices on which this layer runs
        self.dic['gpu'] = [-1]
        
    def parse(self, name, mcp, prev_layers, model=None):
        self.prev_layers = prev_layers
        self.dic['name'] = name
        self.dic['type'] = mcp.safe_get(name, 'type')
        self.dic['id'] = len(prev_layers)

        return self.dic  

    def verify_float_range(self, v, param_name, _min, _max):
        self.verify_num_range(v, param_name, _min, _max, strconv=lambda x: '%.3f' % x)

    def verify_num_range(self, v, param_name, _min, _max, strconv=lambda x:'%d' % x):
        if type(v) == list:
            for i,vv in enumerate(v):
                self._verify_num_range(vv, param_name, _min, _max, i, strconv=strconv)
        else:
            self._verify_num_range(v, param_name, _min, _max, strconv=strconv)
    
    def _verify_num_range(self, v, param_name, _min, _max, input=-1, strconv=lambda x:'%d' % x):
        layer_name = self.dic['name'] if input < 0 else '%s[%d]' % (self.dic['name'], input)
        if _min is not None and _max is not None and (v < _min or v > _max):
            raise LayerParsingError("Layer '%s': parameter '%s' must be in the range %s-%s" % (layer_name, param_name, strconv(_min), strconv(_max)))
        elif _min is not None and v < _min:
            raise LayerParsingError("Layer '%s': parameter '%s' must be greater than or equal to %s" % (layer_name, param_name,  strconv(_min)))
        elif _max is not None and v > _max:
            raise LayerParsingError("Layer '%s': parameter '%s' must be smaller than or equal to %s" % (layer_name, param_name,  strconv(_max)))
    
    def verify_divisible(self, value, div, value_name, div_name=None, input_idx=0):
        layer_name = self.dic['name'] if len(self.dic['inputs']) == 0 else '%s[%d]' % (self.dic['name'], input_idx)
        if value % div != 0:
            raise LayerParsingError("Layer '%s': parameter '%s' must be divisible by %s" % (layer_name, value_name, str(div) if div_name is None else "'%s'" % div_name))
        
    def verify_str_in(self, value, param_name, lst, input_idx=-1):
        lname = self.dic['name'] if input_idx == -1 else ('%s[%d]' % (self.dic['name'], input_idx))
        if value not in lst:
            raise LayerParsingError("Layer '%s': parameter '%s' must be one of %s" % (lname, param_name, ", ".join("'%s'" % s for s in lst)))
        
    def verify_int_in(self, value, param_name, lst):
        if value not in lst:
            raise LayerParsingError("Layer '%s': parameter '%s' must be one of %s" % (self.dic['name'], param_name, ", ".join("'%d'" % s for s in lst)))
        
    def verify_all_ints_in(self, values, param_name, lst):
        if len([v for v in values if v not in lst]) > 0:
            raise LayerParsingError("Layer '%s': all parameters to '%s' must be among %s" % (self.dic['name'], param_name, ", ".join("'%d'" % s for s in lst)))
    
    def verify_input_dims(self, dims):
        for i,d in enumerate(dims):
            if d is not None and self.dic['numInputs'][i] != d: # first input must be labels
                raise LayerParsingError("Layer '%s': dimensionality of input %d must be %d" % (self.dic['name'], i, d))

    # This looks for neuron=x arguments in various layers, and creates
    # separate layer definitions for them.
    @staticmethod
    def detach_neuron_layers(layers):
        for name,l in layers.items():
            if l['type'] != 'neuron' and 'neuron' in l and l['neuron']:
                NeuronLayerParser().detach_neuron_layer(name, layers)
                
    @staticmethod
    def parse_layers(layer_cfg_path, param_cfg_path, model, layers={}):
        try:
            if not os.path.exists(layer_cfg_path):
                raise LayerParsingError("Layer definition file '%s' does not exist" % layer_cfg_path)
            if not os.path.exists(param_cfg_path):
                raise LayerParsingError("Layer parameter file '%s' does not exist" % param_cfg_path)
            if len(layers) == 0:
                mcp = MyConfigParser(dict_type=OrderedDict)
                mcp.readfp(open(layer_cfg_path))
                for name in mcp.sections():
                    if not mcp.has_option(name, 'type'):
                        raise LayerParsingError("Layer '%s': no type given" % name)
                    ltype = mcp.safe_get(name, 'type')
                    if ltype not in layer_parsers:
                        raise LayerParsingError("Layer '%s': Unknown layer type: '%s'" % (name, ltype))
                    layers[name] = layer_parsers[ltype]().parse(name, mcp, layers, model)
                
                LayerParser.detach_neuron_layers(layers)
                for l in layers.values():
                    l['parser'].optimize(layers)
                    del l['parser']
                    
                for name,l in layers.items():
                    if not l['type'].startswith('cost.'):
                        found = max(name in l2['inputs'] for l2 in layers.values() if 'inputs' in l2)
                        if not found:
                            raise LayerParsingError("Layer '%s' of type '%s' is unused" % (name, l['type']))
            
            mcp = MyConfigParser(dict_type=OrderedDict)
            mcp.readfp(open(param_cfg_path))
#            mcp.convnet = model
            for name,l in layers.items():
                if not mcp.has_section(name) and l['requiresParams']:
                    raise LayerParsingError("Layer '%s' of type '%s' requires extra parameters, but none given in file '%s'." % (name, l['type'], param_cfg_path))
                lp = layer_parsers[l['type']]().init(l)
                lp.add_params(mcp)
        except LayerParsingError, e:
            print e
            sys.exit(1)
        return layers
        
    @staticmethod
    def register_layer_parser(ltype, cls):
        if ltype in layer_parsers:
            raise LayerParsingError("Layer type '%s' already registered" % ltype)
        layer_parsers[ltype] = cls

# Any layer that takes an input (i.e. non-data layer)
class LayerWithInputParser(LayerParser):
    def __init__(self, num_inputs=-1):
        LayerParser.__init__(self)
        self.num_inputs = num_inputs
        
    def verify_num_params(self, params, auto_expand=True):
        for param in params:
            if len(self.dic[param]) != len(self.dic['inputs']):
                if auto_expand and len(self.dic[param]) == 1:
                    self.dic[param] *= len(self.dic['inputs'])
                else:
                    raise LayerParsingError("Layer '%s': %s list length does not match number of inputs" % (self.dic['name'], param))        
    
    # layers: dictionary: name -> layer
    def optimize(self, layers):
        LayerParser.optimize(self, layers)
        dic = self.dic
        
        # Check if I have an input that no one else uses.
        #print "Layer %s optimizing" % dic['name']
        if not dic['forceOwnActs']:
            for i, inp in enumerate(dic['inputLayers']):
                if inp['outputs'] == dic['outputs'] and sum(('inputs' in ll) and (inp['name'] in ll['inputs']) for ll in layers.itervalues()) == 1:
                    # I can share my activity matrix with this layer
                    # if it does not use its activity matrix, and I 
                    # do not need to remember my inputs.
                    # TODO: a dropout layer should always be able to overwrite
                    # its input. Make it so.
#                    print "Layer %s(uses inputs=%d), input %s(uses acts = %d)" % (dic['name'], dic['usesInputs'], inp['name'], inp['usesActs'])
                    if not inp['usesActs'] and not dic['usesInputs']:
                        dic['actsTarget'] = i
                        print "Layer %s using acts from layer %s" % (dic['name'], inp['name'])
#                        print "Layer '%s' sharing activity matrix with layer '%s'" % (dic['name'], l['name'])
                    # I can share my gradient matrix with this layer if we're on the same GPU.
                    # This is different from the logic for actsTarget because this guy doesn't
                    # have an actsGrad matrix on my GPU if our GPUs are different, so there's
                    # nothing to share.
                    if dic['gpu'] == inp['gpu']:
                        dic['actsGradTarget'] = i
#                    print "Layer '%s' sharing activity gradient matrix with layer '%s'" % (dic['name'], l['name'])
            
    def parse(self, name, mcp, prev_layers, model=None):
        dic = LayerParser.parse(self, name, mcp, prev_layers, model)
        
        dic['inputs'] = [inp.strip() for inp in mcp.safe_get(name, 'inputs').split(',')]
        
        for inp in dic['inputs']:
            if inp not in prev_layers:
                raise LayerParsingError("Layer '%s': input layer '%s' not defined" % (name, inp))
            
        dic['inputLayers'] = [prev_layers[inp] for inp in dic['inputs']]
        dic['gpu'] = mcp.safe_get_int_list(name, 'gpu', default=dic['inputLayers'][0]['gpu'])
        dic['gpus'] = ", ".join('%s' % d for d in dic['gpu'])
        dic['numReplicas'] = len(dic['gpu'])
        
        if len(set(dic['gpu'])) != len(dic['gpu']):
            raise LayerParsingError("Layer '%s': all replicas must run on different GPUs." % (name))
        
        for inp in dic['inputs']:
            # Data layers do not explicitly define how many replicas they have.
            # The number of replicas for a data layer is given by the number of replicas
            # in the next layer(s). So we set that here.
            inpl = prev_layers[inp]
            if inpl['type'] == 'data':
                inpl['numReplicas'] = dic['numReplicas']
            if inpl['numReplicas'] % dic['numReplicas'] != 0:
                raise LayerParsingError("Layer '%s': number of replicas (%d) must divide number of replicas in all input layers (input %s has %d replicas)." % (name, dic['numReplicas'], inpl['name'], inpl['numReplicas']))
        if len(set(inp['numReplicas'] for inp in dic['inputLayers'])) != 1:
            raise LayerParsingError("Layer '%s': all input layers must have equal numbers of replicas." % (name))

        # Need to also assert that all *next* layers have equal number of replicas but this is hard so it's done in Layer.optimize
        for inp in dic['inputLayers']:
            if inp['outputs'] == 0:
                raise LayerParsingError("Layer '%s': input layer '%s' does not produce any output" % (name, inp['name']))
        dic['numInputs'] = [inp['outputs'] for inp in dic['inputLayers']]
        
        # Layers can declare a neuron activation function to apply to their output, as a shortcut
        # to avoid declaring a separate neuron layer above themselves.
        dic['neuron'] = mcp.safe_get(name, 'neuron', default="")
        if self.num_inputs > 0 and len(dic['numInputs']) != self.num_inputs:
            raise LayerParsingError("Layer '%s': number of inputs must be %d" % (name, self.num_inputs))
        
        if model:
            self.verify_all_ints_in(dic['gpu'], 'gpu', range(len(model.op.get_value('gpu'))))
        return dic
    
    def verify_img_size(self):
        dic = self.dic
        if dic['numInputs'][0] % dic['imgPixels'] != 0 or dic['imgSize'] * dic['imgSize'] != dic['imgPixels']:
            raise LayerParsingError("Layer '%s': has %-d dimensional input, not interpretable as %d-channel images" % (dic['name'], dic['numInputs'][0], dic['channels']))
    
    @staticmethod
    def grad_consumers_below(dic):
        if dic['gradConsumer']:
            return True
        if 'inputLayers' in dic:
            return any(LayerWithInputParser.grad_consumers_below(l) for l in dic['inputLayers'])
        
    def verify_no_grads(self):
        if LayerWithInputParser.grad_consumers_below(self.dic):
            raise LayerParsingError("Layer '%s': layers of type '%s' cannot propagate gradient and must not be placed over layers with parameters." % (self.dic['name'], self.dic['type']))

class NailbedLayerParser(LayerWithInputParser):
    def __init__(self):
        LayerWithInputParser.__init__(self, num_inputs=1)
        
    def parse(self, name, mcp, prev_layers, model=None):
        dic = LayerWithInputParser.parse(self, name, mcp, prev_layers, model)
        dic['forceOwnActs'] = False
        dic['usesActs'] = False
        dic['usesInputs'] = False
        
        dic['channels'] = mcp.safe_get_int(name, 'channels')
        dic['stride'] = mcp.safe_get_int(name, 'stride')

        self.verify_num_range(dic['channels'], 'channels', 1, None)
        
        # Computed values
        dic['imgPixels'] = dic['numInputs'][0] / dic['channels']
        dic['imgSize'] = int(n.sqrt(dic['imgPixels']))
        dic['outputsX'] = (dic['imgSize'] + dic['stride'] - 1) / dic['stride']
        dic['start'] = (dic['imgSize'] - dic['stride'] * (dic['outputsX'] - 1)) / 2
        dic['outputs'] = dic['channels'] * dic['outputsX']**2
        
        self.verify_num_range(dic['outputsX'], 'outputsX', 0, None)
        
        self.verify_img_size()
        
        print "Initialized bed-of-nails layer '%s' on GPUs %s, producing %dx%d %d-channel output" % (name, dic['gpus'], dic['outputsX'], dic['outputsX'], dic['channels'])
        return dic
    
class GaussianBlurLayerParser(LayerWithInputParser):
    def __init__(self):
        LayerWithInputParser.__init__(self, num_inputs=1)
        
    def parse(self, name, mcp, prev_layers, model=None):
        dic = LayerWithInputParser.parse(self, name, mcp, prev_layers, model)
        dic['forceOwnActs'] = False
        dic['usesActs'] = False
        dic['usesInputs'] = False
        dic['outputs'] = dic['numInputs'][0]
        
        dic['channels'] = mcp.safe_get_int(name, 'channels')
        dic['filterSize'] = mcp.safe_get_int(name, 'filterSize')
        dic['stdev'] = mcp.safe_get_float(name, 'stdev')

        self.verify_num_range(dic['channels'], 'channels', 1, None)
        self.verify_int_in(dic['filterSize'], 'filterSize', [3, 5, 7, 9])
        
        # Computed values
        dic['imgPixels'] = dic['numInputs'][0] / dic['channels']
        dic['imgSize'] = int(n.sqrt(dic['imgPixels']))
        dic['filter'] = n.array([exp(-(dic['filterSize']/2 - i)**2 / float(2 * dic['stdev']**2)) 
                                 for i in xrange(dic['filterSize'])], dtype=n.float32).reshape(1, dic['filterSize'])
        dic['filter'] /= dic['filter'].sum()
        self.verify_img_size()
        
        if dic['filterSize'] > dic['imgSize']:
            raise LayerParsingError("Later '%s': filter size (%d) must be smaller than image size (%d)." % (dic['name'], dic['filterSize'], dic['imgSize']))
        
        print "Initialized Gaussian blur layer '%s', producing %dx%d %d-channel output" % (name, dic['imgSize'], dic['imgSize'], dic['channels'])
        
        return dic
    
class HorizontalReflectionLayerParser(LayerWithInputParser):
    def __init__(self):
        LayerWithInputParser.__init__(self, num_inputs=1)
        
    def parse(self, name, mcp, prev_layers, model=None):
        dic = LayerWithInputParser.parse(self, name, mcp, prev_layers, model)
        dic['outputs'] = dic['numInputs'][0]
        dic['channels'] = mcp.safe_get_int(name, 'channels')
  
        self.verify_num_range(dic['channels'], 'channels', 1, 3)

        # Computed values
        dic['imgPixels'] = dic['numInputs'][0] / dic['channels']
        dic['imgSize'] = int(n.sqrt(dic['imgPixels']))
        self.verify_img_size()
        
        print "Initialized horizontal reflection layer '%s', producing %dx%d %d-channel output" % (name, dic['imgSize'], dic['imgSize'], dic['channels'])
        
        return dic
    
class ResizeLayerParser(LayerWithInputParser):
    def __init__(self):
        LayerWithInputParser.__init__(self, num_inputs=1)
        
    def parse(self, name, mcp, prev_layers, model=None):
        dic = LayerWithInputParser.parse(self, name, mcp, prev_layers, model)
        dic['forceOwnActs'] = False
        dic['usesActs'] = False
        dic['usesInputs'] = False
        
        dic['channels'] = mcp.safe_get_int(name, 'channels')
        dic['imgPixels'] = dic['numInputs'][0] / dic['channels']
        dic['imgSize'] = int(n.sqrt(dic['imgPixels']))
        
        dic['scale'] = mcp.safe_get_float(name, 'scale')
        dic['tgtSize'] = int(floor(dic['imgSize'] / dic['scale']))
        dic['tgtPixels'] = dic['tgtSize']**2
        self.verify_num_range(dic['channels'], 'channels', 1, None)
        # Really not recommended to use this for such severe scalings
        self.verify_float_range(dic['scale'], 'scale', 0.5, 2) 

        dic['outputs'] = dic['channels'] * dic['tgtPixels']
        
        self.verify_img_size()
        self.verify_no_grads()
        
        print "Initialized resize layer '%s', producing %dx%d %d-channel output" % (name, dic['tgtSize'], dic['tgtSize'], dic['channels'])
        
        return dic
    
class RandomScaleLayerParser(LayerWithInputParser):
    def __init__(self):
        LayerWithInputParser.__init__(self, num_inputs=1)
        
    def parse(self, name, mcp, prev_layers, model=None):
        dic = LayerWithInputParser.parse(self, name, mcp, prev_layers, model)
        dic['forceOwnActs'] = False
        dic['usesActs'] = False
        dic['usesInputs'] = False
        
        dic['channels'] = mcp.safe_get_int(name, 'channels')
        self.verify_num_range(dic['channels'], 'channels', 1, None)
        
        # Computed values
        dic['imgPixels'] = dic['numInputs'][0] / dic['channels']
        dic['imgSize'] = int(n.sqrt(dic['imgPixels']))
        
        dic['maxScale'] = mcp.safe_get_float(name, 'maxScale')
        dic['tgtSize'] = mcp.safe_get_int(name, 'tgtSize')
        min_size = int(floor(dic['imgSize'] / dic['maxScale']))
        max_size = dic['imgSize'] #int(floor(dic['imgSize'] * dic['maxScale']))
        if dic['tgtSize'] < min_size:
            raise LayerParsingError("Layer '%s': target size must be greater than minimum image size after rescaling (%d)" % (name, min_size))
        if dic['tgtSize'] > max_size:
            raise LayerParsingError("Layer '%s': target size must be smaller than maximum image size after rescaling (%d)" % (name, max_size))
        dic['tgtPixels'] = dic['tgtSize']**2
        
        self.verify_float_range(dic['maxScale'], 'maxScale', 1, 2) 

        dic['outputs'] = dic['channels'] * dic['tgtPixels']
        
        self.verify_img_size()
        self.verify_no_grads()
        
        print "Initialized random scale layer '%s', producing %dx%d %d-channel output" % (name, dic['tgtSize'], dic['tgtSize'], dic['channels'])
        
        return dic
    
class CropLayerParser(LayerWithInputParser):
    def __init__(self):
        LayerWithInputParser.__init__(self, num_inputs=1)
        
    def parse(self, name, mcp, prev_layers, model=None):
        dic = LayerWithInputParser.parse(self, name, mcp, prev_layers, model)
        dic['forceOwnActs'] = False
        dic['usesActs'] = False
        dic['usesInputs'] = False
        
        dic['channels'] = mcp.safe_get_int(name, 'channels')
        self.verify_num_range(dic['channels'], 'channels', 1, None)
        dic['startX'] = mcp.safe_get_int(name, 'startX')
        dic['startY'] = mcp.safe_get_int(name, 'startY', default=dic['startX'])
        dic['sizeX'] = mcp.safe_get_int(name, 'sizeX')
        
        # Computed values
        dic['imgPixels'] = dic['numInputs'][0] / dic['channels']
        dic['imgSize'] = int(n.sqrt(dic['imgPixels']))
     
        dic['outputs'] = dic['channels'] * (dic['sizeX']**2)
        
        self.verify_num_range(dic['startX'], 'startX', 0, dic['imgSize']-1)
        self.verify_num_range(dic['sizeX'], 'sizeX', 1, dic['imgSize'])
        self.verify_num_range(dic['startY'], 'startY', 0, dic['imgSize']-1)
        self.verify_img_size()
        self.verify_no_grads()
        
        if dic['startX'] + dic['sizeX'] > dic['imgSize']:
            raise LayerParsingError("Layer '%s': startX (%d) + sizeX (%d) > imgSize (%d)" % (name, dic['startX'], dic['sizeX'], dic['imgSize']))
        
        print "Initialized cropping layer '%s', producing %dx%d %d-channel output" % (name, dic['sizeX'], dic['sizeX'], dic['channels'])
        
        return dic
    
class ColorTransformLayerParser(LayerWithInputParser):
    def __init__(self):
        LayerWithInputParser.__init__(self, num_inputs=1)
    
    def parse(self, name, mcp, prev_layers, model=None):
        dic = LayerWithInputParser.parse(self, name, mcp, prev_layers, model)
        dic['forceOwnActs'] = False
        dic['usesActs'] = False
        dic['usesInputs'] = False

        # Computed values
        dic['imgPixels'] = dic['numInputs'][0] / 3
        dic['imgSize'] = int(n.sqrt(dic['imgPixels']))
        dic['channels'] = 3
        dic['outputs'] = dic['numInputs'][0]
        
        self.verify_img_size()
        self.verify_no_grads()
        
        return dic
    
class RGBToYUVLayerParser(ColorTransformLayerParser):
    def __init__(self):
        ColorTransformLayerParser.__init__(self)
        
    def parse(self, name, mcp, prev_layers, model=None):
        dic = ColorTransformLayerParser.parse(self, name, mcp, prev_layers, model)
        print "Initialized RGB --> YUV layer '%s', producing %dx%d %d-channel output" % (name, dic['imgSize'], dic['imgSize'], dic['channels'])
        return dic
    
class RGBToLABLayerParser(ColorTransformLayerParser):
    def __init__(self):
        ColorTransformLayerParser.__init__(self)
        
    def parse(self, name, mcp, prev_layers, model=None):
        dic = ColorTransformLayerParser.parse(self, name, mcp, prev_layers, model)
        dic['center'] = mcp.safe_get_bool(name, 'center', default=False)
        print "Initialized RGB --> LAB layer '%s', producing %dx%d %d-channel output" % (name, dic['imgSize'], dic['imgSize'], dic['channels'])
        return dic

class NeuronLayerParser(LayerWithInputParser):
    def __init__(self):
        LayerWithInputParser.__init__(self, num_inputs=1)
    
    @staticmethod
    def get_unused_layer_name(layers, wish):
        if wish not in layers:
            return wish
        for i in xrange(1, 100):
            name = '%s.%d' % (wish, i)
            if name not in layers:
                return name
        raise LayerParsingError("This is insane.")
    
    def parse_neuron(self, neuron_str):
        for n in neuron_parsers:
            p = n.parse(neuron_str)
            if p: # Successfully parsed neuron, return it
                self.dic['neuron'] = p
                self.dic['usesActs'] = self.dic['neuron']['usesActs']
                self.dic['usesInputs'] = self.dic['neuron']['usesInputs']
                
                return
        # Could not parse neuron
        # Print available neuron types
        colnames = ['Neuron type', 'Function']
        m = max(len(colnames[0]), OptionsParser._longest_value(neuron_parsers, key=lambda x:x.type)) + 2
        ntypes = [OptionsParser._bold(colnames[0].ljust(m))] + [n.type.ljust(m) for n in neuron_parsers]
        fnames = [OptionsParser._bold(colnames[1])] + [n.func_str for n in neuron_parsers]
        usage_lines = NL.join(ntype + fname for ntype,fname in zip(ntypes, fnames))
        
        raise LayerParsingError("Layer '%s': unable to parse neuron type '%s'. Valid neuron types: %sWhere neurons have parameters, they must be floats." % (self.dic['name'], neuron_str, NL + usage_lines + NL))
    
    def detach_neuron_layer(self, src_name, layers):
        dic = self.dic
#        self.set_defaults()
        dic['name'] = NeuronLayerParser.get_unused_layer_name(layers, '%s_neuron' % src_name)
        dic['type'] = 'neuron'
        dic['inputs'] = src_name
        dic['neuron'] = layers[src_name]['neuron']
        dic['gpu'] = layers[src_name]['gpu']
        
        # Yes it's not entirely correct to pass all of layers as prev_layers, but it's harmless
        dic = self.parse(dic['name'], FakeConfigParser(dic), layers)
        dic['src_layer'] = src_name
        
        # Link upper layers to this new one
        for l in layers.values():
            if 'inputs' in l:
                l['inputs'] = [inp if inp != src_name else dic['name'] for inp in l['inputs']]
                l['inputLayers'] = [inp if inp['name'] != src_name else dic for inp in l['inputLayers']]
        layers[dic['name']] = dic
    
    def parse(self, name, mcp, prev_layers, model=None):
        dic = LayerWithInputParser.parse(self, name, mcp, prev_layers, model)
        dic['outputs'] = dic['numInputs'][0]
        self.parse_neuron(dic['neuron'])
        dic['forceOwnActs'] = False
        print "Initialized neuron layer '%s' on GPUs %s, producing %d outputs" % (name, dic['gpus'], dic['outputs'])
        return dic

class EltwiseSumLayerParser(LayerWithInputParser):
    def __init__(self):
        LayerWithInputParser.__init__(self)
    
    def add_params(self, mcp):
        LayerWithInputParser.add_params(self, mcp)
        dic, name = self.dic, self.dic['name']
        dic['coeffs'] = mcp.safe_get_float_list(name, 'coeffs', default=[1.0] * len(dic['inputs']))
    
    def parse(self, name, mcp, prev_layers, model):
        dic = LayerWithInputParser.parse(self, name, mcp, prev_layers, model)
        
        if len(set(dic['numInputs'])) != 1:
            raise LayerParsingError("Layer '%s': all inputs must have the same dimensionality. Got dimensionalities: %s" % (name, ", ".join(str(s) for s in dic['numInputs'])))
        dic['outputs'] = dic['numInputs'][0]
        dic['usesInputs'] = False
        dic['usesActs'] = False
        dic['forceOwnActs'] = False
        dic['requiresParams'] = True        
        
        print "Initialized elementwise sum layer '%s' on GPUs %s, producing %d outputs" % (name, dic['gpus'], dic['outputs'])
        return dic
    
class EltwiseMaxLayerParser(LayerWithInputParser):
    def __init__(self):
        LayerWithInputParser.__init__(self)
        
    def parse(self, name, mcp, prev_layers, model):
        dic = LayerWithInputParser.parse(self, name, mcp, prev_layers, model)
        if len(dic['inputs']) < 2:
            raise LayerParsingError("Layer '%s': elementwise max layer must have at least 2 inputs, got %d." % (name, len(dic['inputs'])))
        if len(set(dic['numInputs'])) != 1:
            raise LayerParsingError("Layer '%s': all inputs must have the same dimensionality. Got dimensionalities: %s" % (name, ", ".join(str(s) for s in dic['numInputs'])))
        dic['outputs'] = dic['numInputs'][0]

        print "Initialized elementwise max layer '%s' on GPUs %s, producing %d outputs" % (name, dic['gpus'], dic['outputs'])
        return dic
    
class SumLayerParser(LayerWithInputParser):
    def __init__(self):
        LayerWithInputParser.__init__(self, num_inputs=1)
        
    def parse(self, name, mcp, prev_layers, model):
        dic = LayerWithInputParser.parse(self, name, mcp, prev_layers, model)
        
        dic['stride'] = mcp.safe_get_int(name, 'stride', default=1)
        self.verify_divisible(dic['numInputs'][0], dic['stride'], 'input dimensionality', 'stride')
        dic['outputs'] = dic['numInputs'][0] / dic['stride']
    
        print "Initialized sum layer '%s' on GPUs %s, producing %d outputs" % (name, dic['gpus'], dic['outputs'])
        return dic
    
class DropoutLayerParser(LayerWithInputParser):
    def __init__(self):
        LayerWithInputParser.__init__(self, num_inputs=1)
        
    def add_params(self, mcp):
        LayerWithInputParser.add_params(self, mcp)
        dic, name = self.dic, self.dic['name']
        dic['enable'] = mcp.safe_get_bool(name, 'enable', default=True)
        dic['keep'] = mcp.safe_get_float(name, 'keep', default=0.5)
        
    def parse(self, name, mcp, prev_layers, model):
        dic = LayerWithInputParser.parse(self, name, mcp, prev_layers, model)
        dic['requiresParams'] = True
        dic['usesInputs'] = False
        dic['usesActs'] = False
        dic['forceOwnActs'] = False
        dic['outputs'] = dic['numInputs'][0]

        print "Initialized %s layer '%s' on GPUs %s, producing %d outputs" % (dic['type'], name, dic['gpus'], dic['outputs'])
        return dic
    
class Dropout2LayerParser(DropoutLayerParser):
    def __init__(self):
        DropoutLayerParser.__init__(self)
    
class WeightLayerParser(LayerWithInputParser):
    LAYER_PAT = re.compile(r'^\s*([^\s\[]+)(?:\[(\d+)\])?\s*$') # matches things like layername[5], etc
    
    def __init__(self, num_inputs=-1):
        LayerWithInputParser.__init__(self, num_inputs=num_inputs)
    
    @staticmethod
    def get_layer_name(name_str):
        m = WeightLayerParser.LAYER_PAT.match(name_str)
        if not m:
            return None
        return m.group(1), m.group(2)
        
    def add_params(self, mcp):
        LayerWithInputParser.add_params(self, mcp)
        dic, name = self.dic, self.dic['name']
        dic['momW'] = mcp.safe_get_float_list(name, 'momW')
        dic['momB'] = mcp.safe_get_float(name, 'momB')
        dic['superEps'] = mcp.safe_get_float(name, 'superEps', default=0.0)
        dic['superMom'] = mcp.safe_get_float(name, 'superMom', default=0.0)
        dic['wc'] = mcp.safe_get_float_list(name, 'wc', default=[0.0] * len(dic['inputs']))
        dic['wball'] = mcp.safe_get_float_list(name, 'wball', default=[0.0] * len(dic['inputs']))
        self.verify_num_params(['momW', 'wc', 'wball'])
#        dic['wballNormed'] = [wball * nweights for wball,nweights in zip(dic['wball'], dic['weightsPerFilter'])]
        dic['wballNormed'] = dic['wball']
        
        # Convert from old-style 0.001,0.02 hyperparam specification to new-stye
        # const[base=0.001],const[base=0.02] and so forth
        def convert_scalars_to_schedules(scalars):
            parts = scalars.split(',')
            for i,p in enumerate(parts):
                p = p.strip()
                if re.match('(?:\d*\.)?\d+$', p):
                    parts[i] = 'const[base=%s]' % p
            return parts
            
        dic['epsW'] = self.parse_params(convert_scalars_to_schedules(mcp.safe_get(name, 'epsW')), lrs_parsers, 'epsW', 'learning rate schedule', num_params=len(dic['inputs']))
        dic['epsB'] = self.parse_params(convert_scalars_to_schedules(mcp.safe_get(name, 'epsB')), lrs_parsers, 'epsB', 'learning rate schedule', num_params=1)[0]
        
        dic['updatePeriod'] = mcp.safe_get_int(name, 'updatePeriod', default=0) # 0 means update as often as possible
        # TODO: assert that updatePeriod is a multiple of active pass period, which is unknown here.
        # the assert has to go in some post-processing step..
        dic['gradConsumer'] = dic['epsB']['params']['base'] > 0 or any(w['params']['base'] > 0 for w in dic['epsW'])

    @staticmethod
    def unshare_weights(layer, layers, matrix_idx=None):
        def unshare(layer, layers, indices):
            for i in indices:
                if layer['weightSourceLayers'][i] >= 0:
                    src_matrix_idx = layer['weightSourceMatrixIndices'][i]
                    layer['weightSourceLayers'][i] = ""
                    layer['weightSourceMatrixIndices'][i] = -1
                    layer['weights'][i] = layer['weights'][i].copy()
                    layer['weightsInc'][i] = n.zeros_like(layer['weights'][i])
                    print "Unshared weight matrix %s[%d] from %s[%d]." % (layer['name'], i, layer['weightSourceLayers'][i], src_matrix_idx)
                else:
                    print "Weight matrix %s[%d] already unshared." % (layer['name'], i)
        if 'weightSourceLayers' in layer:
            unshare(layer, layers, range(len(layer['inputs'])) if matrix_idx is None else [matrix_idx])

    # Load weight/biases initialization module
    def call_init_func(self, param_name, shapes, input_idx=-1):
        dic = self.dic
        func_pat = re.compile('^([^\.]+)\.([^\(\)]+)\s*(?:\(([^,]+(?:,[^,]+)*)\))?$')
        m = func_pat.match(dic[param_name])
        if not m:
            raise LayerParsingError("Layer '%s': '%s' parameter must have format 'moduleName.functionName(param1,param2,...)'; got: %s." % (dic['name'], param_name, dic['initWFunc']))
        module, func = m.group(1), m.group(2)
        params = m.group(3).split(',') if m.group(3) is not None else []
        try:
            mod = __import__(module)
            return getattr(mod, func)(dic['name'], input_idx, shapes, params=params) if input_idx >= 0 else getattr(mod, func)(dic['name'], shapes, params=params)
        except (ImportError, AttributeError, TypeError), e:
            raise LayerParsingError("Layer '%s': %s." % (dic['name'], e))
        
    def make_weights(self, initW, rows, cols, order='C'):
        dic = self.dic
        dic['weights'], dic['weightsInc'] = [], []
        if dic['initWFunc']: # Initialize weights from user-supplied python function
            # Initialization function is supplied in the format
            # module.func
            for i in xrange(len(dic['inputs'])):
                dic['weights'] += [self.call_init_func('initWFunc', (rows[i], cols[i]), input_idx=i)]

                if type(dic['weights'][i]) != n.ndarray:
                    raise LayerParsingError("Layer '%s[%d]': weight initialization function %s must return numpy.ndarray object. Got: %s." % (dic['name'], i, dic['initWFunc'], type(dic['weights'][i])))
                if dic['weights'][i].dtype != n.float32:
                    raise LayerParsingError("Layer '%s[%d]': weight initialization function %s must weight matrices consisting of single-precision floats. Got: %s." % (dic['name'], i, dic['initWFunc'], dic['weights'][i].dtype))
                if dic['weights'][i].shape != (rows[i], cols[i]):
                    raise LayerParsingError("Layer '%s[%d]': weight matrix returned by weight initialization function %s has wrong shape. Should be: %s; got: %s." % (dic['name'], i, dic['initWFunc'], (rows[i], cols[i]), dic['weights'][i].shape))
                # Convert to desired order
                dic['weights'][i] = n.require(dic['weights'][i], requirements=order)
                dic['weightsInc'] += [n.zeros_like(dic['weights'][i])]
                print "Layer '%s[%d]' initialized weight matrices from function %s" % (dic['name'], i, dic['initWFunc'])
        else:
            for i in xrange(len(dic['inputs'])):
                if dic['weightSourceLayers'][i] != '': # Shared weight matrix
                    src_layer = self.prev_layers[dic['weightSourceLayers'][i]] if dic['weightSourceLayers'][i] != dic['name'] else dic
                    dic['weights'] += [src_layer['weights'][dic['weightSourceMatrixIndices'][i]]]
                    dic['weightsInc'] += [src_layer['weightsInc'][dic['weightSourceMatrixIndices'][i]]]
                    if dic['weights'][i].shape != (rows[i], cols[i]):
                        raise LayerParsingError("Layer '%s': weight sharing source matrix '%s' has shape %dx%d; should be %dx%d." 
                                                % (dic['name'], dic['weightSource'][i], dic['weights'][i].shape[0], dic['weights'][i].shape[1], rows[i], cols[i]))
                    print "Layer '%s' initialized weight matrix %d from %s" % (dic['name'], i, dic['weightSource'][i])
                else:
                    dic['weights'] += [n.array(initW[i] * nr.randn(rows[i], cols[i]), dtype=n.single, order=order)]
                    dic['weightsInc'] += [n.zeros_like(dic['weights'][i])]
        
    def make_biases(self, rows, cols, order='C'):
        dic = self.dic
        if dic['initBFunc']:
            dic['biases'] = self.call_init_func('initBFunc', (rows, cols))
            if type(dic['biases']) != n.ndarray:
                raise LayerParsingError("Layer '%s': bias initialization function %s must return numpy.ndarray object. Got: %s." % (dic['name'], dic['initBFunc'], type(dic['biases'])))
            if dic['biases'].dtype != n.float32:
                raise LayerParsingError("Layer '%s': bias initialization function %s must return numpy.ndarray object consisting of single-precision floats. Got: %s." % (dic['name'], dic['initBFunc'], dic['biases'].dtype))
            if dic['biases'].shape != (rows, cols):
                raise LayerParsingError("Layer '%s': bias vector returned by bias initialization function %s has wrong shape. Should be: %s; got: %s." % (dic['name'], dic['initBFunc'], (rows, cols), dic['biases'].shape))

            dic['biases'] = n.require(dic['biases'], requirements=order)
            print "Layer '%s' initialized bias vector from function %s" % (dic['name'], dic['initBFunc'])
        else:
            dic['biases'] = dic['initB'] * n.ones((rows, cols), order=order, dtype=n.single)
        dic['biasesInc'] = n.zeros_like(dic['biases'])
        
    def parse(self, name, mcp, prev_layers, model):
        dic = LayerWithInputParser.parse(self, name, mcp, prev_layers, model)
        dic['requiresParams'] = True
        dic['gradConsumer'] = True
        dic['usesActs'] = False
        dic['initW'] = mcp.safe_get_float_list(name, 'initW', default=0.01)
        dic['initB'] = mcp.safe_get_float(name, 'initB', default=0)
        dic['initWFunc'] = mcp.safe_get(name, 'initWFunc', default="")
        dic['initBFunc'] = mcp.safe_get(name, 'initBFunc', default="")
        # Find shared weight matrices
        
        dic['weightSource'] = mcp.safe_get_list(name, 'weightSource', default=[''] * len(dic['inputs']))
        self.verify_num_params(['initW'])
        self.verify_num_params(['weightSource'], auto_expand=False)
        
        dic['weightSourceLayers'] = []
        dic['weightSourceMatrixIndices'] = []

        for i, src_name in enumerate(dic['weightSource']):
            src_layer_matrix_idx = -1
            src_layer_name = ''
            if src_name != '':
                src_layer_match = WeightLayerParser.get_layer_name(src_name)
                if src_layer_match is None:
                    raise LayerParsingError("Layer '%s': unable to parse weight sharing source '%s'. Format is layer[idx] or just layer, in which case idx=0 is used." % (name, src_name))
                src_layer_name = src_layer_match[0]
                src_layer_matrix_idx = int(src_layer_match[1]) if src_layer_match[1] is not None else 0

                if src_layer_name not in prev_layers and src_layer_name != name:
                    raise LayerParsingError("Layer '%s': weight sharing source layer '%s' does not exist." % (name, src_layer_name))
                
#                src_layer_idx = prev_names.index(src_layer_name) if src_layer_name != name else len(prev_names)
                src_layer = prev_layers[src_layer_name] if src_layer_name != name else dic
                if src_layer['gpu'] != dic['gpu']:
                    raise LayerParsingError("Layer '%s': weight sharing source layer '%s' runs on GPUs %s, while '%s' runs on GPUs %s." % (name, src_layer_name, src_layer['gpu'], name, dic['gpu']))
                if src_layer['type'] != dic['type']:
                    raise LayerParsingError("Layer '%s': weight sharing source layer '%s' is of type '%s'; should be '%s'." % (name, src_layer_name, src_layer['type'], dic['type']))
                if src_layer_name != name and len(src_layer['weights']) <= src_layer_matrix_idx:
                    raise LayerParsingError("Layer '%s': weight sharing source layer '%s' has %d weight matrices, but '%s[%d]' requested." % (name, src_layer_name, len(src_layer['weights']), src_name, src_layer_matrix_idx))
                if src_layer_name == name and src_layer_matrix_idx >= i:
                    raise LayerParsingError("Layer '%s': weight sharing source '%s[%d]' not defined yet." % (name, name, src_layer_matrix_idx))

            dic['weightSourceLayers'] += [src_layer_name]
            dic['weightSourceMatrixIndices'] += [src_layer_matrix_idx]
                
        return dic
        
class FCLayerParser(WeightLayerParser):
    def __init__(self):
        WeightLayerParser.__init__(self)
        
    def parse(self, name, mcp, prev_layers, model):
        dic = WeightLayerParser.parse(self, name, mcp, prev_layers, model)
        
        dic['outputs'] = mcp.safe_get_int(name, 'outputs')
        dic['weightsPerFilter'] = dic['numInputs']
        self.verify_num_range(dic['outputs'], 'outputs', 1, None)
        self.make_weights(dic['initW'], dic['numInputs'], [dic['outputs']] * len(dic['numInputs']), order='F')
        self.make_biases(1, dic['outputs'], order='F')

        print "Initialized fully-connected layer '%s' on GPUs %s, producing %d outputs" % (name, dic['gpus'], dic['outputs'])
        return dic
    
class SplitFCLayerParser(WeightLayerParser):
    def __init__(self):
        WeightLayerParser.__init__(self)
    
    def parse(self, name, mcp, prev_layers, model):
        dic = WeightLayerParser.parse(self, name, mcp, prev_layers, model)
        dic['parts'] = mcp.safe_get_int(name, 'parts')
        dic['outputs'] = mcp.safe_get_int(name, 'outputs') * dic['parts']
        dic['weightsPerFilter'] = dic['numInputs']
        self.verify_num_range(dic['parts'], 'parts', 1, None)
        
        self.make_weights(dic['initW'], dic['numInputs'], [dic['outputs']/dic['parts']] * len(dic['numInputs']), order='F')
        self.make_biases(1, dic['outputs'], order='F')
        
        for i in xrange(len(dic['numInputs'])):
            self.verify_divisible(dic['numInputs'][i], dic['parts'], 'numInputs', 'parts', input_idx=i)
            
        print "Initialized split fully-connected layer '%s' on GPUs %s, producing %d outputs in %d parts" % (name, dic['gpus'], dic['outputs'], dic['parts'])
        return dic
    
class LocalLayerParser(WeightLayerParser):
    def __init__(self):
        WeightLayerParser.__init__(self)
        
    # Convert convolutional layer to unshared, locally-connected layer
    @staticmethod
    def conv_to_local(layers, lname):
        layer = layers[lname]
        if layer['type'] == 'conv':
            layer['type'] = 'local'
            for inp,inpname in enumerate(layer['inputs']):
                src_layer_name = layer['weightSourceLayers'][inp]
                if src_layer_name != '':
                    src_layer = layers[src_layer_name]
                    src_matrix_idx = layer['weightSourceMatrixIndices'][inp]
                    LocalLayerParser.conv_to_local(layers, src_layer_name)
                    for w in ('weights', 'weightsInc'):
                        layer[w][inp] = src_layer[w][src_matrix_idx]
                else:
                    layer['weights'][inp] = n.require(n.reshape(n.tile(n.reshape(layer['weights'][inp], (1, n.prod(layer['weights'][inp].shape))), (layer['modules'], 1)),
                                                        (layer['modules'] * layer['filterChannels'][inp] * layer['filterPixels'][inp], layer['filters'])),
                                                      requirements='C')
                    layer['weightsInc'][inp] = n.zeros_like(layer['weights'][inp])
            if layer['sharedBiases']:
                layer['biases'] = n.require(n.repeat(layer['biases'], layer['modules'], axis=0), requirements='C')
                layer['biasesInc'] = n.zeros_like(layer['biases'])
            
            print "Converted layer '%s' from convolutional to unshared, locally-connected" % layer['name']
            
            # Also call this function on any layers sharing my weights
            for l in layers:
                if 'weightSourceLayers' in l and lname in l['weightSourceLayers']:
                    LocalLayerParser.conv_to_local(layers, l)
        return layer
        
    def parse(self, name, mcp, prev_layers, model):
        dic = WeightLayerParser.parse(self, name, mcp, prev_layers, model)
        dic['requiresParams'] = True
        dic['usesActs'] = False
        # Supplied values
        dic['channels'] = mcp.safe_get_int_list(name, 'channels')
        dic['padding'] = mcp.safe_get_int_list(name, 'padding', default=[0]*len(dic['inputs']))
        dic['stride'] = mcp.safe_get_int_list(name, 'stride', default=[1]*len(dic['inputs']))
        dic['filterSize'] = mcp.safe_get_int_list(name, 'filterSize')
        dic['filters'] = mcp.safe_get_int_list(name, 'filters')
        dic['groups'] = mcp.safe_get_int_list(name, 'groups', default=[1]*len(dic['inputs']))
        dic['initW'] = mcp.safe_get_float_list(name, 'initW')
        dic['initCFunc'] = mcp.safe_get(name, 'initCFunc', default='')
        dic['modulesX'] = mcp.safe_get_int(name, 'modulesX', default=0)

        
        self.verify_num_params(['channels', 'padding', 'stride', 'filterSize', \
                                'filters', 'groups', 'initW'])
        
        self.verify_num_range(dic['stride'], 'stride', 1, None)
        self.verify_num_range(dic['filterSize'],'filterSize', 1, None)  
        self.verify_num_range(dic['padding'], 'padding', 0, None)
        self.verify_num_range(dic['channels'], 'channels', 1, None)
        self.verify_num_range(dic['groups'], 'groups', 1, None)
        self.verify_num_range(dic['modulesX'], 'modulesX', 0, None)
        for i in xrange(len(dic['filters'])):
            self.verify_divisible(dic['filters'][i], 16, 'filters', input_idx=i)
        
        # Computed values
        dic['imgPixels'] = [numInputs/channels for numInputs,channels in zip(dic['numInputs'], dic['channels'])]
        dic['imgSize'] = [int(n.sqrt(imgPixels)) for imgPixels in dic['imgPixels']]
        self.verify_num_range(dic['imgSize'], 'imgSize', 1, None)
        dic['filters'] = [filters*groups for filters,groups in zip(dic['filters'], dic['groups'])]
        dic['filterPixels'] = [filterSize**2 for filterSize in dic['filterSize']]
        if dic['modulesX'] <= 0:
            dic['modulesX'] = [1 + int(ceil((2*padding + imgSize - filterSize) / float(stride))) for padding,imgSize,filterSize,stride in zip(dic['padding'], dic['imgSize'], dic['filterSize'], dic['stride'])]
        else:
            dic['modulesX'] = [dic['modulesX']] * len(dic['inputs'])

        dic['filterChannels'] = [channels/groups for channels,groups in zip(dic['channels'], dic['groups'])]
        
        if len(set(dic['modulesX'])) != 1 or len(set(dic['filters'])) != 1:
            raise LayerParsingError("Layer '%s': all inputs must produce equally-dimensioned output. Dimensions are: %s." % (name, ", ".join("%dx%dx%d" % (filters, modulesX, modulesX) for filters,modulesX in zip(dic['filters'], dic['modulesX']))))

        dic['modulesX'] = dic['modulesX'][0]
        dic['modules'] = dic['modulesX']**2
        dic['filters'] = dic['filters'][0]
        dic['outputs'] = dic['modules'] * dic['filters']
#        dic['filterConns'] = [[]] * len(dic['inputs'])
        for i in xrange(len(dic['inputs'])):
            if dic['numInputs'][i] % dic['imgPixels'][i] != 0 or dic['imgSize'][i] * dic['imgSize'][i] != dic['imgPixels'][i]:
                raise LayerParsingError("Layer '%s[%d]': has %-d dimensional input, not interpretable as square %d-channel images" % (name, i, dic['numInputs'][i], dic['channels'][i]))
            if dic['channels'][i] > 3 and dic['channels'][i] % 4 != 0:
                raise LayerParsingError("Layer '%s[%d]': number of channels must be smaller than 4 or divisible by 4" % (name, i))
#            if dic['filterSize'][i] > totalPadding[i] + dic['imgSize'][i]:
#                raise LayerParsingError("Layer '%s[%d]': filter size (%d) greater than image size + padding (%d)" % (name, i, dic['filterSize'][i], dic['padding'][i] + dic['imgSize'][i]))
            if -dic['padding'][i] + dic['stride'][i] * (dic['modulesX'] - 1) + dic['filterSize'][i] < dic['imgSize'][i]:
                raise LayerParsingError("Layer '%s[%d]': %dx%d output map with padding=%d, stride=%d does not cover entire input image." % (name, i, dic['modulesX'], dic['outputsX'], dic['padding'][i], dic['stride'][i]))

            if dic['groups'][i] > 1:
                self.verify_divisible(dic['channels'][i], 4*dic['groups'][i], 'channels', '4 * groups', input_idx=i)
            self.verify_divisible(dic['channels'][i], dic['groups'][i], 'channels', 'groups', input_idx=i)

            self.verify_divisible(dic['filters'], 16*dic['groups'][i], 'filters * groups', input_idx=i)
            
        
            dic['padding'][i] = -dic['padding'][i]
#        dic['overSample'] = [groups*filterChannels/channels for groups,filterChannels,channels in zip(dic['groups'], dic['filterChannels'], dic['channels'])]
        dic['weightsPerFilter'] = [fc * (fz**2) for fc, fz in zip(dic['filterChannels'], dic['filterSize'])]
        
        return dic    

class ConvLayerParser(LocalLayerParser):
    def __init__(self):
        LocalLayerParser.__init__(self)
        
    def add_params(self, mcp):
        LocalLayerParser.add_params(self, mcp)
        self.dic['wcNormMax'] = mcp.safe_get_float_list(self.dic['name'], 'wcNormMax', default=[0.0] * len(self.dic['inputs']))
        self.dic['wcNormMin'] = mcp.safe_get_float_list(self.dic['name'], 'wcNormMin', default=[0.0] * len(self.dic['inputs']))
        self.verify_num_params(['wcNormMax', 'wcNormMin'])
        for min,max in zip(self.dic['wcNormMin'], self.dic['wcNormMax']):
            if min > max:
                raise LayerParsingError("Layer '%s': wcNormMin must be <= wcNormMax." % (self.dic['name']))
        
    def parse(self, name, mcp, prev_layers, model):
        dic = LocalLayerParser.parse(self, name, mcp, prev_layers, model)
        
        dic['sumWidth'] = mcp.safe_get_int(name, 'sumWidth')
        dic['sharedBiases'] = mcp.safe_get_bool(name, 'sharedBiases', default=True)
        
        num_biases = dic['filters'] if dic['sharedBiases'] else dic['modules']*dic['filters']

        eltmult = lambda list1, list2: [l1 * l2 for l1,l2 in zip(list1, list2)]
        self.make_weights(dic['initW'], eltmult(dic['filterPixels'], dic['filterChannels']), [dic['filters']] * len(dic['inputs']), order='C')
        self.make_biases(num_biases, 1, order='C')

        print "Initialized convolutional layer '%s' on GPUs %s, producing %dx%d %d-channel output" % (name, dic['gpus'], dic['modulesX'], dic['modulesX'], dic['filters'])
        return dic    
    
class LocalUnsharedLayerParser(LocalLayerParser):
    def __init__(self):
        LocalLayerParser.__init__(self)
        
    def parse(self, name, mcp, prev_layers, model):
        dic = LocalLayerParser.parse(self, name, mcp, prev_layers, model)
        eltmult = lambda list1, list2: [l1 * l2 for l1,l2 in zip(list1, list2)]
        scmult = lambda x, lst: [x * l for l in lst]
        self.make_weights(dic['initW'], scmult(dic['modules'], eltmult(dic['filterPixels'], dic['filterChannels'])), [dic['filters']] * len(dic['inputs']), order='C')
        self.make_biases(dic['modules'] * dic['filters'], 1, order='C')
        
        print "Initialized locally-connected layer '%s' on GPUs %s, producing %dx%d %d-channel output" % (name, dic['gpus'], dic['modulesX'], dic['modulesX'], dic['filters'])
        return dic  
    
class DataLayerParser(LayerParser):
    def __init__(self):
        LayerParser.__init__(self)
        
    def parse(self, name, mcp, prev_layers, model):
        dic = LayerParser.parse(self, name, mcp, prev_layers, model)
        dic['dataIdx'] = mcp.safe_get_int(name, 'dataIdx')
        dic['start'] = mcp.safe_get_int(name, 'start', default=0)
        dic['end'] = mcp.safe_get_int(name, 'end', default=model.train_data_provider.get_data_dims(idx=dic['dataIdx']))
        dic['outputs'] = dic['end'] - dic['start']
#        dic['usesActs'] = False
        print "Initialized data layer '%s', producing %d outputs" % (name, dic['outputs'])
        return dic

class SoftmaxLayerParser(LayerWithInputParser):
    def __init__(self):
        LayerWithInputParser.__init__(self, num_inputs=1)
        
    def parse(self, name, mcp, prev_layers, model):
        dic = LayerWithInputParser.parse(self, name, mcp, prev_layers, model)
        dic['outputs'] = dic['inputLayers'][0]['outputs']
        print "Initialized softmax layer '%s' on GPUs %s, producing %d outputs" % (name, dic['gpus'], dic['outputs'])
        return dic
    
class ConcatentionLayerParser(LayerWithInputParser):
    def __init__(self):
        LayerWithInputParser.__init__(self)
        
    def parse(self, name, mcp, prev_layers, model):
        dic = LayerWithInputParser.parse(self, name, mcp, prev_layers, model)
        dic['outputs'] = sum(l['outputs'] for l in dic['inputLayers'])
        dic['copyOffsets'] = [sum(dic['inputLayers'][j]['outputs'] for j in xrange(i)) for i in xrange(len(dic['inputLayers']))]
        print "Initialized concatenation layer '%s' on GPUs %s, producing %d outputs" % (name, dic['gpus'], dic['outputs'])
        return dic
    
class PassThroughLayerParser(LayerWithInputParser):
    def __init__(self):
        LayerWithInputParser.__init__(self)
        
    # Note: this doesn't verify all the necessary constraints. Layer construction may still fail in C++ code.
    # For example, it does not verify that every layer only has one pass-through parent. Obviously having 
    # two such parents is incoherent.
    def parse(self, name, mcp, prev_layers, model):
        dic = LayerWithInputParser.parse(self, name, mcp, prev_layers, model)
#        if len(dic['inputLayers']) == 1:
#            raise LayerParsingError("Layer %s: pass-through layer must have more than one input." % dic['name'])
        if len(dic['gpu']) != len(dic['inputLayers'][0]['gpu']):
            raise LayerParsingError("Layer '%s': number of replicas in pass-through layer must be equivalent to number of replicas in input layers." % dic['name'])
        for inp in dic['inputLayers']:
            conflicting_layers = [l for l in prev_layers.values() if l['type'] == 'pass' and inp['name'] in l['inputs'] and len(set(dic['gpu']).intersection(set(l['gpu']))) > 0]
            if len(conflicting_layers) > 0:
                raise LayerParsingError("Layer '%s' conflicts with layer '%s'. Both pass-through layers take layer '%s' as input and operate on an overlapping set of GPUs." % (dic['name'], conflicting_layers[0]['name'], inp['name']))
        dic['outputs'] = sum(l['outputs'] for l in dic['inputLayers'])
#        dic['copyOffsets'] = [sum(dic['inputLayers'][j]['outputs'] for j in xrange(i)) for i in xrange(len(dic['inputLayers']))]
        print "Initialized pass-through layer '%s' on GPUs %s, producing %d outputs" % (name, dic['gpus'], dic['outputs'])
        return dic

class PoolLayerParser(LayerWithInputParser):
    def __init__(self):
        LayerWithInputParser.__init__(self, num_inputs=1)
    
    def add_params(self, mcp):
        LayerWithInputParser.add_params(self, mcp)
        dic, name = self.dic, self.dic['name']
    
    def parse(self, name, mcp, prev_layers, model):
        dic = LayerWithInputParser.parse(self, name, mcp, prev_layers, model)
        dic['channels'] = mcp.safe_get_int(name, 'channels')
        dic['sizeX'] = mcp.safe_get_int(name, 'sizeX')
        dic['start'] = mcp.safe_get_int(name, 'start', default=0)
        dic['stride'] = mcp.safe_get_int(name, 'stride')
        dic['outputsX'] = mcp.safe_get_int(name, 'outputsX', default=0)
        dic['pool'] = mcp.safe_get(name, 'pool')
        
        # Avg pooler does not use its acts or inputs
        dic['usesActs'] = dic['pool'] != 'avg'
        dic['usesInputs'] = dic['pool'] != 'avg'
        
        dic['imgPixels'] = dic['numInputs'][0] / dic['channels']
        dic['imgSize'] = int(n.sqrt(dic['imgPixels']))
        
        if dic['pool'] == 'avg':
            dic['sum'] = mcp.safe_get_bool(name, 'sum', default=False)
        
        self.verify_num_range(dic['sizeX'], 'sizeX', 1, dic['imgSize'])
        self.verify_num_range(dic['stride'], 'stride', 1, dic['sizeX'])
        self.verify_num_range(dic['outputsX'], 'outputsX', 0, None)
        self.verify_num_range(dic['channels'], 'channels', 1, None)
        
        if LayerWithInputParser.grad_consumers_below(dic):
            self.verify_divisible(dic['channels'], 16, 'channels')
        self.verify_str_in(dic['pool'], 'pool', ['max', 'maxabs', 'avg'])
        
        self.verify_img_size()

        if dic['outputsX'] <= 0:
            dic['outputsX'] = int(ceil((dic['imgSize'] - dic['start'] - dic['sizeX']) / float(dic['stride']))) + 1;
        dic['outputs'] = dic['outputsX']**2 * dic['channels']
        
        print "Initialized %s-pooling layer '%s' on GPUs %s, producing %dx%d %d-channel output" % (dic['pool'], name, dic['gpus'], dic['outputsX'], dic['outputsX'], dic['channels'])
        return dic
    

class CrossMapPoolLayerParser(LayerWithInputParser):
    def __init__(self):
        LayerWithInputParser.__init__(self, num_inputs=1)
    
    def parse(self, name, mcp, prev_layers, model):
        dic = LayerWithInputParser.parse(self, name, mcp, prev_layers, model)
        dic['channels'] = mcp.safe_get_int(name, 'channels')
        dic['size'] = mcp.safe_get_int(name, 'size')
        dic['start'] = mcp.safe_get_int(name, 'start', default=0)
        dic['stride'] = mcp.safe_get_int(name, 'stride')
        dic['outputChannels'] = mcp.safe_get_int(name, 'outputs', default=0)
        dic['pool'] = mcp.safe_get(name, 'pool')
        dic['requiresParams'] = False
        
        # Avg pooler does not use its acts or inputs
        dic['usesActs'] = 'pool' != 'avg'
        dic['usesInputs'] = 'pool' != 'avg'
        
        dic['imgPixels'] = dic['numInputs'][0] / dic['channels']
        dic['imgSize'] = int(n.sqrt(dic['imgPixels']))
        dic['outputs'] = dic['outputChannels'] * dic['imgPixels']
        
        self.verify_num_range(dic['size'], 'size', 1, dic['channels'])
        self.verify_num_range(dic['stride'], 'stride', 1, dic['size'])
        self.verify_num_range(dic['outputChannels'], 'outputChannels', 0, None)
        self.verify_num_range(dic['channels'], 'channels', 1, None)
        self.verify_num_range(dic['start'], 'start', None, 0)
        
        self.verify_str_in(dic['pool'], 'pool', ['max'])
        self.verify_img_size()
        
        covered_chans = dic['start'] + (dic['outputChannels'] - 1) * dic['stride'] + dic['size']
        if covered_chans < dic['channels']:
            raise LayerParsingError("Layer '%s': cross-map pooling with start=%d, stride=%d, size=%d, outputs=%d covers only %d of %d input channels." % \
                                    (name, dic['start'], dic['stride'], dic['size'], dic['outputChannels'], covered_chans, dic['channels']))
        
        print "Initialized cross-map %s-pooling layer '%s' on GPUs %s, producing %dx%d %d-channel output" % (dic['pool'], name, dic['gpus'], dic['imgSize'], dic['imgSize'], dic['outputChannels'])
        return dic
    
class NormLayerParser(LayerWithInputParser):
    RESPONSE_NORM = 'response'
    CONTRAST_NORM = 'contrast'
    CROSSMAP_RESPONSE_NORM = 'cross-map response'
    
    def __init__(self, norm_type):
        LayerWithInputParser.__init__(self, num_inputs=1)
        self.norm_type = norm_type
        
    def add_params(self, mcp):
        LayerWithInputParser.add_params(self, mcp)
        dic, name = self.dic, self.dic['name']
        dic['scale'] = mcp.safe_get_float(name, 'scale')
        dic['scale'] /= dic['size'] if self.norm_type == self.CROSSMAP_RESPONSE_NORM else dic['size']**2
        dic['pow'] = mcp.safe_get_float(name, 'pow')
        dic['minDiv'] = mcp.safe_get_float(name, 'minDiv', default=1.0)
        
    def parse(self, name, mcp, prev_layers, model):
        dic = LayerWithInputParser.parse(self, name, mcp, prev_layers, model)
        dic['requiresParams'] = True
        dic['channels'] = mcp.safe_get_int(name, 'channels')
        dic['size'] = mcp.safe_get_int(name, 'size')
        dic['blocked'] = mcp.safe_get_bool(name, 'blocked', default=False)
        
        dic['imgPixels'] = dic['numInputs'][0] / dic['channels']
        dic['imgSize'] = int(n.sqrt(dic['imgPixels']))
        
        # Contrast normalization layer does not use its inputs
        dic['usesInputs'] = self.norm_type != self.CONTRAST_NORM
        
        self.verify_num_range(dic['channels'], 'channels', 1, None)
        if self.norm_type == self.CROSSMAP_RESPONSE_NORM: 
            self.verify_num_range(dic['size'], 'size', 2, dic['channels'])
            if dic['channels'] % 16 != 0:
                raise LayerParsingError("Layer '%s': number of channels must be divisible by 16 when using crossMap" % name)
        else:
            self.verify_num_range(dic['size'], 'size', 1, dic['imgSize'])
        
        if self.norm_type != self.CROSSMAP_RESPONSE_NORM and dic['channels'] > 3 and dic['channels'] % 4 != 0:
            raise LayerParsingError("Layer '%s': number of channels must be smaller than 4 or divisible by 4" % name)

        self.verify_img_size()

        dic['outputs'] = dic['imgPixels'] * dic['channels']
        print "Initialized %s-normalization layer '%s' on GPUs %s, producing %dx%d %d-channel output" % (self.norm_type, name, dic['gpus'], dic['imgSize'], dic['imgSize'], dic['channels'])
        return dic

class CostParser(LayerWithInputParser):
    def __init__(self, num_inputs=-1):
        LayerWithInputParser.__init__(self, num_inputs=num_inputs)
        
    def parse(self, name, mcp, prev_layers, model):
        dic = LayerWithInputParser.parse(self, name, mcp, prev_layers, model)
        dic['requiresParams'] = True
        # Stored as string because python can't pickle lambda functions
        dic['outputFilter'] = 'lambda costs,num_cases: [c/num_cases for c in costs]'
        dic['children'] = mcp.safe_get_list(name, 'children', default=[])
        # Aggregated costs only produce outputs which are additive.
        for c in dic['children']:
            if c not in prev_layers:
                raise LayerParsingError("Layer '%s': child cost layer '%s' not defined" % (name, c))
            if prev_layers[c]['type'] != dic['type']:
                raise LayerParsingError("Layer '%s': child cost layer '%s' must have same type as parent" % (name, c))
            prev_layers[c]['aggregated'] = 1
        dic['aggregated'] = dic['children'] != []
        del dic['neuron']
        return dic

    def add_params(self, mcp):
        LayerWithInputParser.add_params(self, mcp)
        dic, name = self.dic, self.dic['name']
        dic['coeff'] = mcp.safe_get_float(name, 'coeff')
        dic['gradConsumer'] = dic['coeff'] > 0
            
class CrossEntCostParser(CostParser):
    def __init__(self):
        CostParser.__init__(self, num_inputs=2)
        
    def parse(self, name, mcp, prev_layers, model):
        dic = CostParser.parse(self, name, mcp, prev_layers, model)
        if dic['numInputs'][0] != model.train_data_provider.get_num_classes(): # first input must be labels
            raise LayerParsingError("Layer '%s': Dimensionality of first input must be equal to number of labels" % name)
        if dic['inputLayers'][1]['type'] != 'softmax':
            raise LayerParsingError("Layer '%s': Second input must be softmax layer" % name)
        if dic['numInputs'][1] != model.train_data_provider.get_num_classes():
            raise LayerParsingError("Layer '%s': Softmax input '%s' must produce %d outputs, because that is the number of classes in the dataset" \
                                    % (name, dic['inputs'][1], model.train_data_provider.get_num_classes()))
        
        print "Initialized cross-entropy cost '%s' on GPUs %s" % (name, dic['gpus'])
        return dic
    
class LogregCostParser(CostParser):
    def __init__(self):
        CostParser.__init__(self, num_inputs=2)
        
    def add_params(self, mcp):
        CostParser.add_params(self, mcp)
        dic, name = self.dic, self.dic['name']
        dic['topk'] = mcp.safe_get_int(name, 'topk', default=1)
        if dic['topk'] > dic['numInputs'][1]:
            raise LayerParsingError("Layer '%s': parameter 'topk'must not have value greater than the number of classess."  % (name))
        
    def parse(self, name, mcp, prev_layers, model):
        dic = CostParser.parse(self, name, mcp, prev_layers, model)
        dic['requiresParams'] = True
        if dic['numInputs'][0] != 1: # first input must be labels
            raise LayerParsingError("Layer '%s': dimensionality of first input must be 1" % name)
        if dic['inputLayers'][1]['type'] != 'softmax':
            raise LayerParsingError("Layer '%s': second input must be softmax layer" % name)
        if dic['numInputs'][1] != model.train_data_provider.get_num_classes():
            raise LayerParsingError("Layer '%s': softmax input '%s' must produce %d outputs, because that is the number of classes in the dataset" \
                                    % (name, dic['inputs'][1], model.train_data_provider.get_num_classes()))
            
        print "Initialized logistic regression cost '%s' on GPUs %s" % (name, dic['gpus'])
        return dic
    
class BinomialCrossEntCostParser(CostParser):
    def __init__(self):
        CostParser.__init__(self, num_inputs=2)
        
    def add_params(self, mcp):
        CostParser.add_params(self, mcp)
        self.dic['posWeight'] = mcp.safe_get_float(self.dic['name'], 'posWeight', default=1.0)
        
    def parse(self, name, mcp, prev_layers, model):
        dic = CostParser.parse(self, name, mcp, prev_layers, model)

        if dic['numInputs'][0] != dic['numInputs'][1]:
            raise LayerParsingError("Layer '%s': both inputs must produce the same number of outputs" % (name))

        if 'neuron' not in dic['inputLayers'][1] or dic['inputLayers'][1]['neuron'] != 'logistic':
            print "WARNING: Layer '%s': input '%s' is not logistic, results may not be what you intend." % (dic['name'], dic['inputs'][1])
        
        if dic['type'] == 'cost.bce':
            print "Initialized binomial cross-entropy cost '%s' on GPUs %s" % (name, dic['gpus'])
        
        
        dic['computeSoftmaxErrorRate'] = True
        return dic
    
class DetectionCrossEntCostParser(BinomialCrossEntCostParser):
    def __init__(self):
        BinomialCrossEntCostParser.__init__(self)
        
    def parse(self, name, mcp, prev_layers, model):
        dic = BinomialCrossEntCostParser.parse(self, name, mcp, prev_layers, model)
        if dic['numInputs'][0] != model.train_data_provider.get_num_classes(): # first input must be labels
            raise LayerParsingError("Layer '%s': Dimensionality of first input must be equal to number of labels" % name)
        dic['computeSoftmaxErrorRate'] = False
        dic['outputFilter'] = 'lambda costs,num_cases: [c/num_cases for c in costs[:2]] + [(class_cost[2] / class_cost[j] if class_cost[j] > 0 else n.inf) for class_cost in [costs[2:][i*3:(i+1)*3] for i in range(len(costs[2:])/3)] for j in range(2)]'
        dic['outputFilterFormatter'] = 'lambda self,costs: "(crossent) %.6f, (err) %.6f, " % (costs[0], costs[1]) + ", ".join("(%s) %.6f, %.6f" % (self.train_data_provider.batch_meta["label_names"][i/2-1],costs[i],costs[i+1]) for i in xrange(2, len(costs), 2))'
        print "Initialized detection cross-entropy cost '%s' on GPUs %s" % (name, dic['gpus'])
        return dic
    
class SumOfSquaresCostParser(CostParser):
    def __init__(self):
        CostParser.__init__(self, num_inputs=1)
        
    def parse(self, name, mcp, prev_layers, model):
        dic = CostParser.parse(self, name, mcp, prev_layers, model)
        print "Initialized sum-of-squares cost '%s' on GPUs %s" % (name, dic['gpus'])
        return dic
    
# All the layer parsers
layer_parsers = {'data' :           lambda : DataLayerParser(),
                 'fc':              lambda : FCLayerParser(),
                 'sfc':             lambda : SplitFCLayerParser(),
                 'conv':            lambda : ConvLayerParser(),
                 'local':           lambda : LocalUnsharedLayerParser(),
                 'softmax':         lambda : SoftmaxLayerParser(),
                 'eltsum':          lambda : EltwiseSumLayerParser(),
                 'eltmax':          lambda : EltwiseMaxLayerParser(),
                 'sum':             lambda : SumLayerParser(),
                 'neuron':          lambda : NeuronLayerParser(),
                 'pool':            lambda : PoolLayerParser(),
                 'cmpool':          lambda : CrossMapPoolLayerParser(),
                 'rnorm':           lambda : NormLayerParser(NormLayerParser.RESPONSE_NORM),
                 'cnorm':           lambda : NormLayerParser(NormLayerParser.CONTRAST_NORM),
                 'cmrnorm':         lambda : NormLayerParser(NormLayerParser.CROSSMAP_RESPONSE_NORM),
                 'nailbed':         lambda : NailbedLayerParser(),
                 'blur':            lambda : GaussianBlurLayerParser(),
                 'href':            lambda : HorizontalReflectionLayerParser(),
                 'resize':          lambda : ResizeLayerParser(),
                 'rgb2yuv':         lambda : RGBToYUVLayerParser(),
                 'rgb2lab':         lambda : RGBToLABLayerParser(),
                 'rscale':          lambda : RandomScaleLayerParser(),
                 'crop':            lambda : CropLayerParser(),
                 'concat':          lambda : ConcatentionLayerParser(),
                 'pass':            lambda : PassThroughLayerParser(),
                 'dropout':         lambda : DropoutLayerParser(),
                 'dropout2':        lambda : Dropout2LayerParser(),
                 'cost.logreg':     lambda : LogregCostParser(),
                 'cost.crossent':   lambda : CrossEntCostParser(),
                 'cost.bce':        lambda : BinomialCrossEntCostParser(),
                 'cost.dce':        lambda : DetectionCrossEntCostParser(),
                 'cost.sum2':       lambda : SumOfSquaresCostParser()}
 
# All the neuron parsers
# This isn't a name --> parser mapping as the layer parsers above because neurons don't have fixed names.
# A user may write tanh[0.5,0.25], etc.
neuron_parsers = sorted([NeuronParser('ident', 'f(x) = x', uses_acts=False, uses_inputs=False),
                         NeuronParser('logistic', 'f(x) = 1 / (1 + e^-x)', uses_acts=True, uses_inputs=False),
                         NeuronParser('abs', 'f(x) = |x|', uses_acts=False, uses_inputs=True),
                         NeuronParser('relu', 'f(x) = max(0, x)', uses_acts=True, uses_inputs=False),
                         NeuronParser('nrelu', 'f(x) = max(0, x) + noise', uses_acts=True, uses_inputs=False),
                         NeuronParser('softrelu', 'f(x) = log(1 + e^x)', uses_acts=True, uses_inputs=False),
                         NeuronParser('square', 'f(x) = x^2', uses_acts=False, uses_inputs=True),
                         NeuronParser('sqrt', 'f(x) = sqrt(x)', uses_acts=True, uses_inputs=False),
                         ParamNeuronParser('log[a]', 'f(x) = log(a + x)', uses_acts=False, uses_inputs=True),
                         ParamNeuronParser('tanh[a,b]', 'f(x) = a * tanh(b * x)', uses_acts=True, uses_inputs=False),
                         ParamNeuronParser('brelu[a]', 'f(x) = min(a, max(0, x))', uses_acts=True, uses_inputs=False),
                         ParamNeuronParser('linear[a,b]', 'f(x) = a * x + b', uses_acts=True, uses_inputs=False),
                         ParamNeuronParser('drelu[a]', 'f(x) = x - a * tanh(x / a)', uses_acts=False, uses_inputs=True)],
                        key=lambda x:x.type)

# Learning rate schedules
lrs_parsers = sorted([ParamParser('const[fbase]'),
                      ParamParser('linear[fbase;ftgtFactor]'),
                      ParamParser('exp[fbase;ftgtFactor]'),
                      ParamParser('dexp[fbase;ftgtFactor;inumSteps]')])
