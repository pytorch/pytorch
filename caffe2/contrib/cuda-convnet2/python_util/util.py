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

import re
import cPickle
import os
from cStringIO import StringIO

class UnpickleError(Exception):
    pass

GPU_LOCK_NO_SCRIPT = -2
GPU_LOCK_NO_LOCK = -1

def pickle(filename, data):
    fo = filename
    if type(filename) == str:
        fo = open(filename, "w")
    
    cPickle.dump(data, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    fo.close()
    
def unpickle(filename):
    if not os.path.exists(filename):
        raise UnpickleError("Path '%s' does not exist." % filename)

    fo = open(filename, 'r')
    z = StringIO()
    file_size = os.fstat(fo.fileno()).st_size
    # Read 1GB at a time to avoid overflow
    while fo.tell() < file_size:
        z.write(fo.read(1 << 30))
    fo.close()
    dict = cPickle.loads(z.getvalue())
    z.close()
    
    return dict

def is_intel_machine():
    VENDOR_ID_REGEX = re.compile('^vendor_id\s+: (\S+)')
    f = open('/proc/cpuinfo')
    for line in f:
        m = VENDOR_ID_REGEX.match(line)
        if m:
            f.close()
            return m.group(1) == 'GenuineIntel'
    f.close()
    return False

# Returns the CPUs associated with a given GPU
def get_cpus_for_gpu(gpu):
    #proc = subprocess.Popen(['nvidia-smi', '-q', '-i', str(gpu)], stdout=subprocess.PIPE)
    #lines = proc.communicate()[0]
    #lines = subprocess.check_output(['nvidia-smi', '-q', '-i', str(gpu)]).split(os.linesep)

    with open('/proc/driver/nvidia/gpus/%d/information' % gpu) as f:
        for line in f:
            if line.startswith('Bus Location'):
                bus_id = line.split(':', 1)[1].strip()
                bus_id = bus_id[:7] + ':' + bus_id[8:]
                ff = open('/sys/module/nvidia/drivers/pci:nvidia/%s/local_cpulist' % bus_id)
                cpus_str = ff.readline()
                ff.close()
                cpus = [cpu for s in cpus_str.split(',') for cpu in range(int(s.split('-')[0]),int(s.split('-')[1])+1)]
                return cpus
    return [-1]

def get_cpu():
    if is_intel_machine():
        return 'intel'
    return 'amd'

def is_windows_machine():
    return os.name == 'nt'
    
def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    return [tryint(c) for c in re.split('([0-9]+)', s)]
