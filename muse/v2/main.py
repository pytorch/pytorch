import os
import io
import sys
import math
import torch
import shutil
import threading
import subprocess
import torch.nn as nn
from quant_layer import QuantLayer

in_q  = 7
layer_cnts = 0

class makenet():
    def __init__(self, filename):
        self.bns = [""]
        self.layer = []
        self.convs = []
        self.pools = []
        self.qulist= []
        self.fcinit= []
        self.fcford = []
        self.avginit = []
        self.avgford = []
        self.classname = ""
        self.fvggnet = open("vggnet.py", "a")
        self.fmakenet = open("makenet.py", "a")

        self.make_config(filename)
        self.make_block(filename)
        self.make_class(filename)
        self.make_layers(filename)
        self.make_padding(filename)
        self.make_forward(filename)
        self.make_weight(filename)
        self.make_main(filename)

        self.fmakenet.close()
        os.system("python3 makenet.py > layerinfo")
        self.splicing_layers("layerinfo")
        self.bns[0] = self.get_op_code(filename, "bn")
        self.get_op_code(filename, "avgpool")
        self.get_op_code(filename, "weight")
        self.get_op_code(filename, "quant")
        self.get_op_code(filename, "fc")

        self._make_head()
        self._make_init()
        self._make_padding(filename)
        self._make_forward()
        self._make_avgpool()
        self._make_tail()
        self.fvggnet.close()
        print("makenet success")

    def make_config(self, filename):
        with open(filename, 'r') as file:
            while True:
                line = file.readline()
                if line.startswith("#") is False:
                    self.fmakenet.write(line)
                if line.strip() == "}":
                    break
        self.fmakenet.write('\n')

    def make_block(self, filename):
        read_flag = 0
        with open(filename, 'r') as file:
            while True:
                line = file.readline()
                if line.startswith("#") is False:
                    if "class BasicBlock" not in line and read_flag == 0:
                        continue
                    self.classname = "BasicBlock"
                    read_flag = 1
                    if line.startswith("class ") and "BasicBlock" not in line:
                        break
                    self.fmakenet.write(line)

    def make_class(self, filename):
        read_flag = 0
        with open(filename, 'r') as file:
            while True:
                line = file.readline()
                if line.strip().startswith("#") is False:
                    if "class vgg" not in line and read_flag == 0:
                        continue
                    read_flag = 1
                    if line.strip().startswith("def ") and "__init__" not in line:
                        break
                    self.fmakenet.write(line)

    def make_layers(self, filename):
        read_flag = 0
        with open(filename, 'r') as file:
            while True:
                line = file.readline()
                if line.strip().startswith("#") is False:
                    if "def make_layers" not in line and read_flag == 0:
                        continue
                    read_flag = 1
                    if line.strip().startswith("def ") and "make_layers" not in line:
                        break
                    self.fmakenet.write(line)

    def make_padding(self, filename):
        read_flag = 0
        with open(filename, 'r') as file:
            while True:
                line = file.readline()
                if line.strip().startswith("#") is False:
                    if "def padding" not in line and read_flag == 0:
                        continue
                    read_flag = 1
                    if line.strip().startswith("def ") and "padding" not in line:
                        break
                    self.fmakenet.write(line)

    def make_forward(self, filename):
        read_flag = 0
        with open(filename, 'r') as file:
            while True:
                line = file.readline()
                if line.strip().startswith("#") is False:
                    if line.strip().startswith("class "):
                        self.classname = line.split(' ', 1)[1].split('(', 1)[0]
                    if self.classname != "vgg":
                        continue
                    if "def forward" not in line and read_flag == 0:
                        continue
                    read_flag = 1
                    if line.strip().startswith("def ") and "forward" not in line:
                        break
                    self.fmakenet.write(line)

    def make_weight(self, filename):
        read_flag = 0
        with open(filename, 'r') as file:
            while True:
                line = file.readline()
                if line.strip().startswith("#") is False:
                    if "def _initialize_weights" not in line and read_flag == 0:
                        continue
                    read_flag = 1
                    # if "= vgg(" and "_initialize_weights" not in line:
                    if "= vgg(" in line:
                        break
                    self.fmakenet.write(line)

    def make_main(self, filename):
        read_flag = 0
        with open(filename, 'r') as file:
            for line in file:
                if line.strip().startswith("#") is False and line.strip().startswith("print") is False:
                    if "= vgg(" not in line and read_flag == 0:
                        continue
                    read_flag = 1
                    self.fmakenet.write(line)
                if "= vgg(" in line:
                    print_vgg = "{}{}{}".format("print(\"vgg_module = \", ", line.strip().split(" ", 1)[0], ")\n")
                    self.fmakenet.write(print_vgg)
                    return

    def _make_head(self):
        self.fvggnet.write("import torch\n")
        self.fvggnet.write("import torch.nn as nn\n")
        self.fvggnet.write("import torch.nn.functional as F\n")
        self.fvggnet.write("from quant_layer import QuantLayer\n\n")
        self.fvggnet.write("class Net(nn.Module):\n")
        self.fvggnet.write("    def __init__(self, num_classes=10):\n")
        self.fvggnet.write("        super(Net, self).__init__()\n")

    def _make_tail(self):
        self.fvggnet.write("\nn = Net()\n")
        self.fvggnet.write("example_input = torch.rand(1, 3, 224, 224)\n")
        self.fvggnet.write("module = torch.jit.trace(n, example_input)\n")
        self.fvggnet.write("module._c._fun_compile()\n")

    def _make_init(self):
        for i in range(len(self.layer)):
            if "Conv2d" in str(self.layer[i]):
                self.convs.append(self.layer[i])
            elif "BatchNorm2d" in str(self.layer[i]):
                if self.bns[0] == "True":
                    self.bns.append(self.layer[i])
            elif "MaxPool2d" in str(self.layer[i]):
                self.pools.append(self.layer[i])

        for i in range(len(self.pools)):
            self.fvggnet.write("{}{}{}{}{}".format("        self.pool", i+1, " = nn.",
                                             self.pools[i].split(":")[1].strip(), "\n"))
        for j in range(len(self.convs)):
            self.fvggnet.write("{}{}{}{}{}".format("        self.conv", j+1, " = nn.",
                                             self.convs[j].split(":")[1].strip(), "\n"))
            if self.bns[0] == "True":
                self.fvggnet.write("{}{}{}{}{}".format("        self.bn", j+1, " = nn.",
                                                 self.bns[j+1].split(":")[1].strip(), "\n"))
        self.fvggnet.write("\n")
        for i in range(len(self.avginit)):
            self.fvggnet.write("{}{}{}".format("        ", self.avginit[i], "\n"))
        for i in range(len(self.qulist)):
            if "= QuantLayer()" in self.qulist[i]:
                self.fvggnet.write("{}{}{}".format("        ", self.qulist[i], "\n"))
        for i in range(len(self.fcinit)):
            self.fvggnet.write("{}{}{}".format("        ", self.fcinit[i], "\n"))

    def _make_padding(self, filename):
        self.fvggnet.write("\n")
        read_flag = 0
        with open(filename, 'r') as file:
            while True:
                line = file.readline()
                if line.strip().startswith("#") is False:
                    if "def padding" not in line and read_flag == 0:
                        continue
                    read_flag = 1
                    if line.strip().startswith("def ") and "padding" not in line:
                        break
                    self.fvggnet.write(line)

    def _make_convlay(self, convcnt):
        x = "F.relu"
        end = "))\n"
        if self.bns[0] == "True":
            x = "F.relu(self.bn"
            end = ")))\n"
        convcmd = "{}{}{}{}{}{}{}".format("        x = ", x, str(convcnt),
                                          "(self.conv",str(convcnt), "(x", end)
        self.fvggnet.write(convcmd)

    def _make_forward(self):
        self.fvggnet.write("    def forward(self, x):\n")
        convcnt = poolcnt = 0

        for i in range(len(self.layer)):
            if "Conv2d" in str(self.layer[i]):
                convcnt += 1
                if self.bns[0] == "True":
                    if "BatchNorm2d" in str(self.layer[i+1]):
                        if "ReLU" in str(self.layer[i+2]):
                             self._make_convlay(convcnt)
                else:
                    if "ReLU" in str(self.layer[i+1]):
                        self._make_convlay(convcnt)
            elif "MaxPool2d" in str(self.layer[i]):
                poolcnt += 1
                poolcmd = "{}{}{}".format("        x = self.pool", str(poolcnt), "(x)\n")
                self.fvggnet.write(poolcmd)

    def _make_avgpool(self):
        self.fvggnet.write("{}{}".format("        x = self.padding(x)", "\n"))
        for i in range(len(self.avgford)):
            self.fvggnet.write("{}{}{}".format("        ", self.avgford[i], "\n"))
            self.fvggnet.write("{}{}{}".format("        ", self.qulist[i+int(len(self.qulist)/2)], "\n"))
        for i in range(len(self.fcford)):
            self.fvggnet.write("{}{}{}".format("        ", self.fcford[i], "\n"))

        self.fvggnet.write("        return x\n")

    def splicing_layers(self, file):
        with open(file, "r") as f:
            lines = f.readline()
            while lines:
                lines = f.readline()
                self.layer.append(lines)

    def get_op_code(self, code_path, operator):
        with open(code_path, encoding='utf-8') as f:
            lines = f.readline()
            while lines:
                lines = f.readline()
                if operator == "bn":
                    if "self.layers = self.make_layers" in lines:
                        lines = lines.strip()
                        if lines.startswith("#") is False:
                            return lines.rsplit(" ", 3)[3][:-1]
                elif operator == "avgpool":
                    if "= nn.AvgPool2d(" in lines:
                        lines = lines.strip()
                        if lines.startswith("#") is False:
                            self.avginit.append(lines)
                    elif "x = self.avgpool" in lines:
                        lines = lines.strip()
                        if lines.startswith("#") is False:
                            self.avgford.append(lines)
                elif operator == "quant":
                    if "self.quant_avg" in lines:
                        lines = lines.strip()
                        if lines.startswith("#") is False:
                            self.qulist.append(lines)
                elif operator == "fc":
                    if "self.quant_fc1 =" in lines:
                        lines = lines.strip()
                        if lines.startswith("#") is False:
                            self.fcinit.append(lines)
                            while lines.strip() != ")":
                                lines = f.readline()
                                lines = lines.strip()
                                if lines.startswith("#") is False:
                                    self.fcinit.append(lines)
                    elif "= x.view(" in lines or "= self.classifier(" in lines:
                        lines = lines.strip()
                        if lines.startswith("#") is False:
                            self.fcford.append(lines)

def get_layer_info(path, flag):
    layer_cnt = 0
    layer_info = []
    find_flg = flag
    if int(str(find_flg).split(":", 1)[1]) == 0:
        print("please do not send layer_num:0")
        exit(0)
    next_flg = "{}{}".format("layer_num:", int(str(find_flg).split(":", 1)[1]) + 1)
    with open(path, 'r') as file_read:
        for line in file_read:
            if flag in line:
                layer_cnt += 1
            if next_flg in line:
                break
            if "compiler_end" in line:
                break
            if layer_cnt == 1:
                layer_info.append(line)
    return layer_info


def getoneDimList(newlist):
    oneDimList = []
    for element in newlist:
        if not isinstance(element, list):
            oneDimList.append(element)
        else:
            oneDimList.extend(getoneDimList(element))
    return oneDimList

def write_pt_data(filename, filedata):
    path = "{}{}{}".format("./output/", filename, ".txt")
    with open(path, 'w') as fw:
        if "num_batches_tracked" in filename or "quant_" in filename or ".alpha" in filename:
            fw.write(str(filedata))
        else:
            conver = filedata.tolist()
            if filedata.dim() == 0:
                fw.write(str(conver))
                return
            if filedata.dim() > 1:
                conver = getoneDimList(conver)
            for i in range(len(conver)):
                if "bn.bn" in filename:
                    fw.write(str(conver[i]))
                else:
                    fw.write(str(conver[i]))
                fw.write('\n')

    print("%s write data success" % path)

def write_layer_config(layermsg):
    layername = layermsg[0].split(":", 1)[1].split(" ", 1)[0].strip('\n')
    if layername == "0":
        path = "{}{}".format("./output/config", ".txt")
    else:
        path = "{}{}{}".format("./output/config_", layername, ".txt")
    with open(path, 'w') as fw:
        for i in range(len(layermsg)):
            fw.write(layermsg[i])
            fw.write('\n')
    print("%s write config success" % path)

def deal_out_in_q(data, layer_msg):
    bits = 7
    global in_q
    if "pool" in layer_msg:
        out_q = in_q
    else:
        out_q = bits - math.ceil(math.log2(0.5*data))
    layer_msg.append("{}{}".format("in_q = ", str(in_q)))
    layer_msg.append("{}{}".format("out_q = ", str(out_q)))
    in_q = out_q
    write_config = threading.Thread(target=write_layer_config, args=(layer_msg, ))
    write_config.start()
    write_config.join()

def splicing_output(num, flag, layerlist, layer_msg):
    for i in range(len(layerlist)):
        if "quant.alpha" in layerlist[i][0]:
            data = layerlist[i][1]
            if num == 0:
                deal_out_in_q(data, layer_msg)
                return

    for i in range(num):
        name = layerlist[flag+i][0]
        data = layerlist[flag+i][1]
        if "bn.running_mean" in name or "bn.running_var" in name \
              or "bn.weight" in name or "bn.bias" in name:
            continue
        elif "quant.alpha" in name:
            deal_out_in_q(data, layer_msg)
            continue
        write_data = threading.Thread(target=write_pt_data, args=(name, data))
        write_data.start()
        write_data.join()

def get_layercnt(filename):
    with open(filename, 'r') as file:
        while True:
            line = file.readline()
            if line.strip() == "":
                break
            elif ": BasicBlock(" in line or "Pool2d(" in line or "Linear" in line:
                global layer_cnts
                layer_cnts += 1

def load_pt(pt_path):
    name_list    = []
    data_list    = []
    quant_list   = []
    onelayer_cnt = []

    get_layercnt("layerinfo")
    with open(pt_path, 'rb') as f:
        buffer = io.BytesIO(f.read())
        dict = torch.load(buffer, map_location=torch.device('cpu'))
        for k, v in dict.items():
           if "quant_" in k or "classifier." in k:
               quant_list.append(k)
               quant_list.append(v)
           name_list.append(k)
           data_list.append(v)
           a = v

    cnt    = 0
    layers = [["", a]]

    for i in range(layer_cnts):
        layer = "{}{}{}".format("layers.", i, ".")
        for j in range(len(name_list)):
            if layer in name_list[j]:
                 layers.append([name_list[j], data_list[j]])
                 cnt += 1
        onelayer_cnt.append(str(cnt))
        cnt = 0

    del(layers[0])
    logpath = "{}{}".format(os.getcwd(), "/vggnet.log")
    for i in range(layer_cnts):
        layername = "{}{}".format("layer_num:", str(i+1))
        layer_msg = get_layer_info(logpath, layername)
        splicing_output(int(onelayer_cnt[i]), cnt, layers, layer_msg)
        cnt += int(onelayer_cnt[i])
        
    for i in range(int(len(quant_list))):
        tmpstr = str(quant_list[i])
        if "quant_" in tmpstr or "classifier" in tmpstr:
            write_data = threading.Thread(target=write_pt_data, 
                         args=(quant_list[i], quant_list[i+1]))
            write_data.start()
            write_data.join()

def checkfile(filelist):
    for i in range(len(filelist)):
        if not os.path.exists(filelist[i]):
            print("%s not find in %s directory..." %(filelist[i], os.getcwd()))

def cleanup(cleanlist):
    for i in range(len(cleanlist)):
        if os.path.exists(cleanlist[i]):
            if os.path.isdir(cleanlist[i]):
                shutil.rmtree(cleanlist[i])
            else:
                os.remove(cleanlist[i])
            print("%s in %s directory clean success" %(cleanlist[i], os.getcwd()))
        

if __name__ == '__main__':
    fileslist = ["vgg_imagenet.py", "vgg_imagenet2.pt", "quant_layer.py"]
    cleanlist = ["output", "makenet.py", "vggnet.py", "vggnet.log", "layerinfo"]

    if (len(sys.argv) != 2) or not os.path.exists(sys.argv[1]):
        pt_path = "./vgg_imagenet2.pt"
    else:
        pt_path = sys.argv[1]

    cleanup(cleanlist)
    checkfile(fileslist)
    makenet("vgg_imagenet.py")
    os.system("python3 vggnet.py > vggnet.log")
    os.system("mkdir output")
    load_pt(pt_path)
