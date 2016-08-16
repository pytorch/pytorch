import random
import math
import torch
from .Module import Module

# TODO fix THNN...


class SpatialConvolutionMap(Module):

    class maps(object):

        @staticmethod
        def full(nin, nout):
            ft = torch.Tensor(nin * nout, 2)
            p = 0
            for j in range(nout):
                for i in range(nin):
                    ft[p][0] = i
                    ft[p][1] = j
                    p += 1
            return ft

        @staticmethod
        def oneToOne(nfeat):
            ft = torch.Tensor(nfeat, 2)
            for i in range(nfeat):
                ft[i][0] = i
                ft[i][1] = i
            return ft

        @staticmethod
        def random(nin, nout, nto):
            nker = nto * nout
            tbl = torch.Tensor(nker, 2)
            fi = torch.randperm(nin)
            frcntr = 0
            nfi = math.floor(nin / nto)  # number of distinct nto chunks
            totbl = tbl.select(1, 1)
            frtbl = tbl.select(1, 0)
            fitbl = fi.narrow(0, 0, (nfi * nto))  # part of fi that covers distinct chunks
            ufrtbl = frtbl.unfold(0, nto, nto)
            utotbl = totbl.unfold(0, nto, nto)
            ufitbl = fitbl.unfold(0, nto, nto)

            # start fill_ing frtbl
            for i in range(nout):  # fro each unit in target map
                ufrtbl.select(0, i).copy_(ufitbl.select(0, frcntr))
                frcntr += 1
                if frcntr - 1 == nfi:  # reset fi
                    fi.copy_(torch.randperm(nin))
                    frcntr = 1

            for tocntr in range(utotbl.size(0)):
                utotbl.select(0, tocntr).fill_(tocntr)

            return tbl

    def __init__(self, conMatrix, kW, kH, dW=1, dH=1):
        super(SpatialConvolutionMap, self).__init__()

        self.kW = kW
        self.kH = kH
        self.dW = dW
        self.dH = dH
        self.connTable = conMatrix
        self.nInputPlane = int(self.connTable.select(1, 0).max()) + 1
        self.nOutputPlane = int(self.connTable.select(1, 1).max()) + 1
        self.weight = torch.Tensor(self.connTable.size(0), kH, kW)
        self.bias = torch.Tensor(self.nOutputPlane)
        self.gradWeight = torch.Tensor(self.connTable.size(0), kH, kW)
        self.gradBias = torch.Tensor(self.nOutputPlane)

        self.reset()

    def reset(self, stdv=None):
        if stdv is not None:
            stdv = stdv * math.sqrt(3)
            self.weight.uniform_(-stdv, stdv)
            self.bias.uniform_(-stdv, stdv)
        else:
            ninp = torch.Tensor(self.nOutputPlane).zero_()
            for i in range(self.connTable.size(0)):
                idx = int(self.connTable[i, 1])
                ninp[idx] += 1
            for k in range(self.connTable.size(0)):
                idx = int(self.connTable[k, 1])
                stdv = 1. / math.sqrt(self.kW * self.kH * ninp[idx])
                self.weight.select(0, k).uniform_(-stdv, stdv)
            for k in range(self.bias.size(0)):
                stdv = 1. / math.sqrt(self.kW * self.kH * ninp[k])
                # TODO: torch.uniform
                self.bias[k] = random.uniform(-stdv, stdv)

    def updateOutput(self, input):
        self._backend.SpatialConvolutionMap_updateOutput(
            self._backend.library_state,
            input,
            self.output,
            self.weight,
            self.bias,
            self.connTable,
            self.nInputPlane,
            self.nOutputPlane,
            self.dW, self.dH
        )
        return self.output

    def updateGradInput(self, input, gradOutput):
        self._backend.SpatialConvolutionMap_updateGradInput(
            self._backend.library_state,
            input,
            gradOutput,
            self.gradInput,
            self.weight,
            self.bias,
            self.connTable,
            self.nInputPlane,
            self.nOutputPlane,
            self.dW, self.dH
        )
        return self.gradInput

    def accGradParameters(self, input, gradOutput, scale=1):
        self._backend.SpatialConvolutionMap_accGradParameters(
            self._backend.library_state,
            input,
            gradOutput,
            self.gradWeight,
            self.gradBias,
            self.connTable,
            self.nInputPlane,
            self.nOutputPlane,
            self.dW, self.dH,
            scale
        )
