/*
 * Copyright 2014 Google Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MESSAGES_CUH_
#define MESSAGES_CUH_

#include <string>
#include "layer.cuh"

class Layer;

enum MESSAGES { FPROP_TERMINAL,
                BPROP_TERMINAL,
                BPROP_READY,
                FPROP_READY,
                SYNC,
                COPY_TO_CPU,
                COPY_TO_GPU,
                UPDATE_WEIGHTS,
                CONSTRAIN_WEIGHTS,
                RESET,
                RESET_PASS_IDX,
                COST_COMPUTED,
                BPROP_START,
                EXIT_CONVNET};

class Message {
protected:
    MESSAGES _messageType;
public:
    MESSAGES getType() {
        return _messageType;
    }
    virtual Message* clone() {
        return new Message(_messageType);
    }
    Message(MESSAGES messageType) : _messageType(messageType) {
    }
    virtual ~Message() {
    }
};

class PropMessage : public Message {
protected:
    Layer *_toLayer;
    PASS_TYPE _passType;
    int _passIdx;
public:

    Layer& getToLayer() {
        return *_toLayer;
    }

    PASS_TYPE getPassType() {
        return _passType;
    }

    int getPassIdx() {
        return _passIdx;
    }

    virtual PropMessage* clone() {
        return new PropMessage(*_toLayer, _passType, _passIdx, _messageType);
    }

    PropMessage(Layer& toLayer, PASS_TYPE passType, int passIdx, MESSAGES msgType)
        : _toLayer(&toLayer), _passType(passType), _passIdx(passIdx), Message(msgType) {
    }
};

class FpropMessage : public PropMessage {
public:
    FpropMessage(Layer& toLayer, PASS_TYPE passType, int passIdx)
        : PropMessage(toLayer, passType, passIdx, FPROP_READY) {
    }
    virtual FpropMessage* clone() {
        return new FpropMessage(*_toLayer, _passType, _passIdx);
    }
};

class BpropMessage : public PropMessage {
public:
    BpropMessage(Layer& toLayer, PASS_TYPE passType, int passIdx)
        : PropMessage(toLayer, passType, passIdx, BPROP_READY) {
    }
    virtual BpropMessage* clone() {
        return new BpropMessage(*_toLayer, _passType, _passIdx);
    }
};

class BpropStartMessage : public Message {
protected:
    PASS_TYPE _passType;
    int _passIdx;
public:
    PASS_TYPE getPassType() {
        return _passType;
    }

    int getPassIdx() {
        return _passIdx;
    }

    virtual BpropStartMessage* clone() {
        return new BpropStartMessage(_passType, _passIdx);
    }

    BpropStartMessage(PASS_TYPE passType, int passIdx)
        : _passType(passType), Message(BPROP_START), _passIdx(passIdx) {
    }
};



#endif /* MESSAGES_CUH_ */
