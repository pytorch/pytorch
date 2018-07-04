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

#ifndef ACTBROADCASTER_CUH_H_
#define ACTBROADCASTER_CUH_H_

#include <map>
#include "streambroadcast.cuh"
#include "copypipeline.cuh"

class BroadcastMessage {
public:
    enum MESSAGE_TYPE {
        BROADCAST,
        EXIT
    };
protected:
    int _srcDevice;
    std::map<int, NVMatrix*> _mats;
    int _userIdx;
    Queue<int>* _finishQueue;
    MESSAGE_TYPE _type;
    BroadcastMessage(MESSAGE_TYPE type);
public:
    BroadcastMessage(std::map<int, NVMatrix*> mats, int srcDevice, int userIdx, Queue<int>& finishQueue);

    int getSrcDevice();
    std::map<int, NVMatrix*>& getMatrices();
    int getUserIdx();
    Queue<int>& getFinishQueue();
    MESSAGE_TYPE getMessageType();
};

class ExitBroadcastMessage : public BroadcastMessage {
public:
    ExitBroadcastMessage();
};

class ActBroadcaster : public Thread {
protected:
    std::map<int,IBroadcastNetwork*> _broadcasters; // src device --> broadcaster
    Queue<BroadcastMessage*> _messageQueue;
    int _numUsers;
public:
    ActBroadcaster(int numUsers, intv& cpus);
    ~ActBroadcaster();
    Queue<BroadcastMessage*>& getMessageQueue();
    virtual void* run();
    void stop();
};


#endif /* ACTBROADCASTER_CUH_H_ */
