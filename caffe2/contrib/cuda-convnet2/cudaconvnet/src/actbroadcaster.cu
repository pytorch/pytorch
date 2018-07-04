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
#include "../include/actbroadcaster.cuh"

using namespace std;

/*
 * =====================
 * BroadcastMessage
 * =====================
 */
BroadcastMessage::BroadcastMessage(map<int, NVMatrix*> mats, int srcDevice, int userIdx, Queue<int>& finishQueue)
    : _type(BROADCAST), _mats(mats), _srcDevice(srcDevice), _userIdx(userIdx), _finishQueue(&finishQueue) {
}

BroadcastMessage::BroadcastMessage(MESSAGE_TYPE type)
    : _type(type), _finishQueue(NULL) {
}

int BroadcastMessage::getSrcDevice() {
    return _srcDevice;
}

map<int, NVMatrix*>& BroadcastMessage::getMatrices() {
    return _mats;
}

int BroadcastMessage::getUserIdx() {
    return _userIdx;
}

Queue<int>& BroadcastMessage::getFinishQueue() {
    return *_finishQueue;
}

BroadcastMessage::MESSAGE_TYPE BroadcastMessage::getMessageType() {
    return _type;
}

/*
 * =====================
 * ExitBroadcastMessage
 * =====================
 */
ExitBroadcastMessage::ExitBroadcastMessage() : BroadcastMessage(BroadcastMessage::EXIT) {
}

/*
 * =====================
 * ActBroadcaster
 * =====================
 */
ActBroadcaster::ActBroadcaster(int numUsers, intv& cpus) : Thread(true, cpus), _numUsers(numUsers) {
}

ActBroadcaster::~ActBroadcaster() {
    for (map<int,IBroadcastNetwork*>::const_iterator it = _broadcasters.begin(); it != _broadcasters.end(); ++it) {
        delete it->second;
    }
}

Queue<BroadcastMessage*>& ActBroadcaster::getMessageQueue() {
    return _messageQueue;
}

void* ActBroadcaster::run() {
    int nextUserIdx = 0;
    bool exit = false;
    while (!exit) {
        BroadcastMessage& msg = *_messageQueue.dequeue();
        if (msg.getMessageType() == BroadcastMessage::EXIT) {
            exit = true;
            delete &msg;
        } else {
            if (msg.getUserIdx() == nextUserIdx) {
                if (_broadcasters.count(msg.getSrcDevice()) == 0) {
                    _broadcasters[msg.getSrcDevice()] = &IBroadcastNetwork::make(getKeys(msg.getMatrices()), msg.getSrcDevice());
                }
                _broadcasters[msg.getSrcDevice()]->broadcast(msg.getMatrices());
                msg.getFinishQueue().enqueue(0);
                delete &msg;
                nextUserIdx = (nextUserIdx + 1) % _numUsers;
            } else {
                _messageQueue.enqueue(&msg);
            }
        }
    }
    return NULL;
}

void ActBroadcaster::stop() {
    getMessageQueue().enqueue(new ExitBroadcastMessage());
    join();
}
