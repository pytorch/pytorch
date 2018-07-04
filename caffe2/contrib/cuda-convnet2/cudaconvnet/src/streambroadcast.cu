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

#include "../include/streambroadcast.cuh"

using namespace std;

/*
 * =====================
 * StreamBroadcast
 * =====================
 */

StreamBroadcast::StreamBroadcast(map<int,cudaStream_t>& streams) {
    _streams = streams;
}

StreamBroadcast::StreamBroadcast() {
}

void StreamBroadcast::toHostMem(NVMatrix& src, NVMatrix& hostmem, int srcDevice) {
    src.copy(hostmem, _streams[srcDevice]);
}

void StreamBroadcast::toTarget(NVMatrix& hostmem, NVMatrix& tgt, int tgtDevice, float scaleTarget, float scaleOutput) {
    tgt.add(hostmem, scaleTarget, scaleOutput, tgt, _streams[tgtDevice]);
}

void StreamBroadcast::init(map<int, NVMatrix*>& mats) {
    for (map<int, NVMatrix*>::const_iterator it = mats.begin(); it != mats.end(); ++it) {
        if (_streams.count(it->first) == 0) {
            _ownedStreams.insert(it->first);
            NVMatrix::setDeviceID(it->first);
            checkCudaErrors(cudaStreamCreateWithFlags(&_streams[it->first], cudaStreamNonBlocking));
        }
    }
}

StreamBroadcast::~StreamBroadcast() {
    for (set<int>::const_iterator it = _ownedStreams.begin(); it != _ownedStreams.end(); ++it) {
        checkCudaErrors(cudaStreamDestroy(_streams[*it]));
    }
}

cudaStream_t StreamBroadcast::getStream(int deviceID) {
    return _streams[deviceID];
}

// Sync stream associated with given device id
void StreamBroadcast::sync(int deviceID) {
    NVMatrix::syncStream(_streams[deviceID]);
}

void StreamBroadcast::transfer(map<int,NVMatrix*>& mats,  int srcDevice) {
    transfer(mats, _hostMem, srcDevice, 0, 1);
}

void StreamBroadcast::transfer(map<int,NVMatrix*>& mats,  int srcDevice, float scaleTarget, float scaleOutput) {
    transfer(mats, _hostMem, srcDevice, scaleTarget, scaleOutput);
}

void StreamBroadcast::transfer(map<int,NVMatrix*>& mats, HostNVMatrix& hostbuf, int srcDevice, float scaleTarget, float scaleOutput) {
    int oldDeviceID = NVMatrix::getDeviceID();
    assert(mats.count(srcDevice) != 0);
    init(mats);
//    assert(_streams.count(srcDevice) != 0);
    if (mats.size() > 1) {
        if (mats[srcDevice]->getNumElements() == 0) {
            for (map<int,NVMatrix*>::const_iterator it = mats.begin(); it != mats.end(); ++it) {
                it->second->resize(*mats[srcDevice]);
            }
        } else {
            int tgtDevice = mats.begin()->first != srcDevice ? mats.begin()->first : (++mats.begin())->first;
            // This case is a simple copy
            if (mats.size() == 2 && NVMatrix::canAccessPeer(tgtDevice, srcDevice)) {
                NVMatrix::setDeviceID(tgtDevice);
                mats[tgtDevice]->add(*mats[srcDevice], scaleTarget, scaleOutput, *mats[tgtDevice], _streams[tgtDevice]);
            } else {
                NVMatrix& src = *mats[srcDevice];
                if (hostbuf.getNumElements() < src.getNumElements()) {
                    hostbuf.resize(1,src.getNumElements());
                }
                hostbuf.setTrans(src.isTrans());

                NVMatrix& hostmat = hostbuf.sliceCols(0, src.getNumElements());
                assert(hostmat.isView());
                hostmat.reshape(src.getNumRows(), src.getNumCols());

                for (map<int,NVMatrix*>::const_iterator it = mats.begin(); it != mats.end(); ++it) {
                    assert(it->second->isContiguous());
                    NVMatrix::setDeviceID(it->first);
                    it->second->resize(src);
                    assert(it->second->isTrans() == src.isTrans());
                }
                int numChunks = min(DIVUP(src.getNumElements(), SB_MIN_CHUNK_SIZE), SB_MAX_CHUNKS);

                if (numChunks == 1) { // This is a bit faster for small matrices
                    NVMatrix::setDeviceID(srcDevice);
                    toHostMem(src, hostmat, srcDevice);
                    NVMatrix::syncStream(_streams[srcDevice]);

                    for (map<int,NVMatrix*>::const_iterator it = mats.begin(); it != mats.end(); ++it) {
                        if (it->first != src.getDataDeviceID()) {
                            NVMatrix::setDeviceID(it->first);
                            toTarget(hostmat, *it->second, it->first, scaleTarget, scaleOutput);
                        }
                    }
                } else {
                    int n = src.getNumElements();

                    map<int,NVMatrix*> lines;
                    for (map<int,NVMatrix*>::const_iterator it = mats.begin(); it != mats.end(); ++it) {
                        lines[it->first] = &it->second->reshaped(1, n);
                        lines[it->first]->setTrans(src.isTrans());
                    }
                    NVMatrix& srcLine = *lines[srcDevice];
                    hostmat.reshape(1, n);

                    int chunkSize = DIVUP(n, numChunks);
                    bool trans = src.isTrans();
                    for (int i = 0; i < numChunks; ++i) {
                        int start = i * chunkSize;
                        int end = min((i+1) * chunkSize, n);
                        if (start < end) {
                            NVMatrix& tmpSrc = srcLine.sliceCols(start, end); // view
                            NVMatrix& tmpHostmem = hostmat.sliceCols(start, end); // view

                            NVMatrix::setDeviceID(srcDevice);
                            toHostMem(tmpSrc, tmpHostmem, srcDevice);
                            NVMatrix::syncStream(_streams[srcDevice]);

                            for (map<int,NVMatrix*>::const_iterator it = lines.begin(); it != lines.end(); ++it) {
                                if (it->first != srcDevice) {
                                    NVMatrix& tmpTgt = it->second->sliceCols(start, end); // view
                                    NVMatrix::setDeviceID(it->first);
                                    toTarget(tmpHostmem, tmpTgt, it->first, scaleTarget, scaleOutput);
                                    delete &tmpTgt;
                                }
                            }
                            delete &tmpSrc;
                            delete &tmpHostmem;
                        }
                    }
                    for (map<int,NVMatrix*>::const_iterator it = lines.begin(); it != lines.end(); ++it) {
                        delete it->second;
                    }
                }
                delete &hostmat;
            }
            for(map<int,NVMatrix*>::const_iterator it = mats.begin(); it != mats.end(); ++it) {
                if (it->first != srcDevice) {
                    NVMatrix::syncStream(_streams[it->first]);
                }
            }
        }
    }
    if (oldDeviceID >= 0) {
        NVMatrix::setDeviceID(oldDeviceID);
    }
}
