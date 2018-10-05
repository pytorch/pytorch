/*************************************************************************
 * Copyright (c) 2015-2017, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "core.h"
#include "libwrap.h"
#include "common_coll.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sched.h>
#include <fcntl.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <string.h>
#include <errno.h>

DebugLevel ncclDebugLevel;

NCCL_API(ncclResult_t, ncclGetUniqueId, ncclUniqueId* out);
ncclResult_t ncclGetUniqueId(ncclUniqueId* out) {
  NCCLCHECK(PtrCheck(out, "GetUniqueId", "out"));
  pid_t pid = getpid();
  static int count = 0;
  int commId = __sync_fetch_and_add(&count, 1);
  int len = snprintf(out->internal, NCCL_UNIQUE_ID_BYTES, "nccl-%d-%d", pid, commId);
  if(strlen(out->internal) < len) {
    WARN("ncclUniqueId truncated");
    return ncclInternalError;
  }
  return ncclSuccess;
}


static ncclResult_t shmOpen(const char* shmname, size_t bytes, void** ptr) {
  int fd = shm_open(shmname, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
  if (fd == -1) {
    WARN("shm_open failed to open %s", shmname);
    return ncclSystemError;
  }

  if (ftruncate(fd, bytes) == -1) {
    WARN("ftruncate failed to allocate %ld bytes", bytes);
    shm_unlink(shmname);
    close(fd);
    return ncclSystemError;
  }

  *ptr = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (*ptr == MAP_FAILED) {
    WARN("failure in mmap");
    shm_unlink(shmname);
    close(fd);
    return ncclSystemError;
  }

  close(fd);
  return ncclSuccess;
}

static ncclResult_t shmUnlink(const char* shmname) {
  if(shm_unlink(shmname) == -1) {
    WARN("smh_unlink failed");
    return ncclSystemError;
  } else {
    return ncclSuccess;
  }
}

static ncclResult_t shmUnmap(void* ptr, size_t bytes) {
  if(munmap(ptr, bytes) == -1) {
    WARN("munmap failed");
    return ncclSystemError;
  } else {
    return ncclSuccess;
  }
}


typedef struct {
  int rank;
  int ndev;
  int cudaDev;
  int sortId;
  pid_t pid;
  ncclMem* hostptr;
  ncclMem* devptr;
  cudaIpcMemHandle_t devipc;
  size_t buffSize;
} RankEntry;

static int compRanks(const void* a, const void* b) {
  const RankEntry* A = (const RankEntry*)a;
  const RankEntry* B = (const RankEntry*)b;
  if (A->sortId < B->sortId) return -1;
  if (A->sortId > B->sortId) return  1;
  return 0;
}

static void orderRanks(RankEntry* ranks, int count) {
  qsort(ranks, count, sizeof(RankEntry), compRanks);
}


typedef struct {
  union {
    struct {
      volatile int bar;
      int globalMemSpaceBroke;
    };
    char pad[16];
   };
   RankEntry ranks[1];
} RankGather;

static ncclResult_t initGather(RankGather** gather, ncclUniqueId commId,
    int ndev, int rank, RankEntry myInfo) {
  size_t bytes = offsetof(RankGather, ranks) + ndev*sizeof(RankEntry);
  RankGather* tmp = NULL;
  int bar_tmp;

  ncclResult_t res = shmOpen(commId.internal, bytes, (void**)&tmp);
  if (res != ncclSuccess) {
    WARN("rank %d failed to open shm segment for gather", rank);
    return res;
  }

  tmp->ranks[rank] = myInfo;

  bar_tmp = tmp->bar - 1;
  bool swapped;
  do {
    bar_tmp += 1;
    if (bar_tmp == ndev-1) { // everyone is done
      ncclResult_t res = shmUnlink(commId.internal);
      if (res != ncclSuccess) {
        WARN("rank %d failed to unlink shm segment for gather", rank);
        shmUnmap(tmp, bytes);
        return res;
      }

      orderRanks(tmp->ranks, ndev);
    }
    swapped = __sync_bool_compare_and_swap(&tmp->bar, bar_tmp, bar_tmp+1);
  } while(!swapped);

  while (tmp->bar < ndev)
    sched_yield();
  __sync_synchronize();

  *gather = tmp;
  return ncclSuccess;
}

static void syncRingDirect(RankGather* gather, int* globalMemSpaceOk) {
  int bar_tmp = gather->bar - 1;
  int ndev = gather->ranks[0].ndev;
  bool swapped;
  do {
    bar_tmp += 1;
    swapped = __sync_bool_compare_and_swap(&gather->bar, bar_tmp, bar_tmp+1);
  } while(!swapped);

  while (gather->bar < 2*ndev) // Wait for all ranks to arrive at this second barrier
    sched_yield();
  __sync_synchronize();

  *globalMemSpaceOk = gather->globalMemSpaceBroke ? 0 : 1;
}

static ncclResult_t closeGather(RankGather* gather, int ndev) {
  int bar_tmp = gather->bar - 1;
  bool swapped;
  do {
    bar_tmp += 1;
    swapped = __sync_bool_compare_and_swap(&gather->bar, bar_tmp, bar_tmp+1);
  } while(!swapped);

  while (gather->bar < 3*ndev) // Wait for all ranks to arrive at this third barrier
    sched_yield();
  __sync_synchronize();

  size_t bytes = offsetof(RankGather, ranks) + ndev*sizeof(RankEntry);
  ncclResult_t res = shmUnmap(gather, bytes);
  if (res != ncclSuccess) {
    WARN("failed to unmap %ld bytes of gather", bytes);
    return res;
  }

  return ncclSuccess;
}


static ncclResult_t allocDevMem(ncclMem** ptr, size_t buffSize) {
  size_t size = offsetof(struct ncclMem, buff) + buffSize;
  cudaError_t res = cudaMalloc((void**)ptr, size);
  if (res != cudaSuccess) {
    *ptr = NULL;
    WARN("failed to allocate %lu byte device buffer", size);
    return ncclCudaMallocFailed;
  }
  if (cudaMemset(*ptr, 0, size) != cudaSuccess) {
    WARN("failed to memset device buffer.");
    cudaFree(*ptr);
    *ptr = NULL;
    return ncclUnhandledCudaError;
  }
  return ncclSuccess;
}

static const int ShmMapped = 1;
static const int ShmLinked = 2;

static ncclResult_t allocHostMem(ncclMem** ptr, size_t buffSize) {
  size_t size = offsetof(struct ncclMem, buff) + buffSize;
  cudaError_t res = cudaMallocHost((void**)ptr, size);
  if (res != cudaSuccess) {
    *ptr = NULL;
    WARN("failed to allocate %lu byte host buffer", size);
    return ncclSystemError;
  }
  memset(*ptr, 0, size);
  return ncclSuccess;
}

static ncclResult_t openHostMemShm(const char* shmname, ncclMem** ptr, size_t buffSize) {
  size_t size = offsetof(struct ncclMem, buff) + buffSize;
  ncclResult_t res = shmOpen(shmname, size, (void**)ptr);
  if (res != ncclSuccess) {
    WARN("failed to allocate %lu byte shm buffer", size);
    *ptr = NULL;
    return res;
  }

  if(cudaHostRegister(*ptr, size, cudaHostRegisterMapped) != cudaSuccess) {
    WARN("failed to register host buffer");
    shmUnlink(shmname);
    shmUnmap(*ptr, size);
    *ptr = NULL;
    return ncclUnhandledCudaError;
  }
  return ncclSuccess;
}

static ncclResult_t populateRankInfo(RankEntry* info, int rank, ncclComm_t comm) {
  char busId[13];
  nvmlDevice_t nvmlHandle;
  cudaError_t res = cudaDeviceGetPCIBusId(busId, 13, comm->cudaDev);
  if (res == cudaErrorInvalidDevice) {
    WARN("rank %d attempted to access an invalid cuda device %d", rank, comm->cudaDev);
    return ncclInvalidDeviceIndex;
  } else if (res != cudaSuccess) {
    WARN("rank %d failed to get PCI Bus Id for device %d", rank, comm->cudaDev);
    return ncclUnhandledCudaError;
  }
  INFO("rank %d using device %d (%s)", rank, comm->cudaDev, busId);

  if (wrapNvmlDeviceGetHandleByPciBusId(busId, &nvmlHandle) != ncclSuccess) {
    WARN("rank %d failed to get nvml handle for device %s", rank, busId);
    return ncclUnhandledCudaError;
  }
  // Order by nvml index
  if (wrapNvmlDeviceGetIndex(nvmlHandle, (unsigned*)&info->sortId) != ncclSuccess) {
    WARN("rank %d failed to get nvml device index for device %d", rank, comm->cudaDev);
    return ncclUnhandledCudaError;
  }

  info->rank = rank;
  info->ndev = comm->nRanks;
  info->cudaDev = comm->cudaDev;
  info->pid = getpid();
  info->buffSize = comm->buffSize;
  info->hostptr = comm->hostMem;
  info->devptr = comm->devMem;
  if (cudaIpcGetMemHandle(&info->devipc, (void*)comm->devMem) != cudaSuccess) {
    WARN("rank %d failed to open CUDA IPC handle", rank);
    return ncclUnhandledCudaError;
  }

  return ncclSuccess;
}


static ncclResult_t commClearMaps(ncclComm_t comm) {
  ncclResult_t res, retval = ncclSuccess;
  cudaError_t cures;

  for(int d=0; d<comm->nRanks; ++d) {
    if (comm->ptrs[d].hostCleanup != NULL) {
      cures = cudaHostUnregister(comm->ptrs[d].hostCleanup);
      if (cures != cudaSuccess) {
        WARN("rank %d failed to unregister handle to device %d",
          comm->rank, d);
          retval = (retval == ncclSuccess) ? ncclUnhandledCudaError : retval;
      }
      res = shmUnmap(comm->ptrs[d].hostCleanup, offsetof(ncclMem, buff) + comm->buffSize);
      if (res != ncclSuccess) {
        WARN("rank %d failed to unmap handle to device %d",
          comm->rank, d);
          retval = (retval == ncclSuccess) ? res : retval;
      }
      comm->ptrs[d].hostCleanup = NULL;
    }

    if (comm->ptrs[d].devCleanup != NULL) {
      cures = cudaIpcCloseMemHandle((void*)comm->ptrs[d].devCleanup);
      if (cures != cudaSuccess) {
        WARN("rank %d failed to close IPC handle to device %d: %s",
          comm->rank, d, cudaGetErrorString(cures));
        retval = (retval == ncclSuccess) ? ncclUnhandledCudaError : retval;
      }
    }
  }

  if (comm->userFromRing != NULL)
    memset(comm->userFromRing, 0, sizeof(int)*comm->nRanks);
  if (comm->ncclFromRing != NULL)
    memset(comm->ncclFromRing, 0, sizeof(int)*comm->nRanks);

  if (comm->devUserFromRing != NULL) {
    cures = cudaMemset(comm->devUserFromRing, 0, sizeof(int)*comm->nRanks);
    if (cures != cudaSuccess) {
      WARN("Faild to clear dev map: %s", cudaGetErrorString(cures));
      retval = (retval == ncclSuccess) ? ncclUnhandledCudaError : retval;
    }
  }

  if (comm->devRing != NULL) {
    cures = cudaMemset(comm->devRing, 0, sizeof(DevRing<char>));
    if (cures != cudaSuccess) {
      WARN("Failed to clear devRing: %s", cudaGetErrorString(cures));
      retval = (retval == ncclSuccess) ? ncclUnhandledCudaError : retval;
    }
  }
  return retval;
}

static ncclResult_t commBuildMaps(ncclComm_t comm, ncclUniqueId* commId, int rank, RankEntry* ranks, int* globalMemSpaceBroke) {
  int ndev = comm->nRanks;
  comm->rank = rank;

  if (ndev > MAXRANKS) {
    WARN("%d ranks exceeds MAXRANKS of %d", ndev, MAXRANKS);
    return ncclUnsupportedDeviceCount;
  }

  // Check for inconsistencies between ranks
  // If two ranks use the same rank, then one slot of
  // ranks[] will be left unset with zero ndev/buffSize.
  for(int i=0; i<ndev; ++i) {
    if (ranks[i].buffSize != comm->buffSize
        || ranks[i].ndev != comm->nRanks) {
      commClearMaps(comm);
      return ncclRankMismatch;
    }
  }

  // Find self among ranks of gather
  int myNcclId = -1;
  for (int i=0; i<ndev; ++i) {
    if(ranks[i].rank == rank) {
      myNcclId = i;
      break;
    }
  }
  if (myNcclId == -1) {
    WARN("rank %d not found in communicator", rank);
    return ncclInvalidRank;
  }

  for(int ringPos=0; ringPos<ndev; ++ringPos) {
    int ncclPos = (ringPos+myNcclId) % ndev; // ring order relative to self
    int userRank = ranks[ncclPos].rank;
    comm->userFromRing[ringPos] = userRank;
    comm->ncclFromRing[ringPos] = ncclPos;
  }

  int myDev = ranks[myNcclId].cudaDev;
  pid_t myPid = ranks[myNcclId].pid;

  for (int i=0; i<ndev; ++i) {
    int iRank = ranks[i].rank;
    int iDev = ranks[i].cudaDev;
    pid_t iPid = ranks[i].pid;
    int canpeer = 0;

    int iIsNeighbor = (i == (myNcclId+1)%ndev) || (i == (myNcclId+ndev-1)%ndev);

    if (iIsNeighbor && cudaDeviceCanAccessPeer(&canpeer, myDev, iDev) != cudaSuccess) {
      INFO("peer query failed between rank %d (dev %d) and rank %d (dev %d)",
        rank, myDev, iRank, iDev);
      canpeer = 0;
    }

    cudaError_t err;
    ncclMem* remoteHostBuff;

    comm->ptrs[i].type = NodeRef::HOST; // Assume host buffer
    comm->ptrs[i].devCleanup = NULL;
    comm->ptrs[i].hostCleanup = NULL;

    if (iPid == myPid) {
      remoteHostBuff = ranks[i].hostptr;

      if (myDev == iDev) { // shared device
        INFO("rank access %d -> %d via common device", rank, iRank);
        comm->ptrs[i].type = NodeRef::DEVICE;
        comm->ptrs[i].local = ranks[myNcclId].devptr;
        comm->ptrs[i].remote = ranks[i].devptr;
      } else if (canpeer) {
        INFO("rank access %d -> %d via P2P device mem", rank, iRank);
        err = cudaDeviceEnablePeerAccess(iDev, 0);
        if (err == cudaErrorPeerAccessAlreadyEnabled) {
          cudaGetLastError();
        } else if (err != cudaSuccess) {
          WARN("rank %d failed to peer with device %d: %s",
              rank, iDev, cudaGetErrorString(err));
          commClearMaps(comm);
          return ncclUnhandledCudaError;
        }
        comm->ptrs[i].type = NodeRef::DEVICE;
        comm->ptrs[i].local = ranks[myNcclId].devptr;
        comm->ptrs[i].remote = ranks[i].devptr;
      }
    } else { // Separate processes
      *globalMemSpaceBroke = 1;
      char rankname[1024];
      sprintf(rankname, "%s-%d", commId->internal, ranks[i].rank);
      if (openHostMemShm(rankname, &remoteHostBuff, ranks[i].buffSize)
          != ncclSuccess) {
        WARN("rank %d failed to open sysmem buffer of rank %d", rank, iRank);
        commClearMaps(comm);
        return ncclUnhandledCudaError;
      }
      comm->ptrs[i].hostCleanup = remoteHostBuff;

      // TODO: Extend to same device (MPS) case.
      // At present that would go through host mem.
      if (canpeer) {
        INFO("rank access %d -> %d via IPC device mem", rank, iRank);
        comm->ptrs[i].type = NodeRef::DEVICE;
        comm->ptrs[i].local  = ranks[myNcclId].devptr;
        err = cudaIpcOpenMemHandle((void**)(&comm->ptrs[i].remote),
            ranks[i].devipc, cudaIpcMemLazyEnablePeerAccess);
        if (err != cudaSuccess) {
          WARN("rank %d failed to open Ipc handle to rank %d: %s",
              rank, iRank, cudaGetErrorString(err));
          commClearMaps(comm);
          return ncclUnhandledCudaError;
        }
        comm->ptrs[i].devCleanup = comm->ptrs[i].remote;
      }
    }

    err = cudaHostGetDevicePointer(&comm->ptrs[i].opCounter,
          &(remoteHostBuff->opCounter), 0);
    if (err != cudaSuccess) {
      WARN("rank %d failed to obtain %d's zero copy pointer: %s",
          rank, iRank, cudaGetErrorString(err));
      commClearMaps(comm);
      return ncclUnhandledCudaError;
    }

    if (comm->ptrs[i].type == NodeRef::HOST) {
      *globalMemSpaceBroke = 1;
      INFO("rank access %d -> %d via zero-copy host mem", rank, iRank);
      if (cudaHostGetDevicePointer(&comm->ptrs[i].local, ranks[myNcclId].hostptr, 0) != cudaSuccess) {
        WARN("rank %d failed to map zero copy buffer to device", rank);
        commClearMaps(comm);
        return ncclUnhandledCudaError;
      }
      if (cudaHostGetDevicePointer(&comm->ptrs[i].remote, remoteHostBuff, 0) != cudaSuccess) {
        WARN("rank %d failed to map %d's zero copy buffer to device", rank, iRank);
        commClearMaps(comm);
        return ncclUnhandledCudaError;
      }
    }
  }

  // Setup device-side ring view
  if (cudaMemcpy(comm->devUserFromRing, comm->userFromRing, ndev*sizeof(int),
      cudaMemcpyHostToDevice) != cudaSuccess) {
    WARN("rank %d failed to copy maps to device", rank);
    commClearMaps(comm);
    return ncclUnhandledCudaError;
  }

  DevRing<char> ringTemp;
  memcpy(ringTemp.userRank, comm->userFromRing, ndev*sizeof(int));

  int prevIdx = comm->ncclFromRing[comm->nRanks-1];
  int nextIdx = comm->ncclFromRing[1 % comm->nRanks];
  NodeRef* prevPtrs = comm->ptrs+prevIdx;
  NodeRef* nextPtrs = comm->ptrs+nextIdx;

  ringTemp.prevOpCounter    = prevPtrs->opCounter;
  ringTemp.nextOpCounter    = nextPtrs->opCounter;
  ringTemp.sendFlagToNext   = nextPtrs->remote->flags;
  ringTemp.recvFlagFromPrev = prevPtrs->local->flags;
  ringTemp.sendFlagToPrev   = prevPtrs->remote->flags+1;
  ringTemp.recvFlagFromNext = nextPtrs->local->flags+1;

  ringTemp.recvPtrFromNext = (char**)&nextPtrs->local->recvPtrs;
  ringTemp.sendPtrToPrev   = (char**)&prevPtrs->remote->recvPtrs;

  ringTemp.recvBuffer = prevPtrs->local->buff;
  ringTemp.sendBuffer = nextPtrs->remote->buff;

  if (cudaMemcpy(comm->devRing, &ringTemp, sizeof(ringTemp),
      cudaMemcpyHostToDevice) != cudaSuccess) {
    WARN("rank %d failed to copy ring maps to device", rank);
    commClearMaps(comm);
    return ncclUnhandledCudaError;
  }

  return ncclSuccess;
}

static void initDebug() {
  const char* nccl_debug = getenv("NCCL_DEBUG");
  if (nccl_debug == NULL) {
    ncclDebugLevel = NONE;
  } else if (strcmp(nccl_debug, "VERSION") == 0) {
    ncclDebugLevel = VERSION;
  } else if (strcmp(nccl_debug, "WARN") == 0) {
    ncclDebugLevel = WARN;
  } else if (strcmp(nccl_debug, "INFO") == 0) {
    ncclDebugLevel = INFO;
    INFO("NCCL debug level set to INFO");
  } else if (strcmp(nccl_debug, "ABORT") == 0) {
    ncclDebugLevel = ABORT;
    INFO("NCCL debug level set to ABORT");
  }
}

static void commFree(ncclComm_t comm) {
  if (comm == NULL)
    return;

  if (comm->doneEvent != NULL)
    if (cudaEventDestroy(comm->doneEvent) != cudaSuccess)
      INFO("ncclComm failed to destroy doneEvent");

  ncclResult_t res = commClearMaps(comm);
  if (res != ncclSuccess)
    INFO("failed to cleanup comm maps");

  if (comm->devRing != NULL)
    if (cudaFree(comm->devRing) != cudaSuccess)
      INFO("commFree failed to free devRing");

  if (comm->userFromRing != NULL)
    free(comm->userFromRing);

  if (comm->devUserFromRing != NULL)
    if (cudaFree(comm->devUserFromRing) != cudaSuccess)
      INFO("commFree failed to free dev maps");

  if (comm->ncclFromRing != NULL)
    free(comm->ncclFromRing);

  if (comm->devMem != NULL && cudaFree(comm->devMem) != cudaSuccess)
    INFO("Failed to free devMap");

  if (comm->hostMem != NULL) {
    if (comm->hostMemState & ShmMapped) {
      if (cudaHostUnregister(comm->hostMem) != cudaSuccess)
        INFO("Failed to unregister hostMem");
      size_t size = offsetof(ncclMem, buff) + comm->buffSize;
      if (shmUnmap(comm->hostMem, size) != ncclSuccess)
        INFO("Failed to unmap hostMem");
      comm->hostMemState ^= ShmMapped;
    } else {
      cudaFreeHost(comm->hostMem);
    }
  }
  free(comm);
}

static ncclResult_t commAlloc(ncclComm_t* comret, int ndev, const ncclUniqueId* commId, int rank) {
  size_t commBytes = offsetof(ncclComm, ptrs) + ndev*sizeof(NodeRef);
  struct ncclComm* comm = (struct ncclComm*)malloc(commBytes);
  if (comm == NULL) {
    WARN("comm allocation failed");
    return ncclSystemError;
  }
  memset(comm, 0, commBytes);

  comm->nRanks = ndev;
  cudaGetDevice(&comm->cudaDev);

  const char* str = getenv("NCCL_BUFFSIZE");
  int buffsize;
  if (str != NULL) {
    errno = 0;
    buffsize = strtol(str, NULL, 10);
    if (errno == ERANGE || buffsize == 0) {
      INFO("rank %d invalid NCCL_BUFFSIZE: %s, using default %lu",
          rank, str, DEFAULT_BUFFER_SIZE_BYTES);
      buffsize = DEFAULT_BUFFER_SIZE_BYTES;
    }
  } else {
    buffsize = DEFAULT_BUFFER_SIZE_BYTES;
  }
  comm->buffSize = buffsize;
  INFO("rank %d using buffSize = %lu", rank, comm->buffSize);


  ncclResult_t res;
  res = allocDevMem(&comm->devMem, comm->buffSize);
  if (res != ncclSuccess) {
    WARN("rank %d failed to allocate device buffer", rank);
    commFree(comm);
    return res;
  }

  if (cudaMalloc(&comm->devRing, sizeof(DevRing<char>)) != cudaSuccess) {
    WARN("rank %d failed to allocate device-side ring views", rank);
    commFree(comm);
    return ncclCudaMallocFailed;
  }

  if (cudaMalloc(&comm->devUserFromRing, ndev*sizeof(int)) != cudaSuccess ) {
    WARN("rank %d failed to allocated device maps", rank);
    commFree(comm);
    return ncclCudaMallocFailed;
  }

  comm->userFromRing = (int*)malloc(ndev*sizeof(int));
  if (comm->userFromRing == NULL) {
    WARN("rank %d failed to allocate host maps", rank);
    commFree(comm);
    return ncclSystemError;
  }

  comm->ncclFromRing = (int*)malloc(ndev*sizeof(int));
  if (comm->ncclFromRing == NULL) {
    WARN("rank %d failed to allocate host maps", rank);
    commFree(comm);
    return ncclSystemError;
  }

  if (cudaEventCreateWithFlags(&comm->doneEvent, cudaEventDisableTiming) != cudaSuccess) {
    WARN("ncclComm on rank %d failed to create doneEvent", rank);
    commFree(comm);
    return ncclUnhandledCudaError;
  }

  if(commId == NULL) {
    comm->hostMemState = 0;
    res = allocHostMem(&comm->hostMem, comm->buffSize);
  } else {
    char rankname[1024];
    sprintf(rankname, "%s-%d", commId->internal, rank);
    res = openHostMemShm(rankname, &comm->hostMem, comm->buffSize);
    if (res != ncclSuccess) {
      WARN("rank %d failed to allocate host buffer", rank);
      commFree(comm);
      return res;
    }
    comm->hostMemState = ShmMapped | ShmLinked;
  }

  if (cudaHostGetDevicePointer(&comm->opCounter, &comm->hostMem->opCounter, 0) != cudaSuccess) {
    WARN("ncclComm on rank %d failed to map opCounter to device", rank);
    commFree(comm);
    return ncclUnhandledCudaError;
  }

  *comret = comm;
  return ncclSuccess;
}

static ncclResult_t devCommUpdate(ncclComm_t comm) {
  // Copy the comm on the device
  size_t commBytes = offsetof(ncclComm, ptrs) + comm->nRanks*sizeof(NodeRef);
  if (cudaMemcpy(comm->devComm, comm, commBytes, cudaMemcpyHostToDevice) != cudaSuccess) {
    WARN("failed to copy device comm");
    return ncclUnhandledCudaError;
  }
  // Fix the host pointer to be accessible from the device
  void* dptr;
  if (cudaHostGetDevicePointer(&dptr, comm->hostMem, 0) != cudaSuccess) {
    WARN("failed to get device pointer for host mem");
    return ncclUnhandledCudaError;
  }
  if (cudaMemcpy(&comm->devComm->hostMem, &dptr, sizeof(dptr), cudaMemcpyHostToDevice) != cudaSuccess) {
    WARN("failed to update host pointer");
    return ncclUnhandledCudaError;
  }
  return ncclSuccess;
}

static ncclResult_t devCommSetup(ncclComm_t comm) {
  // Fully duplicate the comm on the device
  size_t commBytes = offsetof(ncclComm, ptrs) + comm->nRanks*sizeof(NodeRef);
  if (cudaMalloc(&comm->devComm, commBytes) != cudaSuccess) {
    WARN("failed to allocated device comm");
    return ncclCudaMallocFailed;
  }
  return devCommUpdate(comm);
}

static ncclResult_t commUnlinkHostMem(ncclComm_t comm, ncclUniqueId commId, int rank) {
  char rankname[1024];
  sprintf(rankname, "%s-%d", commId.internal, rank);
  if (comm->hostMemState & ShmLinked)
    comm->hostMemState ^= ShmLinked;
  return shmUnlink(rankname);
}

static void showVersion() {
  static int shown = 0;
  if (shown == 0 && ncclDebugLevel >= VERSION) {
    printf("NCCL version %d.%d.%d compiled with CUDA %d.%d\n", NCCL_MAJOR, NCCL_MINOR, NCCL_PATCH, CUDA_MAJOR, CUDA_MINOR);
    fflush(stdout);
    shown = 1;
  }
}

NCCL_API(ncclResult_t, ncclCommInitRank, ncclComm_t* newcomm, int ndev, ncclUniqueId commId, int myrank);
ncclResult_t ncclCommInitRank(ncclComm_t* newcomm, int ndev, ncclUniqueId commId, int myrank) {
  if (myrank == 0) showVersion();

  NCCLCHECK(PtrCheck(newcomm, "CommInitRank", "newcomm"));

  if (ndev < 1) {
    WARN("Invalid device count requested : %d", ndev);
    return ncclUnsupportedDeviceCount;
  }
  if (myrank >= ndev || myrank < 0) {
    WARN("Invalid rank %d, should be in the range 0..%d", myrank, ndev-1);
    return ncclInvalidRank;
  }

  if (strlen(commId.internal) < 1 ||
      strlen(commId.internal) >= NCCL_UNIQUE_ID_BYTES) {
    WARN("rank %d invalid commId", myrank);
    return ncclInvalidArgument;
  }

  initDebug();
  ncclResult_t res;
  RankEntry myStuff;
  RankGather* gath = NULL;

  res = wrapSymbols();
  if (res != ncclSuccess) {
    WARN("NCCL failed to initialize client libs");
    return res;
  }

  res = wrapNvmlInit();
  if (res != ncclSuccess) {
    WARN("rank %d failed to initialize nvml", myrank);
    return res;
  }

  res = commAlloc(newcomm, ndev, &commId, myrank);
  if (res != ncclSuccess) {
    WARN("rank %d failed to allocate communicator", myrank);
    return res;
  }

  res = populateRankInfo(&myStuff, myrank, *newcomm);
  if (res != ncclSuccess) {
    WARN("rank %d failed to obtain rank info", myrank);
    goto cleanup;
  }

  res = initGather(&gath, commId, ndev, myrank, myStuff);
  if (res != ncclSuccess) {
    WARN("rank %d failed to gather rank info", myrank);
    goto cleanup;
  }

  res = commBuildMaps(*newcomm, &commId, myrank, gath->ranks, &gath->globalMemSpaceBroke);
  syncRingDirect(gath, &((*newcomm)->globalMemSpace));
  if (res != ncclSuccess) {
    WARN("rank %d failed to build comm maps", myrank);
    goto cleanup;
  }

  INFO("Global device memory space is %s", (*newcomm)->globalMemSpace ? "enabled" : "disabled");

  res = closeGather(gath, ndev); // includes a barrier
  gath = NULL;
  if (res != ncclSuccess) {
    WARN("rank %d failed to close gather", myrank);
    goto cleanup;
  }

  res = devCommSetup(*newcomm);
  if (res != ncclSuccess) {
    WARN("rank %d failed to copy dcomm", myrank);
    goto cleanup;
  }

  res = ncclSuccess;
  goto final;

  cleanup:
  if (gath != NULL)
    closeGather(gath, ndev);
  commFree(*newcomm);

  final:
  if ((*newcomm)->hostMemState & ShmLinked) {
    if (commUnlinkHostMem(*newcomm, commId, myrank) != ncclSuccess)
      INFO("rank %d failed to unlink host mem shm segment", myrank);
  }

  if (wrapNvmlShutdown() != ncclSuccess)
    INFO("rank %d did not shutdown nvml properly", myrank);
  return res;
}

NCCL_API(ncclResult_t, ncclCommInitAll, ncclComm_t* comms, int ndev, const int* devlist);
ncclResult_t ncclCommInitAll(ncclComm_t* comms, int ndev, const int* devlist) {
  initDebug();

  showVersion();

  NCCLCHECK(PtrCheck(comms, "CommInitAll", "comms"));

  if (ndev < 1) {
    WARN("Invalid device count requested : %d", ndev);
    return ncclUnsupportedDeviceCount;
  }

  ncclResult_t res;
  int savedDevice;
  RankEntry* ranks = NULL;
  int rank, cudaDev;
  ncclComm_t comm = NULL;
  char busId[13];
  nvmlDevice_t nvmlHandle;
  int affinity_set = 0;
  int globalMemSpaceBroke = 0; // Assume direct access to recv ptr OK

  res = wrapSymbols();
  if (res != ncclSuccess) {
    WARN("NCCL failed to initialize client libs");
    return res;
  }

  cudaGetDevice(&savedDevice);
  ranks = (RankEntry*)malloc(ndev*sizeof(RankEntry));
  if (ranks == NULL) {
    WARN("NCCL allocation failed");
    return ncclSystemError;
  }
  memset(ranks, 0, ndev*sizeof(RankEntry));

  res = wrapNvmlInit();
  if (res != ncclSuccess) {
    WARN("nccl failed to initialize nvml");
    return res;
  }

  for(rank=0; rank<ndev; ++rank)
    comms[rank] = NULL;

  for (rank=0; rank<ndev; ++rank) {
    cudaDev = (devlist == NULL) ? rank : devlist[rank];
    if (cudaSetDevice(cudaDev) != cudaSuccess) {
      WARN("rank %d failed to set cuda device %d", rank, cudaDev);
      res = ncclInvalidDeviceIndex;
      goto cleanup;
    }

    // Set CPU affinity
    affinity_set = 0;
    if (cudaDeviceGetPCIBusId(busId, 13, cudaDev) != cudaSuccess) {
      INFO("rank %d failed to get PCI Bus Id for device %d", rank, cudaDev);
      goto skipaffinity;
    }
    if (wrapNvmlDeviceGetHandleByPciBusId(busId, &nvmlHandle) != ncclSuccess) {
      INFO("rank %d failed to get nvml handle for device %s", rank, busId);
      goto skipaffinity;
    }
    if (wrapNvmlDeviceSetCpuAffinity(nvmlHandle) != ncclSuccess) {
      INFO("rank %d failed to set affinity", rank);
      goto skipaffinity;
    }
    affinity_set = 1;
    skipaffinity:

    res = commAlloc(&comm, ndev, NULL, rank);
    if (res != ncclSuccess) {
      WARN("rank %d failed to allocate communicator", rank);
      goto cleanup;
    }
    comms[rank] = comm;

    if (affinity_set && wrapNvmlDeviceClearCpuAffinity(nvmlHandle) != ncclSuccess) {
      INFO("rank %d set but failed to clear cpu affinity", rank);
    }
    res = populateRankInfo(ranks+rank, rank, comm);
    if (res != ncclSuccess) {
      WARN("rank %d failed to obtain rank info", rank);
      goto cleanup;
    }
  }

  orderRanks(ranks, ndev);
  for(rank=0; rank<ndev; ++rank) {
    comm = comms[rank];
    cudaSetDevice(comm->cudaDev);
    res = commBuildMaps(comm, NULL, rank, ranks, &globalMemSpaceBroke);
    if (res != ncclSuccess) {
      WARN("rank %d failed to build comm maps", rank);
      goto cleanup;
    }
  }

  INFO("Global device memory space is %s", (globalMemSpaceBroke) ? "disabled" : "enabled");
  for(rank=0; rank<ndev; ++rank) {
    comms[rank]->globalMemSpace = globalMemSpaceBroke ? 0 : 1;
  }
 
  for(rank=0; rank<ndev; ++rank) {
    res = devCommSetup(comms[rank]);
    if (res != ncclSuccess) {
      WARN("rank %d failed to copy dcomm", rank);
      goto cleanup;
    }
  }

  free(ranks);
  ranks = NULL;
  res = ncclSuccess;
  goto final;

  cleanup:
  if (ranks != NULL)
    free(ranks);
  for(rank=0; rank<ndev; ++rank) {
    if(comms[rank] != NULL) {
      commFree(comms[rank]);
    }
  }

  final:
  if(wrapNvmlShutdown() != ncclSuccess)
    INFO("NCCL did not shutdown nvml properly");
  cudaSetDevice(savedDevice);
  return res;
}

NCCL_API(void, ncclCommDestroy, ncclComm_t comm);
void ncclCommDestroy(ncclComm_t comm) {
  if (comm == NULL)
    return;

  int savedDevice;
  cudaGetDevice(&savedDevice);
  int commDevice = comm->cudaDev;

  if (savedDevice != commDevice) {
    CUDACHECK(cudaSetDevice(commDevice), void());
  }

  commFree(comm);

  if (savedDevice != commDevice)
    cudaSetDevice(savedDevice);
}

NCCL_API(const char*, ncclGetErrorString, ncclResult_t code);
const char* ncclGetErrorString(ncclResult_t code) {
  switch (code) {
  case ncclSuccess                : return "no error";
  case ncclUnhandledCudaError     : return "unhandled cuda error";
  case ncclSystemError            : return "system error";
  case ncclInternalError          : return "internal error";
  case ncclInvalidDevicePointer   : return "invalid device pointer";
  case ncclInvalidRank            : return "invalid rank";
  case ncclUnsupportedDeviceCount : return "unsupported device count";
  case ncclDeviceNotFound         : return "device not found";
  case ncclInvalidDeviceIndex     : return "invalid device index";
  case ncclLibWrapperNotSet       : return "lib wrapper not initialized";
  case ncclCudaMallocFailed       : return "cuda malloc failed";
  case ncclRankMismatch           : return "parameter mismatch between ranks";
  case ncclInvalidArgument        : return "invalid argument";
  case ncclInvalidType            : return "invalid data type";
  case ncclInvalidOperation       : return "invalid reduction operations";
  }
  return "unknown result code";
}

NCCL_API(ncclResult_t, ncclCommCount, const ncclComm_t comm, int* count);
ncclResult_t ncclCommCount(const ncclComm_t comm, int* count) {
  NCCLCHECK(PtrCheck(comm, "CommCount", "comm"));
  NCCLCHECK(PtrCheck(count, "CommCount", "count"));
  *count = comm->nRanks;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommCuDevice, const ncclComm_t comm, int* devid);
ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int* devid) {
  NCCLCHECK(PtrCheck(comm, "CommCuDevice", "comm"));
  NCCLCHECK(PtrCheck(devid, "CommCuDevice", "devid"));
  *devid = comm->cudaDev;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommUserRank, const ncclComm_t comm, int* rank);
ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank) {
  NCCLCHECK(PtrCheck(comm, "CommUserRank", "comm"));
  NCCLCHECK(PtrCheck(rank, "CommUserRank", "rank"));
  *rank = comm->rank;
  return ncclSuccess;
}

