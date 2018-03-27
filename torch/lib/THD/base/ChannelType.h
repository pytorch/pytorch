#pragma once

enum THDChannelType {
  THDChannelTCP = 0,
  THDChannelMPI,
  THDChannelGloo,
  THDChannelNccl
};
