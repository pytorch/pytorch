#pragma once

enum THDReduceOp {
  THDReduceMIN = 0,
  THDReduceMAX,
  THDReduceSUM,
  THDReducePRODUCT,
};

typedef int THDGroup;
const THDGroup THDGroupWORLD = 0;
