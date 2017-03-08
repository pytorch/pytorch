#pragma once

enum THDReduceOp {
  THDReduceMIN = 0,
  THDReduceMAX,
  THDReduceSUM,
  THDReducePRODUCT,
};

typedef int THDGroup;
static THDGroup THDGroupWORLD = 0;
