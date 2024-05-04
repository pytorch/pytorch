package org.pytorch.testapp;

class Result {

  public final float[] scores;
  public final long totalDuration;
  public final long moduleForwardDuration;

  public Result(float[] scores, long moduleForwardDuration, long totalDuration) {
    this.scores = scores;
    this.moduleForwardDuration = moduleForwardDuration;
    this.totalDuration = totalDuration;
  }
}
