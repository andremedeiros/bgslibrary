#pragma once

#include <iostream>
#include <cv.h>
#include <highgui.h>

#include "ZivkovicAGMM.h"

using namespace Algorithms::BackgroundSubtraction;

class DPZivkovicAGMMBGS
{
private:
  bool firstTime;
  long frameNumber;
  IplImage* frame;
  RgbImage frame_data;

  ZivkovicParams params;
  ZivkovicAGMM bgs;
  BwImage lowThresholdMask;
  BwImage highThresholdMask;

  double threshold;
  double alpha;
  int gaussians;
  bool showOutput;

public:
  DPZivkovicAGMMBGS();
  ~DPZivkovicAGMMBGS();

  void process(const cv::Mat &img_input, cv::Mat &img_output);

private:
  void saveConfig();
  void loadConfig();
};

