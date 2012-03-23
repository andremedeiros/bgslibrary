#pragma once

#include <iostream>
#include <cv.h>
#include <highgui.h>

#include "WrenGA.h"

using namespace Algorithms::BackgroundSubtraction;

class DPWrenGABGS
{
private:
  bool firstTime;
  long frameNumber;
  IplImage* frame;
  RgbImage frame_data;

  WrenParams params;
  WrenGA bgs;
  BwImage lowThresholdMask;
  BwImage highThresholdMask;

  double threshold;
  double alpha;
  int learningFrames;
  bool showOutput;

public:
  DPWrenGABGS();
  ~DPWrenGABGS();

  void process(const cv::Mat &img_input, cv::Mat &img_output);

private:
  void saveConfig();
  void loadConfig();
};

