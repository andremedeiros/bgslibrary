#pragma once

#include <iostream>
#include <cv.h>
#include <highgui.h>

class FrameDifferenceBGS
{
private:
  bool firstTime;
  cv::Mat img_input_prev;
  cv::Mat img_foreground;
  bool enableThreshold;
  int threshold;
  bool showOutput;

public:
  FrameDifferenceBGS();
  ~FrameDifferenceBGS();

  void process(const cv::Mat &img_input, cv::Mat &img_output);

private:
  void saveConfig();
  void loadConfig();
};