#pragma once

#include <iostream>
#include <cv.h>
#include <highgui.h>

class StaticFrameDifferenceBGS
{
private:
  bool firstTime;
  cv::Mat img_background;
  cv::Mat img_foreground;
  bool enableThreshold;
  int threshold;
  bool showOutput;

public:
  StaticFrameDifferenceBGS();
  ~StaticFrameDifferenceBGS();

  void setBackgroundRef(const cv::Mat &img_bkg);
  void process(const cv::Mat &img_input, cv::Mat &img_output);

private:
  void saveConfig();
  void loadConfig();
};

