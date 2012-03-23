#pragma once

#include <iostream>
#include <cv.h>
#include <highgui.h>

class PostProcessor
{
private:
  bool firstTime;
  bool showOutput;
  int enableMorphOpr;
  int openInt;
  int closeInt;
  int enableDEfirst;
  int dilateInt;
  int erodeInt;

public:
  PostProcessor();
  ~PostProcessor();

  void process(const cv::Mat &img_input, cv::Mat &img_output);

private:
  void saveConfig();
  void loadConfig();
};

