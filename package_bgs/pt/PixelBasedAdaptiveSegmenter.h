#pragma once

#include <iostream>
#include <cv.h>
#include <highgui.h>

#include "PBAS.h"
#include "../IBGS.h"

class PixelBasedAdaptiveSegmenter : public IBGS
{
private:
  PBAS pbas;

  bool firstTime;
  bool showOutput;

  bool enableInputBlur;
  bool enableOutputBlur;

public:
  PixelBasedAdaptiveSegmenter();
  ~PixelBasedAdaptiveSegmenter();

  void process(const cv::Mat &img_input, cv::Mat &img_output);

private:
  void saveConfig();
  void loadConfig();
};