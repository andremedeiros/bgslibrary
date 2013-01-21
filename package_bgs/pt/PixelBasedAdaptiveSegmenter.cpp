#include "PixelBasedAdaptiveSegmenter.h"

PixelBasedAdaptiveSegmenter::PixelBasedAdaptiveSegmenter() : firstTime(true), showOutput(true), enableInputBlur(true), enableOutputBlur(true)
{
  std::cout << "PixelBasedAdaptiveSegmenter()" << std::endl;
}

PixelBasedAdaptiveSegmenter::~PixelBasedAdaptiveSegmenter()
{
  std::cout << "~PixelBasedAdaptiveSegmenter()" << std::endl;
}

void PixelBasedAdaptiveSegmenter::process(const cv::Mat &img_input, cv::Mat &img_output)
{
  if(img_input.empty())
    return;

  loadConfig();

  if(firstTime)
  {
    pbas.setAlpha(7.0);
    pbas.setBeta(1.0);
    pbas.setN(20);
    pbas.setRaute_min(2);
    pbas.setR_incdec(0.05);
    pbas.setR_lower(18);
    pbas.setR_scale(5);
    pbas.setT_dec(0.05);
    pbas.setT_inc(1);
    pbas.setT_init(18);
    pbas.setT_lower(2);
    pbas.setT_upper(200);

    saveConfig();
  }

  cv::Mat img_input_new;
  if(enableInputBlur)
    cv::GaussianBlur(img_input, img_input_new, cv::Size(5,5), 1.5);
  else
    img_input.copyTo(img_input_new);

  cv::Mat img_foreground;
  pbas.process(&img_input_new, &img_foreground);

  if(enableOutputBlur)
    cv::medianBlur(img_foreground, img_foreground, 5);

  if(showOutput)
    cv::imshow("PBAS", img_foreground);

  img_foreground.copyTo(img_output);

  firstTime = false;
}

void PixelBasedAdaptiveSegmenter::saveConfig()
{
  CvFileStorage* fs = cvOpenFileStorage("./config/PixelBasedAdaptiveSegmenter.xml", 0, CV_STORAGE_WRITE);

  cvWriteInt(fs, "enableInputBlur", enableInputBlur);
  cvWriteInt(fs, "enableOutputBlur", enableOutputBlur);

  cvWriteInt(fs, "showOutput", showOutput);

  cvReleaseFileStorage(&fs);
}

void PixelBasedAdaptiveSegmenter::loadConfig()
{
  CvFileStorage* fs = cvOpenFileStorage("./config/PixelBasedAdaptiveSegmenter.xml", 0, CV_STORAGE_READ);
  
  enableInputBlur = cvReadIntByName(fs, 0, "enableInputBlur", true);
  enableOutputBlur = cvReadIntByName(fs, 0, "enableOutputBlur", true);
  
  showOutput = cvReadIntByName(fs, 0, "showOutput", true);

  cvReleaseFileStorage(&fs);
}