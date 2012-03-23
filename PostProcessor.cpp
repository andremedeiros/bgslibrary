#include "PostProcessor.h"

PostProcessor::PostProcessor() : firstTime(true), showOutput(true),
  enableMorphOpr(1), openInt(0), closeInt(0), enableDEfirst(1), dilateInt(0), erodeInt(0)
{
  std::cout << "PostProcessor()" << std::endl;
}

PostProcessor::~PostProcessor()
{
  std::cout << "~PostProcessor()" << std::endl;
}

void PostProcessor::process(const cv::Mat &img_input, cv::Mat &img_output)
{
  if(img_input.empty())
    return;

  loadConfig();
  
  if(firstTime)
    saveConfig();

  cv::Mat img_binw;
  img_input.copyTo(img_binw);

  if(enableMorphOpr == 0)
  {
    img_binw.copyTo(img_output);
    return;
  }

  // --------------------------------------------
  // Morphological operations
  // --------------------------------------------
  // Dilates an image by using a specific structuring element.
  // http://opencv.willowgarage.com/documentation/cpp/imgproc_image_filtering.html#cv-dilate
  // Erodes an image by using a specific structuring element.
  // http://opencv.willowgarage.com/documentation/cpp/imgproc_image_filtering.html#cv-erode

  cv::Mat element(3,3,CV_8U,cv::Scalar(1));
  
  //cv::morphologyEx(img_binw, img_binw, cv::MORPH_BLACKHAT, element, cv::Point(-1,-1), interations);
  //cv::morphologyEx(img_binw, img_binw, cv::MORPH_TOPHAT, element, cv::Point(-1,-1), interations);
  //cv::morphologyEx(img_binw, img_binw, cv::MORPH_GRADIENT, element, cv::Point(-1,-1), interations);
  //cv::morphologyEx(img_binw, img_binw, cv::MORPH_CROSS, element, cv::Point(-1,-1), interations);
  //cv::morphologyEx(img_binw, img_binw, cv::MORPH_RECT, element, cv::Point(-1,-1), interations);
  //cv::morphologyEx(img_binw, img_binw, cv::MORPH_ELLIPSE, element, cv::Point(-1,-1), interations);
  //cv::morphologyEx(img_binw, img_binw, cv::MORPH_OPEN, element, cv::Point(-1,-1), interations);
  //cv::morphologyEx(img_binw, img_binw, cv::MORPH_CLOSE, element, cv::Point(-1,-1), interations);

  // --------------------------------------------
  // Opening morphological operation
  // --------------------------------------------
  cv::morphologyEx(img_binw, img_binw, cv::MORPH_OPEN, element, cv::Point(-1,-1), openInt);

  // -------------------------------------------------------------------
  // Closing morphological operation - help to fill disconnected contour
  // -------------------------------------------------------------------
  cv::morphologyEx(img_binw, img_binw, cv::MORPH_CLOSE, element, cv::Point(-1,-1), closeInt);

  // --------------------------------------------
  // Dilate/Erose morphological operation
  // --------------------------------------------
  if(enableDEfirst == 0)
  {
    cv::morphologyEx(img_binw, img_binw, cv::MORPH_DILATE, element, cv::Point(-1,-1), dilateInt);
    cv::morphologyEx(img_binw, img_binw, cv::MORPH_ERODE, element, cv::Point(-1,-1), erodeInt);
  }
  else
  {
    cv::morphologyEx(img_binw, img_binw, cv::MORPH_ERODE, element, cv::Point(-1,-1), erodeInt);
    cv::morphologyEx(img_binw, img_binw, cv::MORPH_DILATE, element, cv::Point(-1,-1), dilateInt);
  }
  
  if(showOutput)
    cv::imshow("Post Processor", img_binw);

  img_binw.copyTo(img_output);

  firstTime = false;
}

void PostProcessor::saveConfig()
{
  CvFileStorage* fs = cvOpenFileStorage("./config/PostProcessor.xml", 0, CV_STORAGE_WRITE);

  cvWriteInt(fs, "enableMorphOpr", enableMorphOpr);
  cvWriteInt(fs, "openInt", openInt);
  cvWriteInt(fs, "closeInt", closeInt);
  cvWriteInt(fs, "enableDEfirst", enableDEfirst);
  cvWriteInt(fs, "dilateInt", dilateInt);
  cvWriteInt(fs, "erodeInt", erodeInt);
  cvWriteInt(fs, "showOutput", showOutput);

  cvReleaseFileStorage(&fs);
}

void PostProcessor::loadConfig()
{
  CvFileStorage* fs = cvOpenFileStorage("./config/PostProcessor.xml", 0, CV_STORAGE_READ);

  enableMorphOpr = cvReadIntByName(fs, 0, "enableMorphOpr", true);
  openInt = cvReadIntByName(fs, 0, "openInt", 0);
  closeInt = cvReadIntByName(fs, 0, "closeInt", 0);
  enableDEfirst = cvReadIntByName(fs, 0, "enableDEfirst", true);
  dilateInt = cvReadIntByName(fs, 0, "dilateInt", 0);
  erodeInt = cvReadIntByName(fs, 0, "erodeInt", 0);
  showOutput = cvReadIntByName(fs, 0, "showOutput", true);

  cvReleaseFileStorage(&fs);
}