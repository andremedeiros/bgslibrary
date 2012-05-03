#include <iostream>
#include <cv.h>
#include <highgui.h>

#include "package_bgs/FrameDifferenceBGS.h"
#include "package_bgs/StaticFrameDifferenceBGS.h"
#include "package_bgs/WeightedMovingMeanBGS.h"
#include "package_bgs/WeightedMovingVarianceBGS.h"
#include "package_bgs/MixtureOfGaussianV1BGS.h"
#include "package_bgs/MixtureOfGaussianV2BGS.h"
#include "package_bgs/AdaptiveBackgroundLearning.h"

#include "package_bgs/dp/DPAdaptiveMedianBGS.h"
#include "package_bgs/dp/DPGrimsonGMMBGS.h"
#include "package_bgs/dp/DPZivkovicAGMMBGS.h"
#include "package_bgs/dp/DPMeanBGS.h"
#include "package_bgs/dp/DPWrenGABGS.h"
#include "package_bgs/dp/DPPratiMediodBGS.h"
#include "package_bgs/dp/DPEigenbackgroundBGS.h"

#include "package_bgs/tb/T2FGMM_UM.h"
#include "package_bgs/tb/T2FGMM_UV.h"
#include "package_bgs/tb/FuzzySugenoIntegral.h"
#include "package_bgs/tb/FuzzyChoquetIntegral.h"

#include "package_bgs/jmo/MultiLayerBGS.h"

#include "package_bgs/lb/LBSimpleGaussian.h"
#include "package_bgs/lb/LBFuzzyGaussian.h"
#include "package_bgs/lb/LBMixtureOfGaussians.h"
#include "package_bgs/lb/LBAdaptiveSOM.h"
#include "package_bgs/lb/LBFuzzyAdaptiveSOM.h"

int main(int argc, char **argv)
{
  CvCapture *capture = 0;
  
  capture = cvCaptureFromCAM(0);
  //capture = cvCaptureFromAVI("video.avi");
  
  if(!capture){
    std::cerr << "Cannot open initialize webcam!" << std::endl;
    return 1;
  }
  
  IplImage *frame = cvQueryFrame(capture);
  
  /* Background Subtraction Methods */
  
  FrameDifferenceBGS* bgs = new FrameDifferenceBGS;
  //StaticFrameDifferenceBGS* bgs = new StaticFrameDifferenceBGS;
  //WeightedMovingMeanBGS* bgs = new WeightedMovingMeanBGS;
  //WeightedMovingVarianceBGS* bgs = new WeightedMovingVarianceBGS;
  //MixtureOfGaussianV1BGS* bgs = new MixtureOfGaussianV1BGS;
  //MixtureOfGaussianV2BGS* bgs = new MixtureOfGaussianV2BGS;
  //AdaptiveBackgroundLearning* bgs = new AdaptiveBackgroundLearning;
  
  /*** DP Package (adapted from Donovan Parks) ***/
  //DPAdaptiveMedianBGS* bgs = new DPAdaptiveMedianBGS;
  //DPGrimsonGMMBGS* bgs = new DPGrimsonGMMBGS;
  //DPZivkovicAGMMBGS* bgs = new DPZivkovicAGMMBGS;
  //DPMeanBGS* bgs = new DPMeanBGS;
  //DPWrenGABGS* bgs = new DPWrenGABGS;
  //DPPratiMediodBGS* bgs = new DPPratiMediodBGS;
  //DPEigenbackgroundBGS* bgs = new DPEigenbackgroundBGS;

  /*** TB Package (adapted from Thierry Bouwmans) ***/
  //T2FGMM_UM* bgs = new T2FGMM_UM;
  //T2FGMM_UV* bgs = new T2FGMM_UV;
  //FuzzySugenoIntegral* bgs = new FuzzySugenoIntegral;
  //FuzzyChoquetIntegral* bgs = new FuzzyChoquetIntegral;

  /*** JMO Package (adapted from Jean-Marc Odobez) ***/
  //MultiLayerBGS* bgs = new MultiLayerBGS;

  /*** LB Package (adapted from Laurence Bender) ***/
  //LBSimpleGaussian* bgs = new LBSimpleGaussian;
  //LBFuzzyGaussian* bgs = new LBFuzzyGaussian;
  //LBMixtureOfGaussians* bgs = new LBMixtureOfGaussians;
  //LBAdaptiveSOM* bgs = new LBAdaptiveSOM;
  //LBFuzzyAdaptiveSOM* bgs = new LBFuzzyAdaptiveSOM;

  int key = 0;
  while(key != 'q')
  {
    frame = cvQueryFrame(capture);

    if(!frame) break;

    cv::Mat img_input(frame,true);
    cv::GaussianBlur(img_input, img_input, cv::Size(7,7), 1.5);
    cv::imshow("input", img_input);

    cv::Mat img_gray;
    cv::cvtColor(img_input, img_gray, CV_BGR2GRAY);
    cv::imshow("gray", img_gray);
    
    // bgs internally shows the foreground mask image
    cv::Mat img_mask;
    bgs->process(img_gray, img_mask); // default
    //bgs->process(img_input, img_mask); // use it for JMO Package and LB Package
    
    //if(!img_mask.empty())
    //  do something
    
    key = cvWaitKey(1);
  }

  delete bgs;

  cvDestroyAllWindows();
  cvReleaseCapture(&capture);
  
  return 0;
}
