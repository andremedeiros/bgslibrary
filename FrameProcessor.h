#pragma once

#include "IFrameProcessor.h"
#include "PreProcessor.h"

#include "FrameDifferenceBGS.h"
#include "StaticFrameDifferenceBGS.h"
#include "WeightedMovingMeanBGS.h"
#include "WeightedMovingVarianceBGS.h"
#include "MixtureOfGaussianV1BGS.h"
#include "MixtureOfGaussianV2BGS.h"
#include "AdaptiveBackgroundLearning.h"

#include "DPAdaptiveMedianBGS.h"
#include "DPGrimsonGMMBGS.h"
#include "DPZivkovicAGMMBGS.h"
#include "DPMeanBGS.h"
#include "DPWrenGABGS.h"
#include "DPPratiMediodBGS.h"
#include "DPEigenbackgroundBGS.h"

class FrameProcessor : public IFrameProcessor
{
private:
  bool firstTime;
  long frameNumber;
  std::string processname;
  double duration;
  
  void saveConfig();
  void loadConfig();

  PreProcessor* preProcessor;
  bool enablePreProcessor;
  
  FrameDifferenceBGS* frameDifference;
  bool enableFrameDifferenceBGS;

  StaticFrameDifferenceBGS* staticFrameDifference;
  bool enableStaticFrameDifferenceBGS;

  WeightedMovingMeanBGS* weightedMovingMean;
  bool enableWeightedMovingMeanBGS;

  WeightedMovingVarianceBGS* weightedMovingVariance;
  bool enableWeightedMovingVarianceBGS;

  MixtureOfGaussianV1BGS* mixtureOfGaussianV1BGS;
  bool enableMixtureOfGaussianV1BGS;

  MixtureOfGaussianV2BGS* mixtureOfGaussianV2BGS;
  bool enableMixtureOfGaussianV2BGS;

  AdaptiveBackgroundLearning* adaptiveBackgroundLearning;
  bool enableAdaptiveBackgroundLearning;

  DPAdaptiveMedianBGS* adaptiveMedian;
  bool enableDPAdaptiveMedianBGS;

  DPGrimsonGMMBGS* grimsonGMM;
  bool enableDPGrimsonGMMBGS;

  DPZivkovicAGMMBGS* zivkovicAGMM;
  bool enableDPZivkovicAGMMBGS;

  DPMeanBGS* temporalMean;
  bool enableDPMeanBGS;

  DPWrenGABGS* wrenGA;
  bool enableDPWrenGABGS;

  DPPratiMediodBGS* pratiMediod;
  bool enableDPPratiMediodBGS;

  DPEigenbackgroundBGS* eigenBackground;
  bool enableDPEigenbackgroundBGS;

public:
  FrameProcessor();
  ~FrameProcessor();

  void process(const cv::Mat &img_input);
  void tic(std::string value);
  void toc();
};

