#pragma once

#include "IFrameProcessor.h"
#include "PreProcessor.h"

#include "IBGS.h"

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

#include "T2FGMM_UM.h"
#include "T2FGMM_UV.h"

#include "ForegroundMaskAnalysis.h"

class FrameProcessor : public IFrameProcessor
{
private:
  bool firstTime;
  long frameNumber;
  std::string processname;
  double duration;
  std::string tictoc;

  cv::Mat img_prep;
  PreProcessor* preProcessor;
  bool enablePreProcessor;
  
  cv::Mat img_framediff;
  FrameDifferenceBGS* frameDifference;
  bool enableFrameDifferenceBGS;
  
  cv::Mat img_staticfdiff;
  StaticFrameDifferenceBGS* staticFrameDifference;
  bool enableStaticFrameDifferenceBGS;

  cv::Mat img_wmovmean;
  WeightedMovingMeanBGS* weightedMovingMean;
  bool enableWeightedMovingMeanBGS;

  cv::Mat img_movvar;
  WeightedMovingVarianceBGS* weightedMovingVariance;
  bool enableWeightedMovingVarianceBGS;

  cv::Mat img_mog1;
  MixtureOfGaussianV1BGS* mixtureOfGaussianV1BGS;
  bool enableMixtureOfGaussianV1BGS;

  cv::Mat img_mog2;
  MixtureOfGaussianV2BGS* mixtureOfGaussianV2BGS;
  bool enableMixtureOfGaussianV2BGS;

  cv::Mat img_bkgl_fgmask;
  AdaptiveBackgroundLearning* adaptiveBackgroundLearning;
  bool enableAdaptiveBackgroundLearning;

  cv::Mat img_adpmed;
  DPAdaptiveMedianBGS* adaptiveMedian;
  bool enableDPAdaptiveMedianBGS;

  cv::Mat img_grigmm;
  DPGrimsonGMMBGS* grimsonGMM;
  bool enableDPGrimsonGMMBGS;

  cv::Mat img_zivgmm;
  DPZivkovicAGMMBGS* zivkovicAGMM;
  bool enableDPZivkovicAGMMBGS;

  cv::Mat img_tmpmean;
  DPMeanBGS* temporalMean;
  bool enableDPMeanBGS;

  cv::Mat img_wrenga;
  DPWrenGABGS* wrenGA;
  bool enableDPWrenGABGS;

  cv::Mat img_pramed;
  DPPratiMediodBGS* pratiMediod;
  bool enableDPPratiMediodBGS;

  cv::Mat img_eigbkg;
  DPEigenbackgroundBGS* eigenBackground;
  bool enableDPEigenbackgroundBGS;

  cv::Mat img_t2fgmm_um;
  T2FGMM_UM* type2FuzzyGMM_UM;
  bool enableT2FGMM_UM;

  cv::Mat img_t2fgmm_uv;
  T2FGMM_UV* type2FuzzyGMM_UV;
  bool enableT2FGMM_UV;

  ForegroundMaskAnalysis* foregroundMaskAnalysis;
  bool enableForegroundMaskAnalysis;

public:
  FrameProcessor();
  ~FrameProcessor();

  long frameToStop;
  std::string imgref;

  void process(const cv::Mat &img_input);

private:
  void process(std::string name, IBGS *bgs, const cv::Mat &img_input, cv::Mat &img_bgs);
  void tic(std::string value);
  void toc();

  void saveConfig();
  void loadConfig();
};

