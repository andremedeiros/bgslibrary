#include "FrameProcessor.h"

FrameProcessor::FrameProcessor() : firstTime(true), frameNumber(0), duration(0)
{
  std::cout << "FrameProcessor()" << std::endl;

  loadConfig();
  saveConfig();
  
  if(enablePreProcessor)
    preProcessor = new PreProcessor;

  if(enableFrameDifferenceBGS)
    frameDifference = new FrameDifferenceBGS;

  if(enableStaticFrameDifferenceBGS)
    staticFrameDifference = new StaticFrameDifferenceBGS;

  if(enableWeightedMovingMeanBGS)
    weightedMovingMean = new WeightedMovingMeanBGS;

  if(enableWeightedMovingVarianceBGS)
    weightedMovingVariance = new WeightedMovingVarianceBGS;

  if(enableMixtureOfGaussianV1BGS)
    mixtureOfGaussianV1BGS = new MixtureOfGaussianV1BGS;

  if(enableMixtureOfGaussianV2BGS)
    mixtureOfGaussianV2BGS = new MixtureOfGaussianV2BGS;

  if(enableAdaptiveBackgroundLearning)
    adaptiveBackgroundLearning = new AdaptiveBackgroundLearning;

  if(enableDPAdaptiveMedianBGS)
    adaptiveMedian = new DPAdaptiveMedianBGS;

  if(enableDPGrimsonGMMBGS)
    grimsonGMM = new DPGrimsonGMMBGS;

  if(enableDPZivkovicAGMMBGS)
    zivkovicAGMM = new DPZivkovicAGMMBGS;

  if(enableDPMeanBGS)
    temporalMean = new DPMeanBGS;

  if(enableDPWrenGABGS)
    wrenGA = new DPWrenGABGS;

  if(enableDPPratiMediodBGS)
    pratiMediod = new DPPratiMediodBGS;

  if(enableDPEigenbackgroundBGS)
    eigenBackground = new DPEigenbackgroundBGS;
}

FrameProcessor::~FrameProcessor()
{
  std::cout << "~FrameProcessor()" << std::endl;
  
  if(enableDPEigenbackgroundBGS)
    delete eigenBackground;

  if(enableDPPratiMediodBGS)
    delete pratiMediod;

  if(enableDPWrenGABGS)
    delete wrenGA;

  if(enableDPMeanBGS)
    delete temporalMean;

  if(enableDPZivkovicAGMMBGS)
    delete zivkovicAGMM;

  if(enableDPGrimsonGMMBGS)
    delete grimsonGMM;

  if(enableDPAdaptiveMedianBGS)
    delete adaptiveMedian;

  if(enableAdaptiveBackgroundLearning)
    delete adaptiveBackgroundLearning;

  if(enableMixtureOfGaussianV2BGS)
    delete mixtureOfGaussianV2BGS;

  if(enableMixtureOfGaussianV1BGS)
    delete mixtureOfGaussianV1BGS;

  if(enableWeightedMovingVarianceBGS)
    delete weightedMovingVariance;

  if(enableWeightedMovingMeanBGS)
    delete weightedMovingMean;

  if(enableStaticFrameDifferenceBGS)
    delete staticFrameDifference;

  if(enableFrameDifferenceBGS)
    delete frameDifference;

  if(enablePreProcessor)
    delete preProcessor;
}

void FrameProcessor::process(const cv::Mat &img_input)
{
  cv::Mat img_prep;
  if(enablePreProcessor)
  {
    //tic("PreProcessor");
    preProcessor->process(img_input, img_prep);
    //toc();
  }
  
  cv::Mat img_framediff;
  if(enableFrameDifferenceBGS)
  {
    //tic("FrameDifferenceBGS");
    frameDifference->process(img_prep,img_framediff);
    //toc();
  }
  
  cv::Mat img_staticfdiff;
  if(enableStaticFrameDifferenceBGS)
  {
    if(firstTime)
      staticFrameDifference->setBackgroundRef(img_prep);
    else
    {
      //tic("StaticFrameDifferenceBGS");
      staticFrameDifference->process(img_prep,img_staticfdiff);
      //toc();
    }
  }
  
  cv::Mat img_wmovmean;
  if(enableWeightedMovingMeanBGS)
  {
    //tic("WeightedMovingMeanBGS");
    weightedMovingMean->process(img_prep,img_wmovmean);
    //toc();
  }
  
  cv::Mat img_movvar;
  if(enableWeightedMovingVarianceBGS)
  {
    //tic("WeightedMovingVarianceBGS");
    //weightedMovingVariance->setEnableThreshold(false);
    weightedMovingVariance->process(img_prep,img_movvar);
    //toc();
  }
  
  cv::Mat img_mog1;
  if(enableMixtureOfGaussianV1BGS)
  {
    //tic("MixtureOfGaussianV1BGS");
    mixtureOfGaussianV1BGS->process(img_prep,img_mog1);
    //toc();
  }
  
  cv::Mat img_mog2;
  if(enableMixtureOfGaussianV2BGS)
  {
    //tic("MixtureOfGaussianV2BGS");
    mixtureOfGaussianV2BGS->process(img_prep,img_mog2);
    //toc();
  }
  
  cv::Mat img_bkgl_fgmask;
  if(enableAdaptiveBackgroundLearning)
  {
    //tic("AdaptiveBackgroundLearning");
    adaptiveBackgroundLearning->process(img_prep,img_bkgl_fgmask);
    //toc();
  }

  cv::Mat img_adpmed;
  if(enableDPAdaptiveMedianBGS)
  {
    //tic("DPAdaptiveMedianBGS");
    adaptiveMedian->process(img_prep,img_adpmed);
    //toc();
  }
  
  cv::Mat img_grigmm;
  if(enableDPGrimsonGMMBGS)
  {
    //tic("DPGrimsonGMMBGS");
    grimsonGMM->process(img_prep,img_grigmm);
    //toc();
  }
  
  cv::Mat img_zivgmm;
  if(enableDPZivkovicAGMMBGS)
  {
    //tic("DPZivkovicAGMMBGS");
    zivkovicAGMM->process(img_prep,img_zivgmm);
    //toc();
  }
  
  cv::Mat img_tmpmean;
  if(enableDPMeanBGS)
  {
    //tic("DPMeanBGS");
    temporalMean->process(img_prep,img_tmpmean);
    //toc();
  }
  
  cv::Mat img_wrenga;
  if(enableDPWrenGABGS)
  {
    //tic("DPWrenGABGS");
    wrenGA->process(img_prep,img_wrenga);
    //toc();
  }
  
  cv::Mat img_pramed;
  if(enableDPPratiMediodBGS)
  {
    //tic("DPPratiMediodBGS");
    pratiMediod->process(img_prep,img_pramed);
    //toc();
  }
  
  cv::Mat img_eigbkg;
  if(enableDPEigenbackgroundBGS)
  {
    //tic("DPEigenbackgroundBGS");
    eigenBackground->process(img_input,img_eigbkg);
    //toc();
  }

  firstTime = false;
  frameNumber++;
}

void FrameProcessor::tic(std::string value)
{
  processname = value;
  duration = static_cast<double>(cv::getTickCount());
}

void FrameProcessor::toc()
{
  duration = (static_cast<double>(cv::getTickCount()) - duration)/cv::getTickFrequency();
  std::cout << processname << "\ttime(sec):" << duration << "" << std::endl;
}

void FrameProcessor::saveConfig()
{
  CvFileStorage* fs = cvOpenFileStorage("./config/FrameProcessor.xml", 0, CV_STORAGE_WRITE);

  cvWriteInt(fs, "enablePreProcessor", enablePreProcessor);

  cvWriteInt(fs, "enableFrameDifferenceBGS", enableFrameDifferenceBGS);
  cvWriteInt(fs, "enableStaticFrameDifferenceBGS", enableStaticFrameDifferenceBGS);
  cvWriteInt(fs, "enableWeightedMovingMeanBGS", enableWeightedMovingMeanBGS);
  cvWriteInt(fs, "enableWeightedMovingVarianceBGS", enableWeightedMovingVarianceBGS);
  cvWriteInt(fs, "enableMixtureOfGaussianV1BGS", enableMixtureOfGaussianV1BGS);
  cvWriteInt(fs, "enableMixtureOfGaussianV2BGS", enableMixtureOfGaussianV2BGS);
  cvWriteInt(fs, "enableAdaptiveBackgroundLearning", enableAdaptiveBackgroundLearning);
  
  cvWriteInt(fs, "enableDPAdaptiveMedianBGS", enableDPAdaptiveMedianBGS);
  cvWriteInt(fs, "enableDPGrimsonGMMBGS", enableDPGrimsonGMMBGS);
  cvWriteInt(fs, "enableDPZivkovicAGMMBGS", enableDPZivkovicAGMMBGS);
  cvWriteInt(fs, "enableDPMeanBGS", enableDPMeanBGS);
  cvWriteInt(fs, "enableDPWrenGABGS", enableDPWrenGABGS);
  cvWriteInt(fs, "enableDPPratiMediodBGS", enableDPPratiMediodBGS);
  cvWriteInt(fs, "enableDPEigenbackgroundBGS", enableDPEigenbackgroundBGS);

  cvReleaseFileStorage(&fs);
}

void FrameProcessor::loadConfig()
{
  CvFileStorage* fs = cvOpenFileStorage("./config/FrameProcessor.xml", 0, CV_STORAGE_READ);
  
  enablePreProcessor = cvReadIntByName(fs, 0, "enablePreProcessor", true);
  
  enableFrameDifferenceBGS = cvReadIntByName(fs, 0, "enableFrameDifferenceBGS", true);
  enableStaticFrameDifferenceBGS = cvReadIntByName(fs, 0, "enableStaticFrameDifferenceBGS", true);
  enableWeightedMovingMeanBGS = cvReadIntByName(fs, 0, "enableWeightedMovingMeanBGS", true);
  enableWeightedMovingVarianceBGS = cvReadIntByName(fs, 0, "enableWeightedMovingVarianceBGS", true);
  enableMixtureOfGaussianV1BGS = cvReadIntByName(fs, 0, "enableMixtureOfGaussianV1BGS", true);
  enableMixtureOfGaussianV2BGS = cvReadIntByName(fs, 0, "enableMixtureOfGaussianV2BGS", true);
  enableAdaptiveBackgroundLearning = cvReadIntByName(fs, 0, "enableAdaptiveBackgroundLearning", true);

  enableDPAdaptiveMedianBGS = cvReadIntByName(fs, 0, "enableDPAdaptiveMedianBGS", true);
  enableDPGrimsonGMMBGS = cvReadIntByName(fs, 0, "enableDPGrimsonGMMBGS", true);
  enableDPZivkovicAGMMBGS = cvReadIntByName(fs, 0, "enableDPZivkovicAGMMBGS", true);
  enableDPMeanBGS = cvReadIntByName(fs, 0, "enableDPMeanBGS", true);
  enableDPWrenGABGS = cvReadIntByName(fs, 0, "enableDPWrenGABGS", true);
  enableDPPratiMediodBGS = cvReadIntByName(fs, 0, "enableDPPratiMediodBGS", true);
  enableDPEigenbackgroundBGS = cvReadIntByName(fs, 0, "enableDPEigenbackgroundBGS", true);

  cvReleaseFileStorage(&fs);
}