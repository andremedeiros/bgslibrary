#include "FrameProcessor.h"

FrameProcessor::FrameProcessor() : firstTime(true), frameNumber(0), duration(0), tictoc("")
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

  if(enableT2FGMM_UM)
    type2FuzzyGMM_UM = new T2FGMM_UM;

  if(enableT2FGMM_UV)
    type2FuzzyGMM_UV = new T2FGMM_UV;
}

FrameProcessor::~FrameProcessor()
{
  std::cout << "~FrameProcessor()" << std::endl;
  
  if(enableT2FGMM_UV)
    delete type2FuzzyGMM_UV;

  if(enableT2FGMM_UM)
    delete type2FuzzyGMM_UM;

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
    if(tictoc == "PreProcessor")
      tic("PreProcessor");

    preProcessor->process(img_input, img_prep);

    if(tictoc == "PreProcessor")
      toc();
  }
  
  cv::Mat img_framediff;
  if(enableFrameDifferenceBGS)
  {
    if(tictoc == "FrameDifferenceBGS")
      tic("FrameDifferenceBGS");

    frameDifference->process(img_prep,img_framediff);

    if(tictoc == "FrameDifferenceBGS")
      toc();
  }
  
  cv::Mat img_staticfdiff;
  if(enableStaticFrameDifferenceBGS)
  {
    if(firstTime)
      staticFrameDifference->setBackgroundRef(img_prep);
    else
    {
      if(tictoc == "StaticFrameDifferenceBGS")
        tic("StaticFrameDifferenceBGS");

      staticFrameDifference->process(img_prep,img_staticfdiff);

      if(tictoc == "StaticFrameDifferenceBGS")
        toc();
    }
  }
  
  cv::Mat img_wmovmean;
  if(enableWeightedMovingMeanBGS)
  {
    if(tictoc == "WeightedMovingMeanBGS")
      tic("WeightedMovingMeanBGS");

    weightedMovingMean->process(img_prep,img_wmovmean);

    if(tictoc == "WeightedMovingMeanBGS")
      toc();
  }
  
  cv::Mat img_movvar;
  if(enableWeightedMovingVarianceBGS)
  {
    if(tictoc == "WeightedMovingVarianceBGS")
      tic("WeightedMovingVarianceBGS");

    weightedMovingVariance->process(img_prep,img_movvar);

    if(tictoc == "WeightedMovingVarianceBGS")
      toc();
  }
  
  cv::Mat img_mog1;
  if(enableMixtureOfGaussianV1BGS)
  {
    if(tictoc == "MixtureOfGaussianV1BGS")
      tic("MixtureOfGaussianV1BGS");

    mixtureOfGaussianV1BGS->process(img_prep,img_mog1);

    if(tictoc == "MixtureOfGaussianV1BGS")
      toc();
  }
  
  cv::Mat img_mog2;
  if(enableMixtureOfGaussianV2BGS)
  {
    if(tictoc == "MixtureOfGaussianV2BGS")
      tic("MixtureOfGaussianV2BGS");

    mixtureOfGaussianV2BGS->process(img_prep,img_mog2);

    if(tictoc == "MixtureOfGaussianV2BGS")
      toc();
  }
  
  cv::Mat img_bkgl_fgmask;
  if(enableAdaptiveBackgroundLearning)
  {
    if(tictoc == "AdaptiveBackgroundLearning")
      tic("AdaptiveBackgroundLearning");

    adaptiveBackgroundLearning->process(img_prep,img_bkgl_fgmask);

    if(tictoc == "AdaptiveBackgroundLearning")
      toc();
  }

  cv::Mat img_adpmed;
  if(enableDPAdaptiveMedianBGS)
  {
    if(tictoc == "DPAdaptiveMedianBGS")
      tic("DPAdaptiveMedianBGS");

    adaptiveMedian->process(img_prep,img_adpmed);

    if(tictoc == "DPAdaptiveMedianBGS")
      toc();
  }
  
  cv::Mat img_grigmm;
  if(enableDPGrimsonGMMBGS)
  {
    if(tictoc == "DPGrimsonGMMBGS")
      tic("DPGrimsonGMMBGS");

    grimsonGMM->process(img_prep,img_grigmm);

    if(tictoc == "DPGrimsonGMMBGS")
      toc();
  }
  
  cv::Mat img_zivgmm;
  if(enableDPZivkovicAGMMBGS)
  {
    if(tictoc == "DPZivkovicAGMMBGS")
      tic("DPZivkovicAGMMBGS");

    zivkovicAGMM->process(img_prep,img_zivgmm);

    if(tictoc == "DPZivkovicAGMMBGS")
      toc();
  }
  
  cv::Mat img_tmpmean;
  if(enableDPMeanBGS)
  {
    if(tictoc == "DPMeanBGS")
      tic("DPMeanBGS");

    temporalMean->process(img_prep,img_tmpmean);

    if(tictoc == "DPMeanBGS")
      toc();
  }
  
  cv::Mat img_wrenga;
  if(enableDPWrenGABGS)
  {
    if(tictoc == "DPWrenGABGS")
      tic("DPWrenGABGS");

    wrenGA->process(img_prep,img_wrenga);

    if(tictoc == "DPWrenGABGS")
      toc();
  }
  
  cv::Mat img_pramed;
  if(enableDPPratiMediodBGS)
  {
    if(tictoc == "DPPratiMediodBGS")
      tic("DPPratiMediodBGS");

    pratiMediod->process(img_prep,img_pramed);

    if(tictoc == "DPPratiMediodBGS")
      toc();
  }
  
  cv::Mat img_eigbkg;
  if(enableDPEigenbackgroundBGS)
  {
    if(tictoc == "DPEigenbackgroundBGS")
      tic("DPEigenbackgroundBGS");

    eigenBackground->process(img_input,img_eigbkg);

    if(tictoc == "DPEigenbackgroundBGS")
      toc();
  }

  cv::Mat img_t2fgmm_um;
  if(enableT2FGMM_UM)
  {
    if(tictoc == "T2FGMM_UM")
      tic("T2FGMM_UM");

    type2FuzzyGMM_UM->process(img_prep,img_t2fgmm_um);

    if(tictoc == "T2FGMM_UM")
      toc();
  }

  cv::Mat img_t2fgmm_uv;
  if(enableT2FGMM_UV)
  {
    if(tictoc == "T2FGMM_UV")
      tic("T2FGMM_UV");

    type2FuzzyGMM_UV->process(img_prep,img_t2fgmm_uv);

    if(tictoc == "T2FGMM_UV")
      toc();
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
  std::cout << processname << "\ttime(sec):" << std::fixed << std::setprecision(6) << duration << std::endl;
}

void FrameProcessor::saveConfig()
{
  CvFileStorage* fs = cvOpenFileStorage("./config/FrameProcessor.xml", 0, CV_STORAGE_WRITE);

  cvWriteString(fs, "tictoc", tictoc.c_str());

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

  cvWriteInt(fs, "enableT2FGMM_UM", enableT2FGMM_UM);
  cvWriteInt(fs, "enableT2FGMM_UV", enableT2FGMM_UV);

  cvReleaseFileStorage(&fs);
}

void FrameProcessor::loadConfig()
{
  CvFileStorage* fs = cvOpenFileStorage("./config/FrameProcessor.xml", 0, CV_STORAGE_READ);
  
  tictoc = cvReadStringByName(fs, 0, "tictoc", "");

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

  enableT2FGMM_UM = cvReadIntByName(fs, 0, "enableT2FGMM_UM", true);
  enableT2FGMM_UV = cvReadIntByName(fs, 0, "enableT2FGMM_UV", true);

  cvReleaseFileStorage(&fs);
}