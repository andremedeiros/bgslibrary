#include "VideoAnalysis.h"

VideoAnalysis::VideoAnalysis() : use_file(false), use_camera(false), cameraIndex(0)
{
  std::cout << "VideoAnalysis()" << std::endl;
}

VideoAnalysis::~VideoAnalysis()
{
  std::cout << "~VideoAnalysis()" << std::endl;
}

bool VideoAnalysis::setup(int argc, const char **argv)
{
  bool flag = false;
  
  const char* keys =
  "{hp|help|false|Print help message}"
  "{uf|use_file|false|Use video file}"
  "{fn|filename||Specify video file}"
  "{uc|use_cam|false|Use camera}"
  "{ca|camera|0|Specify camera index}"
  ;
  cv::CommandLineParser cmd(argc, argv, keys);
  
  if(argc <= 1 || cmd.get<bool>("help") == true)
  {
    std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
    std::cout << "Avaible options:" << std::endl;
    cmd.printParams();
    return false;
  }

  use_file = cmd.get<bool>("use_file");
  use_camera = cmd.get<bool>("use_cam");

  if(use_file)
  {
    filename = cmd.get<std::string>("filename");

    if(filename.empty())
    {
      std::cout << "Specify filename"<< std::endl;
      return false;
    }
    
    flag = true;
  }

  if(use_camera)
  {
    cameraIndex = cmd.get<int>("camera");
    flag = true;
  }

  return flag;
}

void VideoAnalysis::start()
{
  do
  {
    videoCapture = new VideoCapture;
    frameProcessor = new FrameProcessor;

    videoCapture->setFrameProcessor(frameProcessor);

    if(use_file)
      videoCapture->setVideo(filename);
    
    if(use_camera)
      videoCapture->setCamera(cameraIndex);
    
    videoCapture->start();

    if(use_file || use_camera)
      break;

    int key = cvWaitKey(500);
    if(key == KEY_ESC)
      break;

    delete frameProcessor;
    delete videoCapture;

  }while(1);
  
  delete frameProcessor;
  delete videoCapture;
}