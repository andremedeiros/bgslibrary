#
# HOW TO COMPILE ON LINUX
#
# Requirements:
# cmake >= 2.8
# opencv >= 2.3.1

cd build
cmake ..
make
cd ..

chmod +x use_video.sh use_camera.sh
./use_video.sh
./use_camera.sh
