# file CMakeLists.txt

cmake_minimum_required(VERSION 2.8)

project(bgs)

set(CMAKE_CXX_FLAGS "${CMAKE_CSS_FLAGS} -std=gnu++0x")

find_package(OpenCV REQUIRED)

if (${OpenCV_VERSION} VERSION_LESS 2.3.1)
  message (FATAL_ERROR "OpenCV version is not compatible: ${OpenCV_VERSION}")
endif()

file(GLOB sources *.cpp)

file(GLOB_RECURSE analysis package_analysis/*.cpp)
file(GLOB_RECURSE bgs package_bgs/*.cpp)

include_directories(
	${CMAKE_SOURCE_DIR}
)

add_executable(bgs
  ${sources}
  ${analysis}
  ${bgs}
)

target_link_libraries(bgs
  ${OpenCV_LIBS}
)
