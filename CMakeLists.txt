cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(demo)

set(TENSORFLOW_ROOT_DIR "/home/pi/xinrui/tensorflow")

find_package(OpenCV REQUIRED)

include_directories( "${TENSORFLOW_ROOT_DIR}/tensorflow/lite/tools/make/downloads/flatbuffers/include" )

include_directories( "${TENSORFLOW_ROOT_DIR}" )
include_directories( ${OpenCV_INCLUDE_DIRS} )

set (
	CMAKE_CXX_FLGS
	"${CMAKE_CXX_FLGS}"
	"-Wall -Werror -O3"
	)

add_executable(demo minimal.cc)
target_link_libraries(demo pthread ${OpenCV_LIBS})
target_link_libraries(demo "${CMAKE_SOURCE_DIR}/libtensorflow-lite.a")
set_property(TARGET demo PROPERTY CXX_STANDARD 11)
