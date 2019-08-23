cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(demo)

set(TENSORFLOW_ROOT_DIR "/home/pi/xinrui/tensorflow")

include_directories( "${TENSORFLOW_ROOT_DIR}/tensorflow/lite/tools/make/downloads/flatbuffers/include" )

include_directories( "${TENSORFLOW_ROOT_DIR}" )

set (
	CMAKE_CXX_FLGS
	"${CMAKE_CXX_FLGS}"
	"-Wall -Werror -O3"
	)

add_executable(demo minimal.cc)
target_link_libraries(demo pthread)
target_link_libraries(demo "${CMAKE_SOURCE_DIR}/libtensorflow-lite.a")
set_property(TARGET demo PROPERTY CXX_STANDARD 11)
