cmake_minimum_required(VERSION 2.8)

PROJECT (RecalageImage)

find_package(OpenCV  REQUIRED)


file(GLOB RecalageImage_SRCS
    "*.h"
    "*.cpp")
ADD_EXECUTABLE (RecalageImage ${RecalageImage_SRCS})



if (OpenCV_FOUND)
	include_directories( ${OpenCV_INCLUDE_DIRS} )
	target_link_libraries( RecalageImage ${OpenCV_LIBS} )
else (OpenCV_FOUND)
message("PB->OPENCV = ${OpenCV_INCLUDE_DIRS}")
endif (OpenCV_FOUND)

