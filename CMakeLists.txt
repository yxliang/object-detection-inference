cmake_minimum_required(VERSION 3.20)
project(object-detection-inference)

set(CMAKE_CXX_STANDARD 17)

SET(3rdlib_DIR  "D:/Libraries")

#eigen3
SET(Eigen3_DIR "${3rdlib_DIR}/Eigen3/cmake")
FIND_PACKAGE(Eigen3)	

#opencv
SET(OpenCV_DIR "${3rdlib_DIR}/OpenCV")
find_package(OpenCV REQUIRED)

#gflags
ADD_LIBRARY(gflags STATIC IMPORTED GLOBAL)
SET_TARGET_PROPERTIES(gflags PROPERTIES IMPORTED_LOCATION_DEBUG "${3rdlib_DIR}/gflags/lib/gflags_static_debug.lib" 
										IMPORTED_LOCATION_RELEASE "${3rdlib_DIR}/gflags/lib/gflags_static.lib" 
										INTERFACE_COMPILE_DEFINITIONS "GFLAGS_IS_A_DLL=0"
										INTERFACE_INCLUDE_DIRECTORIES "${3rdlib_DIR}/gflags/include"
										INTERFACE_LINK_LIBRARIES "shlwapi.lib")
LIST(APPEND 3rd_LIBS gflags)
										
#glog
ADD_LIBRARY(glog STATIC IMPORTED GLOBAL)
SET_TARGET_PROPERTIES(glog PROPERTIES IMPORTED_LOCATION_DEBUG "${3rdlib_DIR}/glog/lib/glogd.lib"
									  IMPORTED_LOCATION_RELEASE "${3rdlib_DIR}/glog/lib/glog.lib"
									  INTERFACE_COMPILE_DEFINITIONS "GFLAGS_DLL_DECLARE_FLAG=;GFLAGS_DLL_DEFINE_FLAG=;GLOG_NO_ABBREVIATED_SEVERITIES;GOOGLE_GLOG_DLL_DECL="
									  INTERFACE_INCLUDE_DIRECTORIES "${3rdlib_DIR}/glog/include")

# SET(spdlog_DIR "${3rdlib_DIR}/spdlog/lib/cmake/spdlog")
# find_package(spdlog REQUIRED)
# message("spdlog_DIR: ${spdlog_DIR}")

# Define an option to enable or disable GStreamer support
unset(USE_GSTREAMER CACHE)
option(USE_GSTREAMER "Use GStreamer for video capture (optional)" OFF)

option(BUILD_TESTS "Build test target" OFF) # Work in progress...
# option(BUILD_BENCHMARKS "Build benchmark target" OFF) TODO

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
message(STATUS ${CMAKE_MODULE_PATH})

# Set the path to the selected framework (modify accordingly)
set(DEFAULT_BACKEND "ONNX_RUNTIME")  # Options: ONNX_RUNTIME, LIBTORCH, TENSORRT, OPENVINO, LIBTENSORFLOW, OPENCV_DNN

# Define the supported backends
set(SUPPORTED_BACKENDS "ONNX_RUNTIME" "LIBTORCH" "LIBTENSORFLOW" "OPENCV_DNN" "TENSORRT" "OPENVINO")

# Check if the specified backend is supported
list(FIND SUPPORTED_BACKENDS ${DEFAULT_BACKEND} SUPPORTED_BACKEND_INDEX)
if (SUPPORTED_BACKEND_INDEX EQUAL -1)
    message(STATUS "Unsupported default backend: ${DEFAULT_BACKEND}")
    set(DEFAULT_BACKEND "OPENCV_DNN")
endif()

message(STATUS "Default backend: ${DEFAULT_BACKEND}")

set(DETECTORS_ROOT src/detectors)
set(DETECTORS_SOURCES 
    ${DETECTORS_ROOT}/Detector.cpp 
    ${DETECTORS_ROOT}/YoloNas.cpp 
    ${DETECTORS_ROOT}/RtDetr.cpp 
    ${DETECTORS_ROOT}/RtDetrUltralytics.cpp 
    ${DETECTORS_ROOT}/YoloV4.cpp 
    ${DETECTORS_ROOT}/YoloVn.cpp
    )

set(SOURCES main.cpp src/inference-engines/InferenceInterface.cpp ${DETECTORS_SOURCES})

# Include GStreamer-related settings and source files if USE_GSTREAMER is ON
if (USE_GSTREAMER)
    include(GStreamer)
endif()

include(SelectBackend)

if(BUILD_TESTS) # Work in progress ...
    message(STATUS "Test enabled")
    find_package(GTest REQUIRED)
    enable_testing()
    add_subdirectory(tests)  # Create a "tests" subdirectory for your test CMakeLists.txt
endif()

add_executable(${PROJECT_NAME} ${SOURCES})

# Include directories and link libraries
target_include_directories(${PROJECT_NAME} PRIVATE
    inc
    src
    src/detectors
    src/inference-engines
    src/videocapture
    ${OpenCV_INCLUDE_DIRS}
    # ${spdlog_INCLUDE_DIRS}
)


# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE gflags glog ${OpenCV_LIBS}
)
				  
# Link against GStreamer libraries if USE_GSTREAMER is ON
if (USE_GSTREAMER)
    target_include_directories(${PROJECT_NAME} PRIVATE
        ${GSTREAMER_INCLUDE_DIRS}
        ${GST_APP_INCLUDE_DIRS}
        ${GST_VIDEO_INCLUDE_DIRS}
    )
    target_link_libraries(${PROJECT_NAME} PRIVATE
        ${GSTREAMER_LIBRARIES}
        ${GST_APP_LIBRARIES}
        ${GST_VIDEO_LIBRARIES}
    )
endif()

include(LinkBackend)

# Set the appropriate compiler flags
include(SetCompilerFlags)


if (MSVC)
	SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES
				  RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin
				  RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin
				  DEBUG_POSTFIX "d"
				  )
				  
	if (DEFAULT_BACKEND STREQUAL "ONNX_RUNTIME")
	FILE(GLOB ORT_DLLS "${3rdlib_DIR}/onnxruntime/lib/*.dll")
	ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME}
						POST_BUILD
						COMMAND ${CMAKE_COMMAND} -E copy_if_different
						${ORT_DLLS}
						$<TARGET_FILE_DIR:${PROJECT_NAME}>)
	endif()
endif (MSVC)

################################################################################
set(SRC_LIST Task_specific_main.cpp src/inference-engines/InferenceInterface.cpp ${DETECTORS_SOURCES})
if (DEFAULT_BACKEND STREQUAL "ONNX_RUNTIME")
	list(APPEND SRC_LIST ${ONNX_RUNTIME_SOURCES})
	add_compile_definitions(USE_ONNX_RUNTIME)
endif()

SET(TestName "TestSpecificTask")
ADD_EXECUTABLE(${TestName} ${SRC_LIST})
TARGET_INCLUDE_DIRECTORIES(${TestName} PRIVATE inc src src/detectors src/inference-engines  src/videocapture ${OpenCV_INCLUDE_DIRS})
TARGET_LINK_LIBRARIES(${TestName} PRIVATE gflags glog ${OpenCV_LIBS})

if (DEFAULT_BACKEND STREQUAL "ONNX_RUNTIME")
    TARGET_INCLUDE_DIRECTORIES(${TestName} PRIVATE ${ONNX_RUNTIME_DIR}/include src/inference-engines/onnx-runtime)
    TARGET_LINK_DIRECTORIES(${TestName} PRIVATE ${ONNX_RUNTIME_DIR}/lib)
    TARGET_LINK_LIBRARIES(${TestName} PRIVATE ${ONNX_RUNTIME_DIR}/lib/onnxruntime.lib)
endif()


if (MSVC)
	SET_TARGET_PROPERTIES(${TestName} PROPERTIES
				  RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin
				  RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin
				  DEBUG_POSTFIX "d"
				  )
endif (MSVC)
