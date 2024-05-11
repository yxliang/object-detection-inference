#pragma once
#include "common.hpp"
#include "InferenceInterface.hpp"
#ifdef USE_ONNX_RUNTIME
#include "ORTInfer.hpp"
#elif USE_LIBTORCH 
#include "LibtorchInfer.hpp"
#elif USE_LIBTENSORFLOW 
#include "TFDetectionAPI.hpp"
#elif USE_OPENCV_DNN 
#include "OCVDNNInfer.hpp"
#elif USE_TENSORRT
#include "TRTInfer.hpp"
#elif USE_OPENVINO
#include "OVInfer.hpp"
#endif

std::unique_ptr<InferenceInterface> setup_inference_engine(const std::string& weights, const std::string& modelConfiguration, bool use_gpu = false) {
#ifdef USE_ONNX_RUNTIME
	return std::make_unique<ORTInfer>(weights, use_gpu);
#elif USE_LIBTORCH 
	return std::make_unique<LibtorchInfer>(weights, use_gpu);
#elif USE_LIBTENSORFLOW 
	return std::make_unique<TFDetectionAPI>(weights, use_gpu);
#elif USE_OPENCV_DNN 
	return std::make_unique<OCVDNNInfer>(weights, modelConfiguration);
#elif USE_TENSORRT
	return std::make_unique<TRTInfer>(weights);
#elif USE_OPENVINO
	return std::make_unique<OVInfer>("", modelConfiguration, use_gpu);
#endif
	return nullptr;


}