#pragma once
#include "common.hpp"

std::vector<uint8_t> load_file(const std::string& file);

class InferenceInterface {

public:
	InferenceInterface(const std::string& weights_file, const std::string& modelConfiguration, bool use_gpu = false) {

	}

	InferenceInterface(const void* model_data, size_t model_data_length, const std::string& modelConfiguration, bool use_gpu = false) {

	}

	//static void SetLogger(const std::shared_ptr<spdlog::logger>& logger) 
	//{
	//    logger_ = logger;
	//}


	virtual std::tuple<std::vector<std::vector<std::any>>, std::vector<std::vector<int64_t>>> get_infer_results(const cv::Mat& input_blob) = 0;

protected:
	std::vector<float> blob2vec(const cv::Mat& input_blob);
	size_t get_blob_size(const cv::Mat& input_blob);
	//static std::shared_ptr<spdlog::logger> logger_; 



};