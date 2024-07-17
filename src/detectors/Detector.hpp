#pragma once
#include "common.hpp"

struct Detection {
	cv::Rect bbox;
	float score;
	int label;
};

class Detector {
protected:
	float confidenceThreshold_;
	float nms_threshold_ = 0.4f;
	size_t network_width_;
	size_t network_height_;
	std::string backend_;
	int channels_{ -1 };

	cv::Rect get_rect(const cv::Size& imgSz, const std::vector<float>& bbox);

public:
	Detector(float confidenceThreshold = 0.5f, size_t network_width = -1, size_t network_height = -1) : confidenceThreshold_{ confidenceThreshold }, network_width_{ network_width }, network_height_{ network_height }
	{
	}

	virtual std::vector<Detection> postprocess(const std::vector<std::vector<std::any>>& outputs, const std::vector<std::vector<int64_t>>& shapes, const cv::Size& frame_size) = 0;
	virtual cv::Mat preprocess_image(const cv::Mat& image) = 0;
	void set_shapes(size_t w, size_t h) {
		network_width_ = w; network_height_ = h;
	}
};
