#include "InferenceInterface.hpp"

std::vector<float> InferenceInterface::blob2vec(const cv::Mat& input_blob) {
	const auto channels = input_blob.size[1];
	const auto network_width = input_blob.size[2];
	const auto network_height = input_blob.size[3];
	size_t img_byte_size = network_width * network_height * channels * sizeof(float);  // Allocate a buffer to hold all image elements.
	std::vector<float> input_data = std::vector<float>(network_width * network_height * channels);
	std::memcpy(input_data.data(), input_blob.data, img_byte_size);

	std::vector<cv::Mat> chw;
	for (size_t i = 0; i < channels; ++i) {
		chw.emplace_back(cv::Mat(cv::Size(network_width, network_height), CV_32FC1, &(input_data[i * network_width * network_height])));
	}
	cv::split(input_blob, chw);

	return input_data;
}

size_t InferenceInterface::get_blob_size(const cv::Mat& input_blob) {
	size_t total_size = 1;
	for (size_t i = 0; i < input_blob.size.dims(); i++)
		total_size *= input_blob.size[i];
	return total_size;
}

std::vector<uint8_t> load_file(const std::string& file) {
	std::ifstream in(file, std::ios::in | std::ios::binary);
	if (!in.is_open()) return {};

	in.seekg(0, std::ios::end);
	size_t length = in.tellg();

	std::vector<uint8_t> data;
	if (length > 0) {
		in.seekg(0, std::ios::beg);
		data.resize(length);

		in.read((char*)&data[0], length);
	}
	in.close();
	return data;
}