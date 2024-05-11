#pragma once
#include "common.hpp"

std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v) {
	const int h_i = static_cast<int>(h * 6);
	const float f = h * 6 - h_i;
	const float p = v * (1 - s);
	const float q = v * (1 - f * s);
	const float t = v * (1 - (1 - f) * s);
	float r, g, b;
	switch (h_i)
	{
	case 0:
		r = v, g = t, b = p;
		break;
	case 1:
		r = q, g = v, b = p;
		break;
	case 2:
		r = p, g = v, b = t;
		break;
	case 3:
		r = p, g = q, b = v;
		break;
	case 4:
		r = t, g = p, b = v;
		break;
	case 5:
		r = v, g = p, b = q;
		break;
	default:
		r = 1, g = 1, b = 1;
		break;
	}
	return std::make_tuple(static_cast<uint8_t>(b * 255), static_cast<uint8_t>(g * 255),
		static_cast<uint8_t>(r * 255));
}

cv::Scalar id2color(int id) {
	float h_plane = ((((unsigned int)id << 2) ^ 0x937151) % 100) / 100.0f;
	float s_plane = ((((unsigned int)id << 3) ^ 0x315793) % 100) / 100.0f;
	auto color_tuple = hsv2bgr(h_plane, s_plane, 1);
	int blue, green, red;
	std::tie(blue, green, red) = color_tuple;
	cv::Scalar color(blue, green, red);
	return color;
}

bool isDirectory(const std::string& path) {
	std::filesystem::path fsPath(path);
	return std::filesystem::is_directory(fsPath);
}

bool isFile(const std::string& path) {
	return std::filesystem::exists(path);
}


void draw_label(cv::Mat& input_image, const std::string& label, float confidence, int left, int top) {
	const float FONT_SCALE = 0.7;
	const int FONT_FACE = cv::FONT_HERSHEY_DUPLEX; // Change font type to what you think is better for you
	const int THICKNESS = 2;
	cv::Scalar YELLOW = cv::Scalar(0, 255, 255);

	// Display the label and confidence at the top of the bounding box.
	int baseLine;
	std::string display_text = label + ": " + std::to_string(confidence);
	cv::Size label_size = cv::getTextSize(display_text, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
	top = cv::max(top, label_size.height);

	// Top left corner.
	cv::Point tlc = cv::Point(left, top);
	// Bottom right corner.
	cv::Point brc = cv::Point(left + label_size.width, top + label_size.height + baseLine);

	// Draw black rectangle.
	cv::rectangle(input_image, tlc, brc, cv::Scalar(255, 0, 255), cv::FILLED);

	// Put the label and confidence on the black rectangle.
	cv::putText(input_image, display_text, cv::Point(left, top + label_size.height), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS);
}


std::vector<std::string> readLabelNames(const std::string& fileName) {
	if (!std::filesystem::exists(fileName)) {
		std::cerr << "Wrong path to labels " << fileName << std::endl;
		exit(1);
	}
	std::vector<std::string> classes;
	std::ifstream ifs(fileName.c_str());
	std::string line;
	while (getline(ifs, line))
		classes.push_back(line);
	return classes;
}

std::string getFileExtension(const std::string& filename) {
	size_t dotPos = filename.find_last_of(".");
	if (dotPos != std::string::npos) {
		return filename.substr(dotPos + 1);
	}
	return ""; // Return empty string if no extension found
}