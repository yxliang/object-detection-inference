#include "VideoCaptureFactory.hpp"
#include "DetectorSetup.hpp"
#include "InferenceBackendSetup.hpp"
#include "Logger.hpp"
#include "utils.hpp"


static const std::string params = "{ help h   |   | print help message }"
"{ type     |  yolov9 | yolov4, yolov5, yolov6, yolov7,yolov8, yolov9, rtdetr, rtdetrul}"
"{ source s   |   | path to image or video source}"
"{ labels lb  |  | path to class labels}"
"{ config c   |   | optional model configuration file}"
"{ weights w  |   | path to models weights}"
"{ use_gpu   | false  | activate gpu support}"
"{ min_confidence | 0.5   | optional min confidence}";


int main(int argc, char* argv[]) {
	DLOG(INFO) << "Initializing application";
	initializeLogger();

	// Command line parser
	cv::CommandLineParser parser(argc, argv, params);
	parser.about("Detect objets from video or image input source");
	if (parser.has("help")) {
		parser.printMessage();
		std::exit(1);
	}

	if (!parser.check()) {
		parser.printErrors();
		std::exit(1);
	}

	std::string source = parser.get<std::string>("source");
	if (source.empty()) {
		LOG(INFO) << "Can not open video stream";
		std::exit(1);
	}
	LOG(INFO) << "Source: " << source;

	const bool use_gpu = parser.get<bool>("use_gpu");
	const std::string config = parser.get<std::string>("config");
	if (!config.empty() && !isFile(config)) {
		LOG(ERROR) << config << " doesn't exist";
		std::exit(1);
	}

	const std::string weights = parser.get<std::string>("weights");
	if (!isFile(weights) && getFileExtension(config) != "xml") {
		LOG(ERROR) << weights << "weights file doesn't exist";
		std::exit(1);
	}
	LOG(INFO) << "Weights: " << weights;

	const std::string labelsPath = parser.get<std::string>("labels");
	if (!isFile(labelsPath)) {
		//logger->error("labels file {} doesn't exist", labelsPath);
		LOG(ERROR) << labelsPath << "labels file doesn't exist";
		std::exit(1);
	}
	LOG(INFO) << "Labels file: " << labelsPath;

	const std::string detectorType = parser.get<std::string>("type");
	LOG(INFO) << "Detector typ: " << detectorType;

	float confidenceThreshold = parser.get<float>("min_confidence");
	std::vector<std::string> classes = readLabelNames(labelsPath);
	LOG(INFO) << "Current path is " << std::filesystem::current_path();

	std::unique_ptr<Detector> detector = createDetector(detectorType);

	if (!detector) {
		LOG(ERROR) << "Can't setup a detector " << detectorType;
		std::exit(1);
	}

	std::unique_ptr<InferenceInterface> engine = setup_inference_engine(weights, config, use_gpu);
	if (!engine) {
		LOG(ERROR) << "Can't setup an inference engine for " << weights << ": " << config;
		std::exit(1);
	}

	if (!detector) {
		LOG(ERROR) << "Detector creation fail!";
		std::exit(1);
	}

	if (source.find(".jpg") != std::string::npos || source.find(".png") != std::string::npos) {
		cv::Mat image = cv::imread(source);
		auto start = std::chrono::steady_clock::now();
		const auto input_blob = detector->preprocess_image(image);
		const auto [outputs, shapes] = engine->get_infer_results(input_blob);
		std::vector<Detection> detections = detector->postprocess(outputs, shapes, image.size());
		auto end = std::chrono::steady_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		//logger->info("Inference time: {} ms", duration);
		LOG(INFO) << "Inference time (ms): " << duration;
		for (const auto& d : detections) {
			cv::rectangle(image, d.bbox, cv::Scalar(255, 0, 0), 3);
			draw_label(image, classes[d.label], d.score, d.bbox.x, d.bbox.y);
		}
		cv::imwrite("data/processed.png", image);
		return 0;
	}

	std::unique_ptr<VideoCaptureInterface> videoInterface = createVideoInterface();

	if (!videoInterface->initialize(source)) {
		LOG(ERROR) << "Failed to initialize video capture for input: " << source;
		return 1;
	}

	cv::Mat frame;
	while (videoInterface->readFrame(frame)) {
		auto start = std::chrono::steady_clock::now();
		const auto input_blob = detector->preprocess_image(frame);
		const auto [outputs, shapes] = engine->get_infer_results(input_blob);
		std::vector<Detection> detections = detector->postprocess(outputs, shapes, frame.size());
		auto end = std::chrono::steady_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		double fps = 1000.0 / duration;
		std::string fpsText = "FPS: " + std::to_string(fps);
		cv::putText(frame, fpsText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
		for (const auto& d : detections) {
			cv::rectangle(frame, d.bbox, cv::Scalar(255, 0, 0), 3);
			draw_label(frame, classes[d.label], d.score, d.bbox.x, d.bbox.y);
		}

		cv::imshow("opencv feed", frame);
		char key = cv::waitKey(1);
		if (key == 27 || key == 'q') {
			LOG(INFO) << "Exit requested";
			break;
		}
	}

	videoInterface->release();
	return 0;
}
