#pragma once
#include "common.hpp"
// Define a global logger variable
//std::shared_ptr<spdlog::logger> logger;
//
//void initializeLogger() {
//
//    std::vector<spdlog::sink_ptr> sinks;
//    sinks.push_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
//    sinks.push_back( std::make_shared<spdlog::sinks::rotating_file_sink_mt>("logs/output.log", 1024*1024*10, 3, true));
//    logger = std::make_shared<spdlog::logger>("logger", begin(sinks), end(sinks));
//
//    spdlog::register_logger(logger);
//    logger->flush_on(spdlog::level::info);
//}


void initializeLogger() {
	google::InitGoogleLogging("ODInfer", [](std::ostream& s, const google::LogMessageInfo& l, void*) { s << l.severity[0]
		<< std::setw(4) << 1900 + l.time.year()
		<< std::setw(2) << 1 + l.time.month()
		<< std::setw(2) << l.time.day()
		<< ' '
		<< std::setw(2) << l.time.hour() << ':'
		<< std::setw(2) << l.time.min() << ':'
		<< std::setw(2) << l.time.sec()
		<< ' '
		<< std::setfill(' ') << std::setw(5)
		<< l.thread_id << std::setfill('0')
		<< "]"; }); //
	FLAGS_alsologtostderr = true;
	FLAGS_colorlogtostderr = true;
	//FLAGS_log_prefix = false;            
	//FLAGS_log_year_in_prefix = true;
	//FLAGS_logbufsecs = 10;                    
	FLAGS_max_log_size = 256;
	FLAGS_stop_logging_if_full_disk = true;
	google::SetLogFilenameExtension("txt");
	FLAGS_timestamp_in_logfile_name = false;
	FLAGS_log_file_header = false;
	FLAGS_minloglevel = google::GLOG_INFO;
	google::EnableLogCleaner(180);             // keep your logs for 180 days
}