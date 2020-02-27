/*
 * General.h
 *
 *  Created on: Nov 13, 2013
 *      Author: coert
 */

#ifndef GENERAL_H_
#define GENERAL_H_

#include <opencv2/core/core.hpp>
#include <string_view>


namespace nl_uu_science_gmt
{

// Version and Main OpenCV window name
constexpr std::string_view VERSION = "2.5";
constexpr std::string_view VIDEO_WINDOW = "Video";
constexpr std::string_view SCENE_WINDOW = "OpenGL 3D scene";

// Some OpenCV colors
const static cv::Scalar Color_BLUE = cv::Scalar(255, 0, 0);
const static cv::Scalar Color_GREEN = cv::Scalar(0, 200, 0);
const static cv::Scalar Color_RED = cv::Scalar(0, 0, 255);
const static cv::Scalar Color_YELLOW = cv::Scalar(0, 255, 255);
const static cv::Scalar Color_MAGENTA = cv::Scalar(255, 0, 255);
const static cv::Scalar Color_CYAN = cv::Scalar(255, 255, 0);
const static cv::Scalar Color_WHITE = cv::Scalar(255, 255, 255);
const static cv::Scalar Color_BLACK = cv::Scalar(0, 0, 0);

namespace General {
constexpr std::string_view CBConfigFile = "checkerboard.xml";
constexpr std::string_view CalibrationVideo = "calibration.avi";
constexpr std::string_view CheckerboadVideo = "checkerboard.avi";
constexpr std::string_view BackgroundImageFile = "background.png";
constexpr std::string_view VideoFile = "video.avi";
constexpr std::string_view IntrinsicsFile = "intrinsics.xml";
constexpr std::string_view CheckerboadCorners = "boardcorners.xml";
constexpr std::string_view ConfigFile = "config.xml";
}

} /* namespace nl_uu_science_gmt */

#endif /* GENERAL_H_ */
