#pragma once
#include <vector>
#include <opencv2/core/mat.hpp>

class ForegroundOptimizer
{
	//sorted from high to low
	std::vector<double> maxContourAreas; //save the amount of max contours you can have
	std::vector<int> maxContourIndices;
	std::vector<int> blackContours;
	int savedContoursCounter = 0;
	int nrContoursTracked; //the number of contours tracked

	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy; //not in use

	void saveMaxContour(double area, int contourIndex);

public:
	void FindContours(const cv::Mat& thresholdedImg);
	void SaveMaxContours(); //save the max nr contours (nrContoursTracked) which have the largest size.
	void DrawMaxContours(cv::Mat& image, bool removeBackground = true, cv::Scalar color = 255);
	void optimizeThresholds(int maxExtraContoursS, int maxExtraContoursV, const cv::Mat& h_image, const cv::Mat& s_image, const cv::Mat& v_image, std::vector<cv::Mat>& channels, uint8_t &h_threshold, uint8_t &s_threshold, uint8_t &v_threshold);
	cv::Mat runHSVThresholding(const cv::Mat & h_image, const cv::Mat & s_image, const cv::Mat & v_image, std::vector<cv::Mat>& channels, uint8_t h_threshold, uint8_t s_threshold, uint8_t v_threshold);

	explicit ForegroundOptimizer(int nrContoursTracked);

};
