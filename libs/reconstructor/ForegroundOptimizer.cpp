#include "ForegroundOptimizer.h"
#include <opencv2/imgproc/imgproc.hpp>

ForegroundOptimizer::ForegroundOptimizer(int nrContoursTracked)
	: nrContoursTracked(nrContoursTracked)
{
}

void ForegroundOptimizer::optimizeThresholds(int maxExtraContoursS, int maxExtraContoursV, const cv::Mat& h_image, const cv::Mat& s_image, const cv::Mat& v_image, std::vector<cv::Mat>& channels, uint8_t &h_threshold, uint8_t &s_threshold, uint8_t &v_threshold)
{
	contours.clear();
	int lastMergedContours_i = 255;
	int lastNrContours = std::numeric_limits<int>::max() - 1000;
	//optimize V
	for (int i = 255; i > 5; i -= 5)
	{
		cv::Mat foreground = runHSVThresholding(
			h_image, s_image, v_image,
			channels,
			h_threshold, s_threshold, i
		);
		FindContours(foreground);
		int currNrContours = contours.size();
		if (currNrContours > lastNrContours + maxExtraContoursV)
		{
			v_threshold = lastMergedContours_i;
			break;
		}
		else if (currNrContours < lastNrContours)
		{
			lastMergedContours_i = i;
		}

		lastNrContours = currNrContours;
	}

	contours.clear();
	lastMergedContours_i = 255;
	lastNrContours = std::numeric_limits<int>::max() - 1000;
	//optimize S
	for (int i = 255; i > 5; i -= 5)
	{
		cv::Mat foreground = runHSVThresholding(
			h_image, s_image, v_image,
			channels,
			h_threshold, i,	v_threshold
		);
		FindContours(foreground);
		int currNrContours = contours.size();
		if (currNrContours > lastNrContours + maxExtraContoursS)
		{
			s_threshold = lastMergedContours_i;
			break;
		}
		
		else if (currNrContours <= lastNrContours)
		{
			lastMergedContours_i = i;
		}
	
		lastNrContours = currNrContours;
	}
	
}

cv::Mat ForegroundOptimizer::runHSVThresholding(const cv::Mat& h_image, const cv::Mat& s_image, const cv::Mat& v_image, std::vector<cv::Mat>& channels, uint8_t h_threshold, uint8_t s_threshold, uint8_t v_threshold)
{

	// Background subtraction H
	cv::Mat tmp, foreground, background;
	cv::absdiff(channels[0], h_image, tmp);
	cv::threshold(tmp, foreground, h_threshold, 255, cv::THRESH_BINARY);

	// Background subtraction S
	cv::absdiff(channels[1], s_image, tmp);
	cv::threshold(tmp, background, s_threshold, 255, cv::THRESH_BINARY);
	cv::bitwise_and(foreground, background, foreground);

	// Background subtraction V
	cv::absdiff(channels[2], v_image, tmp);
	cv::threshold(tmp, background, v_threshold, 255, cv::THRESH_BINARY);
	cv::bitwise_or(foreground, background, foreground);

	return foreground;
}

void ForegroundOptimizer::saveMaxContour(double area, int contourIndex)
{
	for (int i = 0; i != nrContoursTracked; i++)
	{
		if (maxContourAreas[i] < area) //maxContours are sorted from high to low.
		{
			maxContourAreas.insert(maxContourAreas.begin() + i, area);
			maxContourIndices.insert(maxContourIndices.begin() + i, contourIndex);
			maxContourAreas.pop_back();
			maxContourIndices.pop_back();
			return;
		}
	}
}

void ForegroundOptimizer::FindContours(const cv::Mat& thresholdedImg)
{
	cv::findContours(thresholdedImg, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
}

void ForegroundOptimizer::SaveMaxContours()
{
	maxContourAreas.clear();
	maxContourAreas.resize(nrContoursTracked);
	std::fill(maxContourAreas.begin(), maxContourAreas.end(), 0);
	maxContourIndices.clear();
	maxContourIndices.resize(nrContoursTracked);
	blackContours.clear();
	
	for (int i = 0; i < contours.size(); i++)
	{
		double areaBlackPositive = contourArea(contours[i], true);
		if (areaBlackPositive > 20)
		{
			blackContours.push_back(i);
		}
		double area = -areaBlackPositive; // negative means white apparently, and you want white
		if (area > maxContourAreas[nrContoursTracked - 1]) 
		{
			saveMaxContour(area, i);
		}

	}
}

void ForegroundOptimizer::DrawMaxContours(cv::Mat& image, bool removeBackground, cv::Scalar color)
{
	if (removeBackground)
	{
		image = cv::Mat::zeros(image.size(), image.type());
		for (int i = 0; i < nrContoursTracked; i++)
		{
			cv::drawContours(image, contours, maxContourIndices[i], color, cv::FILLED);
		}
		for (int i = 0; i < blackContours.size(); i++)
		{
			cv::drawContours(image, contours, blackContours[i], cv::Scalar(0), cv::FILLED); //draw black contours over the white ones (holes between the legs, etc)
		}
	}
	else
	{
		for (int i = 0; i < nrContoursTracked; i++)
		{
			cv::drawContours(image, contours, maxContourIndices[i], color, cv::FILLED);
		}
	}

	
}
