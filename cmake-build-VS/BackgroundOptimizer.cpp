#include "BackgroundOptimizer.h"
#include <opencv2/imgproc/imgproc.hpp>


void BackgroundOptimizer::saveMaxContour(double area, int contourIndex)
{
	for (int i = 0; i != nrContoursTracked; i++)
	{
		if (maxContourAreas[i] < area) //maxContours are sorted from high to low.
		{
			maxContourAreas.insert(maxContourAreas.begin() + i, area);
			maxContourIndices.insert(maxContourIndices.begin() + i, contourIndex);
			maxContourAreas.pop_back(); //remove the last element
			maxContourIndices.pop_back();
		}
	}
}

void BackgroundOptimizer::FindContours(cv::Mat& thresholdedImg)
{
	cv::findContours(thresholdedImg, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
}

void BackgroundOptimizer::SaveMaxContours()
{
	maxContourAreas.clear();
	maxContourAreas.resize(nrContoursTracked);
	std::fill(maxContourAreas.begin(), maxContourAreas.end(), 0);
	maxContourIndices.clear();
	maxContourIndices.resize(nrContoursTracked);
	
	for (int i = 0; i < contours.size(); i++)
	{
		double area = contourArea(contours[i], true);
		if (area > maxContourAreas[nrContoursTracked - 1]) //check if its higher than the lowest value in the array
		{
			saveMaxContour(area, i);
		}
	}
}

void BackgroundOptimizer::DrawMaxContours(cv::Mat& image)
{
	std::vector<std::vector<cv::Point>> maxContours;
	for (int i = 0; i < nrContoursTracked; i++)
	{
		cv::drawContours(image, maxContours, maxContourIndices[i], (0, 255, 0), 3);
		//maxContours.push_back(contours[maxContourIndices[i]]);
	}
	
}
