#include "ClusterLabeler.h"
#include "ForegroundOptimizer.h"

#include <opencv2/highgui.hpp> //imshow debug
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp> //EM include, use with cv::ml::EM

#include "Camera.h"

using nl_uu_science_gmt::Camera;
using nl_uu_science_gmt::ClusterLabeler;
using nl_uu_science_gmt::Voxel;






std::vector<std::vector<cv::Ptr<cv::ml::EM>>> ems;

std::pair<cv::Mat, std::vector<int>> ClusterLabeler::FindClusters(uint8_t num_clusters, uint8_t num_retries, const std::vector<Voxel> &voxels, const std::vector<uint32_t> &indices)
{
	// Project the voxels into 2d, ignoring up vector
	std::vector<cv::Point2f> voxels_2d;
	voxels_2d.reserve(indices.size());
	for (auto i : indices) {
		voxels_2d.emplace_back(voxels[i].coordinate.x, voxels[i].coordinate.y);
	}

	// Run k-means for labels and centers
	std::vector<int> labels;
	labels.reserve(voxels_2d.size());
	cv::Mat centers;
	cv::kmeans(voxels_2d, num_clusters, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 1000, 0.001), num_retries, cv::KMEANS_PP_CENTERS, centers);

	return std::make_pair(centers, labels);
}

std::vector<cv::Mat> nl_uu_science_gmt::ClusterLabeler::ProjectTShirt(uint8_t num_clusters, const Camera& camera, float voxel_step_size, const std::vector<Voxel> &voxels, const std::vector<uint32_t> &indices, const std::vector<int> &labels)
{
	constexpr float t_shirt_min_z = 800.0f;
	constexpr float t_shirt_max_z = 1400.0f;
	// Create black textures
	std::vector<cv::Mat> masks(num_clusters);
	for (auto& mask : masks)
	{
		mask = cv::Mat::zeros(camera.getSize(), CV_8U);
	}

	for (uint32_t i = 0; i < labels.size(); ++i)
	{
		auto person_index = labels[i];
		auto voxel_index = indices[i];
		auto& voxel = voxels[voxel_index];
		// Cull voxels which are too low or too high to be part of the shirt
		if (voxel.coordinate.z < t_shirt_min_z || voxel.coordinate.z > t_shirt_max_z)
		{
			continue;
		}
			auto& mask = masks[person_index];
			mask.at<uint8_t>(camera.projectOnView(voxel.coordinate)) = 0xFF;
//			mask.at<uint8_t>(camera.projectOnView(cv::Point3f(voxel.coordinate) + cv::Point3f(voxel_step_size, 0, 0))) = 0xFF;
//			mask.at<uint8_t>(camera.projectOnView(cv::Point3f(voxel.coordinate) + cv::Point3f(voxel_step_size, voxel_step_size, 0))) = 0xFF;
//			mask.at<uint8_t>(camera.projectOnView(cv::Point3f(voxel.coordinate) + cv::Point3f(0, voxel_step_size, 0))) = 0xFF;
//			mask.at<uint8_t>(camera.projectOnView(cv::Point3f(voxel.coordinate) + cv::Point3f(0, voxel_step_size, voxel_step_size))) = 0xFF;
//			mask.at<uint8_t>(camera.projectOnView(cv::Point3f(voxel.coordinate) + cv::Point3f(0, 0, voxel_step_size))) = 0xFF;
//			mask.at<uint8_t>(camera.projectOnView(cv::Point3f(voxel.coordinate) + cv::Point3f(voxel_step_size, 0, voxel_step_size))) = 0xFF;
//			mask.at<uint8_t>(camera.projectOnView(cv::Point3f(voxel.coordinate) + cv::Point3f(voxel_step_size, voxel_step_size, voxel_step_size))) = 0xFF;
	}

	return masks;
}

void ClusterLabeler::CleanupMasks(std::vector<std::vector<cv::Mat>> &masks)
{
	ForegroundOptimizer optimizer(1);
	for (auto camera : masks)
	{
		for (auto mask : camera)
		{
			optimizer.FindContours(mask);
			optimizer.SaveMaxContours(1000, 50);
			optimizer.DrawMaxContours(mask);
		}
	}
}
#pragma optimize("", off)
//most useful opencv example: https://github.com/opencv/opencv/blob/master/samples/cpp/em.cpp
//example EM in use: http://seiya-kumada.blogspot.com/2013/03/em-algorithm-practice-by-opencv.html
//documentation EM: https://docs.opencv.org/3.4/d1/dfb/classcv_1_1ml_1_1EM.html#ae3f12147ba846a53601b60c784ee263d
//opencv samle EM: https://github.com/opencv/opencv/blob/master/samples/cpp/em.cpp
void ClusterLabeler::CreateColorScheme(std::vector<std::vector<cv::Mat>>& masks, std::vector<cv::Mat>& hsvImages, std::vector<std::vector<cv::Mat>>& cutouts)
{

	using namespace cv;
	using namespace cv::ml;
	using namespace std;

	Ptr<EM> em = EM::create();

	//cv::Mat cutout;
	//cv::bitwise_and(hsvImages[0], hsvImages[0], cutout, masks[0][0]);
	//cv::Mat reshaped_cutout = cutout.reshape(1, cutout.rows * cutout.cols);
	//cv::Mat samples;
	//reshaped_cutout.convertTo(samples, CV_64FC1, 1.0 / 255.0);

	em->setClustersNumber(4);
	em->setCovarianceMatrixType(EM::COV_MAT_SPHERICAL);
	em->setTermCriteria(TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 300, 0.1));
	
	cv::Mat& mask = masks[0][0];
	int whitePixels = sum(mask)[0] / 255;
	Mat cutout( Size(whitePixels, 1), CV_64FC3);
	int counter = 0;
	for (int i = 0; i < mask.rows; i++)
	{
		for (int j = 0; j < mask.cols; j++)
		{
			if (mask.at<uchar>(i, j))
			{
				cutout.at<Vec3d>(0, counter) = ((Vec3d) hsvImages[0].at<Vec3b>(i, j))/ 255.0;
				//cutout.at<double>(counter, 1, 1) = ((double) hsvImages[0].at<uchar>(i, j, 1)) / 255.0;
				//cutout.at<double>(counter, 1, 2) = ((double) hsvImages[0].at<uchar>(i, j, 2)) / 255.0;
				counter++;
			}
		}
	}

	Mat reshaped_cutout = cutout.reshape(1, 3).t(); //put 1 pixel value per row (each sample is 1 pixel)
	cv::Mat likelyhoods; //do log(likelyhood[nr] to find the likelyhood it belongs to the cluster (log(1.7) = 0.25 for example)
	cv::Mat labels; //
	em->trainEM(reshaped_cutout, likelyhoods, labels, cv::noArray());


	//el 0 is likelyhood, el 1 is Index
	cv::Mat prob;
	float result = em->predict(reshaped_cutout, prob);
	int w = 0;

	//cutouts.reserve(masks.size());
	//ems.resize(masks.size());
	//for (int i = 0; i < masks.size(); i++) //loop over cameras
	//{
	//	std::vector<cv::Mat> camMasks;
	//	for (int j = 0; j < masks[i].size();j++) //loop over masks
	//	{
	//		cv::Mat cutout;
	//		cv::bitwise_and(hsvImages[i], hsvImages[i], cutout, masks[i][j]);
	//		camMasks.push_back(cutout.clone());

	//		cv::Mat reshaped_cutout = cutout.reshape(1, cutout.rows * cutout.cols);
	//		cv::Mat samples;
	//		reshaped_cutout.convertTo(samples, CV_64FC1, 1.0 / 255.0);


	//		auto& em = ems[i].emplace_back(EM::create()); //transfer ownership of the Ptr to ems
	//		em->setClustersNumber(4);
	//		em->setCovarianceMatrixType(EM::COV_MAT_SPHERICAL);
	//		em->setTermCriteria(TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 300, 0.1));
	//		
	//	
	//		

	//		cv::Mat likelyhoods;
	//		cv::Mat labels;
	//		ems[i][j]->trainEM(samples, likelyhoods, labels, cv::noArray());
	//		

	//		//el 0 is likelyhood, el 1 is Index
	//		Vec2d likelyhoodIndex = ems[i][j]->predict2(samples, noArray());

	//		int w = 0;
	//		
	//	}
	//	cutouts.push_back(camMasks);
	//}
}

#pragma optimize("", on)