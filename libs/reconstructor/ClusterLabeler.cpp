#include "ClusterLabeler.h"
#include "ForegroundOptimizer.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using nl_uu_science_gmt::Camera;
using nl_uu_science_gmt::ClusterLabeler;
using nl_uu_science_gmt::Voxel;



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

std::vector<std::vector<cv::Mat>> nl_uu_science_gmt::ClusterLabeler::ProjectTShirt(uint8_t num_clusters, const std::vector<Camera> &cameras, float voxel_step_size, const std::vector<Voxel> &voxels, const std::vector<uint32_t> &indices, const std::vector<int> &labels)
{
    constexpr float t_shirt_min_z = 800.0f;
    constexpr float t_shirt_max_z = 1400.0f;
	// Create black textures
	std::vector<std::vector<cv::Mat>> masks(cameras.size());
	for (auto& c : masks)
	{
		c.resize(num_clusters);
		for (auto& mask : c)
		{
			mask = cv::Mat::zeros(cameras.front().getSize(), CV_8U);
		}
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
		for (uint32_t j = 0; j < cameras.size(); ++j)
		{
			auto& camera = cameras[j];
			auto& mask = masks[j][person_index];
			mask.at<uint8_t>(camera.projectOnView(voxel.coordinate)) = 0xFF;
			mask.at<uint8_t>(camera.projectOnView(cv::Point3f(voxel.coordinate) + cv::Point3f(voxel_step_size, 0, 0))) = 0xFF;
			mask.at<uint8_t>(camera.projectOnView(cv::Point3f(voxel.coordinate) + cv::Point3f(voxel_step_size, voxel_step_size, 0))) = 0xFF;
			mask.at<uint8_t>(camera.projectOnView(cv::Point3f(voxel.coordinate) + cv::Point3f(0, voxel_step_size, 0))) = 0xFF;
			mask.at<uint8_t>(camera.projectOnView(cv::Point3f(voxel.coordinate) + cv::Point3f(0, voxel_step_size, voxel_step_size))) = 0xFF;
			mask.at<uint8_t>(camera.projectOnView(cv::Point3f(voxel.coordinate) + cv::Point3f(0, 0, voxel_step_size))) = 0xFF;
			mask.at<uint8_t>(camera.projectOnView(cv::Point3f(voxel.coordinate) + cv::Point3f(voxel_step_size, 0, voxel_step_size))) = 0xFF;
			mask.at<uint8_t>(camera.projectOnView(cv::Point3f(voxel.coordinate) + cv::Point3f(voxel_step_size, voxel_step_size, voxel_step_size))) = 0xFF;
		}
	}

	return masks;
}

void ClusterLabeler::CleanupMasks(std::vector<std::vector<cv::Mat>> masks)
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

