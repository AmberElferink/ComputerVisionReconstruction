#pragma once

#include <cstdint>
#include <filesystem>
#include <tuple>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/ml/ml.hpp> //EM include, use with cv::ml::EM

#include "Voxel.h"

namespace nl_uu_science_gmt
{

class Camera;

class ClusterLabeler
{
private:
	std::vector<std::vector<cv::Ptr<cv::ml::EM>>> ems;
public:
	std::pair<cv::Mat, std::vector<int>> FindClusters(uint8_t num_clusters, uint8_t num_retries, const std::vector<Voxel> &voxels, const std::vector<uint32_t> &indices);
	std::vector<cv::Mat> ProjectTShirt(uint8_t num_clusters, const Camera& cameras, float voxel_step_size, const std::vector<Voxel> &voxels, const std::vector<uint32_t> &indices, const std::vector<int> &labels);
	void CleanupMasks(std::vector<std::vector<cv::Mat>>& masks);
	void ShowMaskCutouts(std::vector<std::vector<cv::Mat>>& masks, std::vector<cv::Mat>& hsvImages, std::vector<std::vector<cv::Mat>>& cutouts);
	void CreateColorScheme(std::vector<std::vector<cv::Mat>>& masks, std::vector<cv::Mat>& hsvImages, std::vector<std::vector<cv::Mat>>& reshaped_cutouts);
	void CheckEMS(std::vector<std::vector<cv::Mat>>& reshaped_cutouts);
	void SaveEMS(std::filesystem::path dataPath);
};

}
