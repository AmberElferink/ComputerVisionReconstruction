#pragma once

#include <cstdint>
#include <filesystem>
#include <tuple>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/ml/ml.hpp> //EM include, use with cv::ml::EM

#include "Voxel.h"
constexpr uint32_t NUM_CONTOURS = 4;
constexpr uint32_t NUM_VIEWS = 4;

namespace nl_uu_science_gmt
{

class Camera;

class ClusterLabeler
{
private:
	std::vector<std::vector<cv::Ptr<cv::ml::EM>>> ems;
	int m_numClusters;
	int m_numCameras;
public:
	std::pair<cv::Mat, std::vector<int>> FindClusters(uint8_t num_clusters, uint8_t num_retries, const std::vector<Voxel> &voxels, const std::vector<uint32_t> &indices);
	std::vector<cv::Mat> ProjectTShirt(uint8_t num_clusters, const Camera& cameras, float voxel_step_size, const std::vector<Voxel> &voxels, const std::vector<uint32_t> &indices, const std::vector<int> &labels);
	void CleanupMasks(std::vector<std::vector<cv::Mat>>& masks);
	void ShowMaskCutouts(std::vector<std::vector<cv::Mat>>& masks, std::vector<cv::Mat>& hsvImages, std::vector<std::vector<cv::Mat>>& cutouts);
	void TrainEMS(std::vector<std::vector<cv::Mat>>& masks, std::vector<cv::Mat>& hsvImages, std::vector<std::vector<cv::Mat>>& reshaped_cutouts);
	void PredictEMS(const std::vector<Camera>& cameras, const std::vector<std::vector<cv::Mat>>& masks);
	void InitializeEMS();
	void CheckEMS(std::vector<std::vector<cv::Mat>>& reshaped_cutouts);
	void SaveEMS(const std::filesystem::path& dataPath);
	void LoadEMS(const std::filesystem::path& dataPath);

	cv::Mat GetCutout(const cv::Mat& mask, const cv::Mat& hsv_image);

	int getNumClusters() { return m_numClusters; }
	int getNumCameras() { return m_numCameras; }

	//clusterLabeler will have the defined number of clusters and camera's in ClusterLabeler.h by default.
	ClusterLabeler(int numClusters, int numCameras) :
		m_numClusters(numClusters),
		m_numCameras(numCameras)
	{};

	ClusterLabeler() :
		m_numClusters(NUM_CONTOURS),
		m_numCameras(NUM_VIEWS)
	{};
};

}
