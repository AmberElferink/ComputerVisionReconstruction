#pragma once

#include <cstdint>
#include <tuple>
#include <vector>

#include "Voxel.h"
#include "Camera.h"

namespace nl_uu_science_gmt
{

class ClusterLabeler
{
public:
	std::pair<cv::Mat, std::vector<int>> FindClusters(uint8_t num_clusters, uint8_t num_retries, const std::vector<Voxel> &voxels, const std::vector<uint32_t> &indices);
	std::vector<std::vector<cv::Mat>> ProjectTShirt(uint8_t num_clusters, const std::vector<Camera> &cameras, float voxel_step_size, const std::vector<Voxel> &voxels, const std::vector<uint32_t> &indices, const std::vector<int> &labels);
};

}
