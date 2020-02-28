#pragma once

#include <vector>
#include <opencv2/core/types.hpp>

namespace nl_uu_science_gmt
{
/*
 * Voxel structure
 * Represents a 3D pixel in the half space
 */
struct Voxel
{
  cv::Point3i coordinate;                     // Coordinates
  cv::Scalar color;                           // Color
  std::vector <cv::Point> camera_projection;  // Projection location for camera[c]'s FoV (2D)
  std::vector<int> valid_camera_projection;   // Flag if camera projection is in camera[c]'s FoV
};
} /* namespace nl_uu_science_gmt */
