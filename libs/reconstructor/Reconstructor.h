/*
 * Reconstructor.h
 *
 *  Created on: Nov 15, 2013
 *      Author: coert
 */

#ifndef RECONSTRUCTOR_H_
#define RECONSTRUCTOR_H_

#include <cstddef>
#include <vector>
#include <glm/vec4.hpp>

#include "Camera.h"
#include "Voxel.h"

namespace nl_uu_science_gmt
{

class Reconstructor
{
private:
	const std::vector<Camera> &m_cameras;  // vector of pointers to cameras
	const int m_height;                     // Cube half-space height from floor to ceiling
	const int m_step;                       // Step size (space between voxels)

	std::vector<cv::Point3f> m_corners ;    // Cube half-space corner locations

	cv::Vec3w m_voxels_dimension;           // Voxel count in each dimension
	size_t m_voxels_amount;                 // Voxel count
	cv::Size m_plane_size;                  // Camera FoV plane WxH

	std::vector<Voxel> m_voxels;           // Pointer vector to all voxels in the half-space
	std::vector<uint32_t> m_visible_voxels_indices;   // Pointer vector to all visible voxels
	std::vector<glm::vec4> m_scalar_field; // Values for each point in the half-space

	void initialize();

public:
	explicit Reconstructor(const std::vector<Camera>&);
	virtual ~Reconstructor();

	void update();
	void color(const std::vector<int>& labels, const std::vector<glm::vec4>& colors);

	cv::Vec3w getVoxelDimension() const
	{
		return m_voxels_dimension;
	}

	cv::Vec3i getOffset() const
	{
		return cv::Vec3i(-m_height, -m_height, 0);
	}

	uint32_t getVoxelCount() const
	{
		return m_voxels_amount;
	}

	uint32_t getVoxelSize() const
	{
		return m_step;
	}

	const std::vector<uint32_t>& getVisibleVoxelIndices() const
	{
		return m_visible_voxels_indices;
	}

	const std::vector<Voxel>& getVoxels() const
	{
		return m_voxels;
	}

	const std::vector<glm::vec4>& getScalarField() const
	{
		return m_scalar_field;
	}

	const std::vector<cv::Point3f>& getCorners() const
	{
		return m_corners;
	}

	int getSize() const
	{
		return m_height;
	}

	const cv::Size& getPlaneSize() const
	{
		return m_plane_size;
	}
};

} /* namespace nl_uu_science_gmt */

#endif /* RECONSTRUCTOR_H_ */
