/*
 * VoxelReconstruction.h
 *
 *  Created on: Nov 13, 2013
 *      Author: coert
 */

#ifndef VOXELRECONSTRUCTION_H_
#define VOXELRECONSTRUCTION_H_

#include <filesystem>
#include <vector>

#include <Camera.h>

namespace nl_uu_science_gmt
{

class VoxelReconstruction
{
	const std::filesystem::path m_data_path;
	std::vector<Camera> m_cam_views;

public:
	VoxelReconstruction(std::filesystem::path , int);
	virtual ~VoxelReconstruction();

	static void showKeys();

	void run(int, char**);
};

} /* namespace nl_uu_science_gmt */

#endif /* VOXELRECONSTRUCTION_H_ */
