/*
 * Reconstructor.cpp
 *
 *  Created on: Nov 15, 2013
 *      Author: coert
 */

#include "Reconstructor.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/core/operations.hpp>
#include <cassert>
#include <iostream>

using namespace cv;

namespace nl_uu_science_gmt
{

/**
 * Constructor
 * Voxel reconstruction class
 */
Reconstructor::Reconstructor(
		const std::vector<Camera> &cs) :
				m_cameras(cs),
				m_height(2048),
				m_step(32)
{
	for (auto c : m_cameras)
	{
		if (m_plane_size.area() > 0)
			assert(m_plane_size.width == c.getSize().width && m_plane_size.height == c.getSize().height);
		else
			m_plane_size = c.getSize();
	}

	const size_t edge = 2 * m_height;
	m_voxels_dimension = Vec3w(edge / m_step, edge / m_step, m_height / m_step);
	m_voxels_amount = (edge / m_step) * (edge / m_step) * (m_height / m_step);
	m_scalar_field.resize(m_voxels_amount, 0.0f);

	initialize();
}

Reconstructor::~Reconstructor() = default;

/**
 * Create some Look Up Tables
 * 	- LUT for the scene's box corners
 * 	- LUT with a map of the entire voxelspace: point-on-cam to voxels
 * 	- LUT with a map of the entire voxelspace: voxel to cam points-on-cam
 */
void Reconstructor::initialize()
{
	// Cube dimensions from [(-m_height, m_height), (-m_height, m_height), (0, m_height)]
	const int xL = -m_height;
	const int xR = m_height;
	const int yL = -m_height;
	const int yR = m_height;
	const int zL = 0;
	const int zR = m_height;
	const int plane_y = (yR - yL) / m_step;
	const int plane_x = (xR - xL) / m_step;
	const int plane = plane_y * plane_x;

	// Save the 8 volume corners
	// bottom
	m_corners.push_back(new Point3f((float) xL, (float) yL, (float) zL));
	m_corners.push_back(new Point3f((float) xL, (float) yR, (float) zL));
	m_corners.push_back(new Point3f((float) xR, (float) yR, (float) zL));
	m_corners.push_back(new Point3f((float) xR, (float) yL, (float) zL));

	// top
	m_corners.push_back(new Point3f((float) xL, (float) yL, (float) zR));
	m_corners.push_back(new Point3f((float) xL, (float) yR, (float) zR));
	m_corners.push_back(new Point3f((float) xR, (float) yR, (float) zR));
	m_corners.push_back(new Point3f((float) xR, (float) yL, (float) zR));

	// Acquire some memory for efficiency
	std::cout << "Initializing " << m_voxels_amount << " voxels..." << std::endl;
	m_voxels.resize(m_voxels_amount);

	int z;
	int pdone = 0;
#pragma omp parallel for schedule(runtime) private(z) shared(pdone)
	for (z = zL; z < zR; z += m_step)
	{
		const int zp = (z - zL) / m_step;
		int done = cvRound((zp * plane / (double) m_voxels_amount) * 100.0);

#pragma omp critical
		if (done > pdone)
		{
			pdone = done;
			std::cout << done << "%\r" << std::flush;
		}

		int y, x;
		for (y = yL; y < yR; y += m_step)
		{
			const int yp = (y - yL) / m_step;

			for (x = xL; x < xR; x += m_step)
			{
				const int xp = (x - xL) / m_step;

				const int p = zp * plane + yp * plane_x + xp;  // The voxel's index

				// Create all voxels
				Voxel* voxel = &m_voxels[p];
				voxel->x = x;
				voxel->y = y;
				voxel->z = z;
				voxel->camera_projection = std::vector<Point>(m_cameras.size());
				voxel->valid_camera_projection = std::vector<int>(m_cameras.size(), 0);

				for (size_t c = 0; c < m_cameras.size(); ++c)
				{
					Point point = m_cameras[c].projectOnView(Point3f((float) x, (float) y, (float) z));

					// Save the pixel coordinates 'point' of the voxel projection on camera 'c'
					voxel->camera_projection[(int) c] = point;

					// If it's within the camera's FoV, flag the projection
					if (point.x >= 0 && point.x < m_plane_size.width && point.y >= 0 && point.y < m_plane_size.height)
						voxel->valid_camera_projection[(int) c] = 1;
				}
			}
		}
	}

	std::cout << "done!" << std::endl;
}

/**
 * Count the amount of camera's each voxel in the space appears on,
 * if that amount equals the amount of cameras, add that voxel to the
 * visible_voxels vector
 */
void Reconstructor::update()
{
	m_visible_voxels_indices.clear();
	std::vector<uint32_t> visible_voxels;

	uint32_t v;
#pragma omp parallel for schedule(runtime) private(v) shared(visible_voxels)
	for (v = 0; v < (uint32_t) m_voxels_amount; ++v)
	{
		int camera_counter = 0;
		const Voxel* voxel = &m_voxels[v];
		m_scalar_field[v] = 0.0f;
		for (size_t c = 0; c < m_cameras.size(); ++c)
		{
			if (voxel->valid_camera_projection[c])
			{
				const Point point = voxel->camera_projection[c];

				//If there's a white pixel on the foreground image at the projection point, add the camera
				if (m_cameras[c].getForegroundImage().at<uchar>(point) == 255)
				{
					++camera_counter;
				}
			}
		}

		// If the voxel is present on all cameras
		if (camera_counter == m_cameras.size())
		{
			m_scalar_field[v] = 1.0f;
#pragma omp critical //push_back is critical
			visible_voxels.push_back(v);
		}
	}

	m_visible_voxels_indices.insert(m_visible_voxels_indices.end(), visible_voxels.begin(), visible_voxels.end());
}

} /* namespace nl_uu_science_gmt */
