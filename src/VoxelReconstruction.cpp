/*
 * VoxelReconstruction.cpp
 *
 *  Created on: Nov 13, 2013
 *      Author: coert
 */

#include "VoxelReconstruction.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <cassert>
#include <iostream>
#include <utility>

#include "controllers/Renderer.h"
#include <Reconstructor.h>
#include "controllers/Scene3DRenderer.h"
#include "utilities/General.h"

using namespace nl_uu_science_gmt;
using namespace cv;

constexpr std::string_view VERSION = "2.5";

namespace nl_uu_science_gmt
{

/**
 * Main constructor, initialized all cameras
 */
VoxelReconstruction::VoxelReconstruction(std::filesystem::path dp, int cam_views_amount) :
		m_data_path(std::move(dp))
{
	for (int v = 0; v < cam_views_amount; ++v)
	{
		auto full_path = m_data_path / ("cam" + std::to_string(v + 1));

		/*
		 * Assert that there's a background image or video file and \
		 * that there's a video file
		 */
		std::cout << full_path / General::BackgroundImageFile << std::endl;
		std::cout << full_path / General::VideoFile << std::endl;
		assert(
			std::filesystem::exists(full_path / General::BackgroundImageFile)
			&&
			std::filesystem::exists(full_path /General::VideoFile)
		);

		/*
		 * Assert that if there's no config.xml file, there's an intrinsics file and
		 * a checkerboard video to create the extrinsics from
		 */
		assert(std::filesystem::exists(full_path / General::ConfigFile));

		m_cam_views.emplace_back(full_path, General::ConfigFile.data(), v);
	}
}

VoxelReconstruction::~VoxelReconstruction() = default;

/**
 * What you can hit
 */
void VoxelReconstruction::showKeys()
{
	std::cout << "VoxelReconstruction v" << VERSION << std::endl << std::endl;
	std::cout << "Use these keys:" << std::endl;
	std::cout << "q       : Quit" << std::endl;
	std::cout << "p       : Pause" << std::endl;
	std::cout << "b       : Frame back" << std::endl;
	std::cout << "n       : Next frame" << std::endl;
	std::cout << "r       : Rotate voxel space" << std::endl;
	std::cout << "s       : Show/hide arcball wire sphere (Linux only)" << std::endl;
	std::cout << "v       : Show/hide voxel space box" << std::endl;
	std::cout << "g       : Show/hide ground plane" << std::endl;
	std::cout << "c       : Show/hide cameras" << std::endl;
	std::cout << "i       : Show/hide camera numbers (Linux only)" << std::endl;
	std::cout << "o       : Show/hide origin" << std::endl;
	std::cout << "t       : Top view" << std::endl;
	std::cout << "1,2,3,4 : Switch camera #" << std::endl << std::endl;
	std::cout << "Zoom with the scrollwheel while on the 3D scene" << std::endl;
	std::cout << "Rotate the 3D scene with left click+drag" << std::endl << std::endl;
}

/**
 * - If the xml-file with camera intrinsics, extrinsics and distortion is missing,
 *   create it from the checkerboard video and the measured camera intrinsics
 * - After that initialize the scene rendering classes
 * - Run it!
 */
void VoxelReconstruction::run(int argc, char** argv)
{
	for (auto& v : m_cam_views)
	{
		auto ok = v.initialize(General::BackgroundImageFile, General::VideoFile);
		assert(ok);
	}

	destroyAllWindows();
	namedWindow(VIDEO_WINDOW.data(), CV_WINDOW_KEEPRATIO);

	Reconstructor reconstructor(m_cam_views);
	Scene3DRenderer scene3d(reconstructor, m_cam_views);
	Renderer glut(scene3d);

	glut.initialize(SCENE_WINDOW.data(), argc, argv);
}

} /* namespace nl_uu_science_gmt */
