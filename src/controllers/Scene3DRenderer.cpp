/*
 * Scene3DRenderer.cpp
 *
 *  Created on: Nov 15, 2013
 *      Author: coert
 */

#include "Scene3DRenderer.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/plot.hpp>
#include <ClusterLabeler.h>
#include <ForegroundOptimizer.h>
#include <opencv2/ml/ml.hpp>
#include "../utilities/General.h"


using namespace std;
using namespace cv;



namespace nl_uu_science_gmt
{
	std::vector<std::vector<cv::Ptr<cv::ml::EM>>> ems;

/**
 * Constructor
 * Scene properties class (mostly called by Glut)
 */
Scene3DRenderer::Scene3DRenderer(Reconstructor &r, vector<Camera> &cs)
	:
	  m_clusterLabeler(std::make_unique<ClusterLabeler>())
	, m_foregroundOptimizer(std::make_unique<ForegroundOptimizer>(m_clusterLabeler->getNumClusters()))
	, m_reconstructor(r)
	, m_cameras(cs)
	, m_num(4)
	, m_square_side_len()
	, m_sphere_radius(1850)
	, m_width(640)
	, m_height(480)
	, m_aspect_ratio(static_cast<float>(m_width) / static_cast<float>(m_height))
	, m_arcball_eye()
	, m_arcball_centre()
	, m_arcball_up()
	, m_camera_view(true)
	, m_show_volume(true)
	, m_show_grd_flr(true)
	, m_show_cam(true)
	, m_show_org(true)
	, m_show_arcball(false)
	, m_show_info(true)
	, m_fullscreen(false)
	, m_quit(false)
	, m_paused(false)
	, m_rotate(false)
	, m_number_of_frames(m_cameras.front().getFramesAmount())
	, m_current_frame(0)
	, m_previous_frame(-1)
	, m_current_camera(0)
	, m_previous_camera(0)
	, m_h_threshold(0)
	, m_ph_threshold(m_h_threshold)
	, m_s_threshold(19)
	, m_ps_threshold(m_s_threshold)
	, m_v_threshold(48)
	, m_pv_threshold(m_v_threshold)
	, m_thresholdMaxNoise(15)
	, m_cluster_traces{
		std::vector<cv::Point2f>(m_number_of_frames),
		std::vector<cv::Point2f>(m_number_of_frames),
		std::vector<cv::Point2f>(m_number_of_frames),
		std::vector<cv::Point2f>(m_number_of_frames),
	}
{
	m_clusterLabeler->LoadEMS(m_cameras.front().getDataPath() / "..");

	// Read the checkerboard properties (XML)
	FileStorage fs;
	fs.open((m_cameras.front().getDataPath() / ".." / General::CBConfigFile).u8string(), FileStorage::READ);
	if (fs.isOpened())
	{
		fs["CheckerBoardWidth"] >> m_board_size.width;
		fs["CheckerBoardHeight"] >> m_board_size.height;
		fs["CheckerBoardSquareSize"] >> m_square_side_len;
	}
	fs.release();

	//calibThresholds();

	updateTrackbars();

	createFloorGrid();
	setTopView();
}

void Scene3DRenderer::updateTrackbars()
{
	createTrackbar("max noise", VIDEO_WINDOW.data(), &m_thresholdMaxNoise, 255);
	createTrackbar("Frame", VIDEO_WINDOW.data(), &m_current_frame, m_number_of_frames - 2);

	int h_threshold = m_h_threshold;
	int s_threshold = m_s_threshold;
	int v_threshold = m_v_threshold;

	createTrackbar("H", VIDEO_WINDOW.data(), &h_threshold, 255);
	createTrackbar("S", VIDEO_WINDOW.data(), &s_threshold, 255);
	createTrackbar("V", VIDEO_WINDOW.data(), &v_threshold, 255);

	h_threshold = static_cast<uint8_t>(m_h_threshold);
	s_threshold = static_cast<uint8_t>(m_s_threshold);
	v_threshold = static_cast<uint8_t>(m_v_threshold);
}

void Scene3DRenderer::calibThresholds()
{
	m_h_threshold = 0;
	m_s_threshold = 255;
	m_v_threshold = 255;

	m_cameras[3].advanceVideoFrame();
	assert(!m_cameras[3].getVideoFrame(0).empty());
	Mat hsv_image;
	cvtColor(m_cameras[3].getVideoFrame(0), hsv_image, CV_BGR2HSV);  // from BGR to HSV color space

	std::vector<cv::Mat> channels;
	cv::split(hsv_image, channels);  // Split the HSV-channels for further analysis

	m_foregroundOptimizer->optimizeThresholds(
		m_thresholdMaxNoise,
		m_thresholdMaxNoise,
		m_cameras[3].getBgHsvChannels().at(0),
		m_cameras[3].getBgHsvChannels().at(1),
		m_cameras[3].getBgHsvChannels().at(2),
		channels,
		m_h_threshold,
		m_s_threshold,
		m_v_threshold
	);
	updateTrackbars();

}


/**
 * Deconstructor
 * Free the memory of the floor_grid pointer vector
 */
Scene3DRenderer::~Scene3DRenderer() = default;

/**
 * Process the current frame on each camera
 */
bool Scene3DRenderer::processFrame()
{
	for (auto & camera : m_cameras)
	{
		if (m_current_frame == m_previous_frame + 1)
		{
			camera.advanceVideoFrame();
		}
		else if (m_current_frame != m_previous_frame)
		{
			camera.getVideoFrame(m_current_frame);
		}
		processForeground(camera);
	}

	m_reconstructor.update();

	constexpr uint8_t NUM_CONTOURS = 4;
	constexpr uint8_t NUM_RETRIES = 10;

	auto [centers, labels] = m_clusterLabeler->FindClusters(
		NUM_CONTOURS,
		NUM_RETRIES,
		m_reconstructor.getVoxels(),
		m_reconstructor.getVisibleVoxelIndices());

	std::vector<std::vector<cv::Mat>> masks;
	for (auto & camera : m_cameras)
	{
		masks.push_back(m_clusterLabeler->ProjectTShirt(
			NUM_CONTOURS,
			camera,
			m_reconstructor.getVoxelSize() * 0.5f,
			m_reconstructor.getVoxels(),
			m_reconstructor.getVisibleVoxelIndices(),
			labels));
	}

	m_clusterLabeler->CleanupMasks(masks);
	vector<int> maskToEmNr = m_clusterLabeler->PredictEMS(m_cameras, masks);

	// TODO: Change the order of the colors based on the color modeling
	std::vector<glm::vec4> colors = {
		glm::vec4(1.0f, 0.0f, 0.0f, 1.0f),
		glm::vec4(0.0f, 1.0f, 0.0f, 1.0f),
		glm::vec4(0.0f, 0.0f, 1.0f, 1.0f),
		glm::vec4(0.0f, 0.0f, 0.0f, 1.0f)
	};

	std::vector<glm::vec4> swizzled_colors(4);

	for (int i = 0; i < 4; i++)
	{
		swizzled_colors[i] = colors[maskToEmNr[i]];
	}
	
	m_reconstructor.color(labels, swizzled_colors);

	// TODO: Change the order of the centers based on the color modeling
	m_cluster_traces[maskToEmNr[0]][m_current_frame] = reinterpret_cast<cv::Point2f*>(centers.data)[0] * 0.1;
	m_cluster_traces[maskToEmNr[1]][m_current_frame] = reinterpret_cast<cv::Point2f*>(centers.data)[1] * 0.1;
	m_cluster_traces[maskToEmNr[2]][m_current_frame] = reinterpret_cast<cv::Point2f*>(centers.data)[2] * 0.1;
	m_cluster_traces[maskToEmNr[3]][m_current_frame] = reinterpret_cast<cv::Point2f*>(centers.data)[3] * 0.1;

	m_cluster_traces[0][m_current_frame].x += 250;
	m_cluster_traces[0][m_current_frame].y = 250 - m_cluster_traces[0][m_current_frame].y;
	m_cluster_traces[1][m_current_frame].x += 250;
	m_cluster_traces[1][m_current_frame].y = 250 - m_cluster_traces[1][m_current_frame].y;
	m_cluster_traces[2][m_current_frame].x += 250;
	m_cluster_traces[2][m_current_frame].y = 250 - m_cluster_traces[2][m_current_frame].y;
	m_cluster_traces[3][m_current_frame].x += 250;
	m_cluster_traces[3][m_current_frame].y = 250 - m_cluster_traces[3][m_current_frame].y;

	Mat display = Mat(cv::Size(500, 500), CV_8UC3, Scalar::all(255));

	for (uint32_t i = 2; i < m_current_frame; ++i)
	{
		cv::line(display, m_cluster_traces[0][i - 1], m_cluster_traces[0][i], cv::Scalar(colors[0][2] * 255, colors[0][1] * 255, colors[0][0] * 255));
		cv::line(display, m_cluster_traces[1][i - 1], m_cluster_traces[1][i], cv::Scalar(colors[1][2] * 255, colors[1][1] * 255, colors[1][0] * 255));
		cv::line(display, m_cluster_traces[2][i - 1], m_cluster_traces[2][i], cv::Scalar(colors[2][2] * 255, colors[2][1] * 255, colors[2][0] * 255));
		cv::line(display, m_cluster_traces[3][i - 1], m_cluster_traces[3][i], cv::Scalar(colors[3][2] * 255, colors[3][1] * 255, colors[3][0] * 255));
	}

	cv::imshow("path", display);

	return true;
}

/**
 * Separate the background from the foreground
 * ie.: Create an 8 bit image where only the foreground of the scene is white (255)
 */
void Scene3DRenderer::processForeground(Camera& camera)
{
	assert(!camera.getFrame().empty());
	Mat hsv_image;
	cvtColor(camera.getFrame(), hsv_image, CV_BGR2HSV);  // from BGR to HSV color space

	std::vector<cv::Mat> channels;
	cv::split(hsv_image, channels);  // Split the HSV-channels for further analysis

	cv::Mat foreground = m_foregroundOptimizer->runHSVThresholding(
		camera.getBgHsvChannels().at(0),
		camera.getBgHsvChannels().at(1),
		camera.getBgHsvChannels().at(2),
		channels,
		m_h_threshold,
		m_s_threshold,
		m_v_threshold
	);

	m_foregroundOptimizer->FindContours(foreground);
	m_foregroundOptimizer->SaveMaxContours(1000, 100);
	m_foregroundOptimizer->DrawMaxContours(foreground, true, 255);

	// Improve the foreground image
	camera.setForegroundImage(foreground);
}

/**
 * Set currently visible camera to the given camera id
 */
void Scene3DRenderer::setCamera(
		int camera)
{
	m_camera_view = true;

	if (m_current_camera != camera)
	{
		m_previous_camera = m_current_camera;
		m_current_camera = camera;
		m_arcball_eye.x = m_cameras[camera].getCameraPlane()[0].x;
		m_arcball_eye.y = m_cameras[camera].getCameraPlane()[0].y;
		m_arcball_eye.z = m_cameras[camera].getCameraPlane()[0].z;
		m_arcball_up.x = 0.0f;
		m_arcball_up.y = 0.0f;
		m_arcball_up.z = 1.0f;
	}
}

/**
 * Set the 3D scene to bird's eye view
 */
void Scene3DRenderer::setTopView()
{
	m_camera_view = false;
	if (m_current_camera != -1)
		m_previous_camera = m_current_camera;
	m_current_camera = -1;

	m_arcball_eye = glm::vec3(0.0f, 0.0f, 10000.0f);
	m_arcball_centre = glm::vec3(0.0f, 0.0f, 0.0f);
	m_arcball_up = glm::vec3(0.0f, 1.0f, 0.0f);
}

/**
 * Create a LUT for the floor grid
 */
void Scene3DRenderer::createFloorGrid()
{
	const int size = m_reconstructor.getSize() / m_num;
	const int z_offset = 3;

	// edge 1
	vector<Point3i> edge1;
	for (int y = -size * m_num; y <= size * m_num; y += size)
		edge1.emplace_back(-size * m_num, y, z_offset);

	// edge 2
	vector<Point3i> edge2;
	for (int x = -size * m_num; x <= size * m_num; x += size)
		edge2.emplace_back(x, size * m_num, z_offset);

	// edge 3
	vector<Point3i> edge3;
	for (int y = -size * m_num; y <= size * m_num; y += size)
		edge3.emplace_back(size * m_num, y, z_offset);

	// edge 4
	vector<Point3i> edge4;
	for (int x = -size * m_num; x <= size * m_num; x += size)
		edge4.emplace_back(x, -size * m_num, z_offset);

	m_floor_grid.push_back(edge1);
	m_floor_grid.push_back(edge2);
	m_floor_grid.push_back(edge3);
	m_floor_grid.push_back(edge4);
}

} /* namespace nl_uu_science_gmt */
