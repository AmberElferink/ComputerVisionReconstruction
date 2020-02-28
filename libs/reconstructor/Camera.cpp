/*
 * Camera.cpp
 *
 *  Created on: Nov 13, 2013
 *      Author: coert
 */

#include "Camera.h"

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cassert>
#include <iostream>
#include <utility>

using namespace cv;

namespace nl_uu_science_gmt
{

Camera::Camera(
		std::filesystem::path dp, std::filesystem::path cp, const int id) :
				m_data_path(std::move(dp)),
				m_cam_props_file(std::move(cp)),
				m_id(id)
{
	m_initialized = false;

	m_fx = 0;
	m_fy = 0;
	m_cx = 0;
	m_cy = 0;
	m_frame_amount = 0;
}

Camera::~Camera() = default;

/**
 * Initialize this camera
 */
bool Camera::initialize(const std::filesystem::path &background_image_file, const std::filesystem::path &video_file)
{
	m_initialized = true;

	Mat bg_image;
	if (std::filesystem::exists(m_data_path / background_image_file))
	{
		bg_image = imread(m_data_path / background_image_file);
		if (bg_image.empty())
		{
			std::cout << "Unable to read: " << m_data_path / background_image_file << std::endl;
			return false;
		}
	}
	else
	{
		std::cout << "Unable to find background image: " << m_data_path / background_image_file << std::endl;
		return false;
	}
	assert(!bg_image.empty());

	// Disect the background image in HSV-color space
	Mat bg_hsv_im;
	cvtColor(bg_image, bg_hsv_im, cv::COLOR_BGR2HSV);
	split(bg_hsv_im, m_bg_hsv_channels);

	// Open the video for this camera
	m_video = VideoCapture(m_data_path / video_file);
	assert(m_video.isOpened());

	// Assess the image size
	m_plane_size.width = (int) m_video.get(cv::CAP_PROP_FRAME_WIDTH);
	m_plane_size.height = (int) m_video.get(cv::CAP_PROP_FRAME_HEIGHT);
	assert(m_plane_size.area() > 0);

	// Get the amount of video frames
	m_video.set(cv::CAP_PROP_POS_AVI_RATIO, 1);  // Go to the end of the video; 1 = 100%
	m_frame_amount = (long) m_video.get(cv::CAP_PROP_POS_FRAMES);
	assert(m_frame_amount > 1);
	m_video.set(cv::CAP_PROP_POS_AVI_RATIO, 0);  // Go back to the start

	m_video.release(); //Re-open the file because _video.set(CV_CAP_PROP_POS_AVI_RATIO, 1) may screw it up
	m_video = cv::VideoCapture(m_data_path / video_file);

	// Read the camera properties (XML)
	FileStorage fs;
	fs.open(m_data_path / m_cam_props_file, FileStorage::READ);
	if (fs.isOpened())
	{
		Mat cam_mat, dis_coe, rot_val, tra_val;
		fs["CameraMatrix"] >> cam_mat;
		fs["DistortionCoeffs"] >> dis_coe;
		fs["RotationValues"] >> rot_val;
		fs["TranslationValues"] >> tra_val;

		cam_mat.convertTo(m_camera_matrix, CV_32F);
		dis_coe.convertTo(m_distortion_coeffs, CV_32F);
		rot_val.convertTo(m_rotation_values, CV_32F);
		tra_val.convertTo(m_translation_values, CV_32F);

		fs.release();

		/*
		 * [ [ fx  0 cx ]
		 *   [  0 fy cy ]
		 *   [  0  0  0 ] ]
		 */
		m_fx = m_camera_matrix.at<float>(0, 0);
		m_fy = m_camera_matrix.at<float>(1, 1);
		m_cx = m_camera_matrix.at<float>(0, 2);
		m_cy = m_camera_matrix.at<float>(1, 2);
	}
	else
	{
		std::cerr << "Unable to locate: " << m_data_path << m_cam_props_file << std::endl;
		m_initialized = false;
	}

	initCamLoc();
	camPtInWorld();

	return m_initialized;
}

/**
 * Set and return the next frame from the video
 */
Mat& Camera::advanceVideoFrame()
{
	m_video >> m_frame;
	assert(!m_frame.empty());
	return m_frame;
}

/**
 * Set the video location to the given frame number
 */
void Camera::setVideoFrame(
		int frame_number)
{
	m_video.set(cv::CAP_PROP_POS_FRAMES, frame_number);
}

/**
 * Set and return frame of the video location at the given frame number
 */
Mat& Camera::getVideoFrame(
		int frame_number)
{
	setVideoFrame(frame_number);
	return advanceVideoFrame();
}

/**
 * Calculate the camera's location in the world
 */
void Camera::initCamLoc()
{
	Mat r;
	Rodrigues(m_rotation_values, r);

	/*
	 * [ [ r11 r12 r13   0 ]
	 *   [ r21 r22 r23   0 ]
	 *   [ r31 r32 r33   0 ]
	 *   [   0   0   0 1.0 ] ]
	 */
	Mat rotation = Mat::zeros(4, 4, CV_32F);
	rotation.at<float>(3, 3) = 1.0;
	Mat r_sub = rotation(Rect(0, 0, 3, 3));
	r.copyTo(r_sub);

	/*
	 * [ [ 1.0   0   0   0 ]
	 *   [   0 1.0   0   0 ]
	 *   [   0   0 1.0   0 ]
	 *   [  tx  ty  tz 1.0 ] ]
	 */
	Mat translation = Mat::eye(4, 4, CV_32F);
	translation.at<float>(3, 0) = -m_translation_values.at<float>(0, 0);
	translation.at<float>(3, 1) = -m_translation_values.at<float>(1, 0);
	translation.at<float>(3, 2) = -m_translation_values.at<float>(2, 0);

	Mat camera_mat = translation * rotation;
	m_camera_location = Point3f(
			camera_mat.at<float>(0, 0) + camera_mat.at<float>(3, 0),
			camera_mat.at<float>(1, 1) + camera_mat.at<float>(3, 1),
			camera_mat.at<float>(2, 2) + camera_mat.at<float>(3, 2));

	std::cout << "Camera " << m_id + 1 << " " << m_camera_location << std::endl;

	m_rt = rotation;

	/*
	 * [ [ r11 r12 r13 tx ]
	 *   [ r21 r22 r23 ty ]
	 *   [ r31 r32 r33 tz ]
	 *   [   0   0   0  0 ] ]
	 */
	Mat t_sub = m_rt(Rect(3, 0, 1, 3));
	m_translation_values.copyTo(t_sub);

	invert(m_rt, m_inverse_rt);
}

/**
 * Calculate the camera's plane and fov in the 3D scene
 */
void Camera::camPtInWorld()
{
	m_camera_plane.clear();
	m_camera_plane.push_back(m_camera_location);

	// clockwise four image plane corners
	// 1 image plane's left upper corner
	Point3f p1 = cam3DtoW3D(Point3f(-m_cx, -m_cy, (m_fx + m_fy) / 2));
	m_camera_plane.push_back(p1);
	// 2 image plane's right upper conner
	Point3f p2 = cam3DtoW3D(Point3f(m_plane_size.width - m_cx, -m_cy, (m_fx + m_fy) / 2));
	m_camera_plane.push_back(p2);
	// 3 image plane's right bottom conner
	Point3f p3 = cam3DtoW3D(Point3f(m_plane_size.width - m_cx, m_plane_size.height - m_cy, (m_fx + m_fy) / 2));
	m_camera_plane.push_back(p3);
	// 4 image plane's left bottom conner
	Point3f p4 = cam3DtoW3D(Point3f(-m_cx, m_plane_size.height - m_cy, (m_fx + m_fy) / 2));
	m_camera_plane.push_back(p4);

	// principal point on the image plane
	Point3f p5 = cam3DtoW3D(Point3f(m_cx, m_cy, (m_fx + m_fy) / 2));
	m_camera_plane.push_back(p5);
}

/**
 * Convert a point on the camera image to a point in the world
 */
Point3f Camera::ptToW3D(
		const Point &point)
{
	return cam3DtoW3D(Point3f(float(point.x - m_cx), float(point.y - m_cy), (m_fx + m_fy) / 2));
}

/**
 * Convert a point on the camera to a point in the world
 */
Point3f Camera::cam3DtoW3D(
		const Point3f &cam_point)
{
	Mat Xc(4, 1, CV_32F);
	Xc.at<float>(0, 0) = cam_point.x;
	Xc.at<float>(1, 0) = cam_point.y;
	Xc.at<float>(2, 0) = cam_point.z;
	Xc.at<float>(3, 0) = 1;

	Mat Xw = m_inverse_rt * Xc;

	return Point3f(Xw.at<float>(0, 0), Xw.at<float>(1, 0), Xw.at<float>(2, 0));
}

/**
 * Projects points from the scene space to the image coordinates
 */
cv::Point Camera::projectOnView(
		const cv::Point3f &coords, const cv::Mat &rotation_values, const cv::Mat &translation_values, const cv::Mat &camera_matrix,
		const cv::Mat &distortion_coeffs)
{
	std::vector<Point3f> object_points;
	object_points.push_back(coords);

	std::vector<Point2f> image_points;
	projectPoints(object_points, rotation_values, translation_values, camera_matrix, distortion_coeffs, image_points);

	return image_points.front();
}

/**
 * Non-static for backwards compatibility
 */
Point Camera::projectOnView(const Point3f &coords) const
{
	return projectOnView(coords, m_rotation_values, m_translation_values, m_camera_matrix, m_distortion_coeffs);
}

} /* namespace nl_uu_science_gmt */
