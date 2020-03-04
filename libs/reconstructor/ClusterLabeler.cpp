#include "ClusterLabeler.h"
#include "ForegroundOptimizer.h"

#include <opencv2/highgui.hpp> //imshow debug
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp> //EM include, use with cv::ml::EM

#include "Camera.h"

using nl_uu_science_gmt::Camera;
using nl_uu_science_gmt::ClusterLabeler;
using nl_uu_science_gmt::Voxel;







std::pair<cv::Mat, std::vector<int>> ClusterLabeler::FindClusters(uint8_t num_clusters, uint8_t num_retries, const std::vector<Voxel> &voxels, const std::vector<uint32_t> &indices)
{
	// Project the voxels into 2d, ignoring up vector
	std::vector<cv::Point2f> voxels_2d;
	voxels_2d.reserve(indices.size());
	for (auto i : indices) {
		voxels_2d.emplace_back(voxels[i].coordinate.x, voxels[i].coordinate.y);
	}

	// Run k-means for labels and centers
	std::vector<int> labels;
	labels.reserve(voxels_2d.size());
	cv::Mat centers;
	cv::kmeans(voxels_2d, num_clusters, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 1000, 0.001), num_retries, cv::KMEANS_PP_CENTERS, centers);

	return std::make_pair(centers, labels);
}

std::vector<cv::Mat> nl_uu_science_gmt::ClusterLabeler::ProjectTShirt(uint8_t num_clusters, const Camera& camera, float voxel_step_size, const std::vector<Voxel> &voxels, const std::vector<uint32_t> &indices, const std::vector<int> &labels)
{
	constexpr float t_shirt_min_z = 800.0f;
	constexpr float t_shirt_max_z = 1400.0f;
	// Create black textures
	std::vector<cv::Mat> masks(num_clusters);
	for (auto& mask : masks)
	{
		mask = cv::Mat::zeros(camera.getSize(), CV_8U);
	}

	for (uint32_t i = 0; i < labels.size(); ++i)
	{
		auto person_index = labels[i];
		auto voxel_index = indices[i];
		auto& voxel = voxels[voxel_index];
		// Cull voxels which are too low or too high to be part of the shirt
		if (voxel.coordinate.z < t_shirt_min_z || voxel.coordinate.z > t_shirt_max_z)
		{
			continue;
		}
		auto& mask = masks[person_index];
		mask.at<uint8_t>(camera.projectOnView(voxel.coordinate)) = 0xFF;


		mask.at<uint8_t>(camera.projectOnView(cv::Point3f(voxel.coordinate) + cv::Point3f(voxel_step_size, 0, 0))) = 0xFF;
		mask.at<uint8_t>(camera.projectOnView(cv::Point3f(voxel.coordinate) + cv::Point3f(voxel_step_size, voxel_step_size, 0))) = 0xFF;
		mask.at<uint8_t>(camera.projectOnView(cv::Point3f(voxel.coordinate) + cv::Point3f(0, voxel_step_size, 0))) = 0xFF;
		mask.at<uint8_t>(camera.projectOnView(cv::Point3f(voxel.coordinate) + cv::Point3f(0, voxel_step_size, voxel_step_size))) = 0xFF;
		mask.at<uint8_t>(camera.projectOnView(cv::Point3f(voxel.coordinate) + cv::Point3f(0, 0, voxel_step_size))) = 0xFF;
		mask.at<uint8_t>(camera.projectOnView(cv::Point3f(voxel.coordinate) + cv::Point3f(voxel_step_size, 0, voxel_step_size))) = 0xFF;
		mask.at<uint8_t>(camera.projectOnView(cv::Point3f(voxel.coordinate) + cv::Point3f(voxel_step_size, voxel_step_size, voxel_step_size))) = 0xFF;
	}

	return masks;
}

void ClusterLabeler::CleanupMasks(std::vector<std::vector<cv::Mat>> &masks)
{
	ForegroundOptimizer optimizer(1);
	for (auto camera : masks)
	{
		for (auto mask : camera)
		{
			optimizer.FindContours(mask);
			optimizer.SaveMaxContours(1000, 50);
			optimizer.DrawMaxContours(mask);
		}
	}
}

#pragma optimize("", off)

void ClusterLabeler::ShowMaskCutouts(std::vector<std::vector<cv::Mat>>& masks, std::vector<cv::Mat>& hsvImages, std::vector<std::vector<cv::Mat>>& blackendCutouts)
{
	for (int i = 0; i < masks.size(); i++) //loop over cameras
	{
		std::vector<cv::Mat> camMasks;
		for (int j = 0; j < masks[i].size(); j++) //loop over masks
		{
			cv::Mat cutout;
			cv::bitwise_and(hsvImages[i], hsvImages[i], cutout, masks[i][j]);
			camMasks.push_back(cutout.clone());
		}
		blackendCutouts.push_back(camMasks);
	}

	for (uint32_t i = 0; i < masks.size(); ++i)
	{
		for (uint32_t j = 0; j < masks[i].size(); ++j)
		{
			//cv::imshow("mask #" + std::to_string(j), masks[i][j]);
			cvtColor(blackendCutouts[i][j], blackendCutouts[i][j], cv::COLOR_HSV2BGR);
			cv::imshow("mask #" + std::to_string(j), blackendCutouts[i][j]);
		}
		//cv::imshow("camera image", cameras[i].getFrame());
		//cv::waitKey();
		//cv::imshow("camera image", cutouts[i][j]);
		cv::waitKey();
	}
}

void ClusterLabeler::CheckEMS(std::vector<std::vector<cv::Mat>>& reshaped_cutouts)
{
	std::cout << "results checkEMS: \n";
	for (int i = 0; i < reshaped_cutouts.size(); i++)
	{
		for (int j = 0; j < reshaped_cutouts[i].size(); j++)
		{
			cv::Mat results;
			ems[i][j]->predict(reshaped_cutouts[i][j], results);
			//returns values between 0 and chance / clusternrs (so 0.2 max for 5 clusters)
			float value = cv::mean(results)[0];

			std::cout << "cam: " << i << " mask: " << j << " result: " << value << "\n";
		}
	}
	
}

void ClusterLabeler::SaveEMS(std::filesystem::path dataPath)
{
	for (int i = 0; i < ems.size(); i++)
	{
		auto cameraPath = std::filesystem::path("cam" + std::to_string(i + 1));
		for (int j = 0; j < ems[i].size(); j++)
		{
			auto maskPath = std::filesystem::path("maskEM" + std::to_string(j + 1) + ".yml");
			ems[i][j]->save((dataPath / cameraPath / maskPath).u8string());
		}
	}
}

//most useful opencv example: https://github.com/opencv/opencv/blob/master/samples/cpp/em.cpp
//example EM in use: http://seiya-kumada.blogspot.com/2013/03/em-algorithm-practice-by-opencv.html
//documentation EM: https://docs.opencv.org/3.4/d1/dfb/classcv_1_1ml_1_1EM.html#ae3f12147ba846a53601b60c784ee263d
//opencv samle EM: https://github.com/opencv/opencv/blob/master/samples/cpp/em.cpp
void ClusterLabeler::CreateColorScheme(std::vector<std::vector<cv::Mat>>& masks, std::vector<cv::Mat>& hsvImages, std::vector<std::vector<cv::Mat>>& reshaped_cutouts)
{

	using namespace cv;
	using namespace cv::ml;
	using namespace std;

	//One EM per T-shirt. Those are trained in this offline stage. Online EM.predict can be used to return the probability of the match 
	ems.reserve(4);


	ems.resize(masks.size());

	//initialize the EMS
	for (int i = 0; i < masks.size(); i++) //loop over cameras
	{
		for (int j = 0; j < masks[i].size(); j++) //loop over masks
		{
			Ptr<EM>& em = ems[i].emplace_back(EM::create()); //transfer ownership of the Ptr to ems
			em->setClustersNumber(4);
			em->setCovarianceMatrixType(EM::COV_MAT_SPHERICAL);
			em->setTermCriteria(TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 300, 0.1));
		}
	}

	for (int i = 0; i < masks.size(); i++) //loop over cameras
	{
		std::vector<cv::Mat> cam_cutouts;
		for (int j = 0; j < masks[i].size(); j++) //loop over masks
		{
			cv::Mat& mask = masks[i][j];
			int whitePixels = sum(mask)[0] / 255;

			//cutout contains one row, three channels, (for h, s and v) of all pixel values that are on the white part of the mask.
			Mat cutout(Size(whitePixels, 1), CV_64FC3);

			int counter = 0;
			for (int y = 0; y < mask.rows; y++)
			{
				for (int x = 0; x < mask.cols; x++)
				{
					if (mask.at<uchar>(y, x)) //if the mask is white here
					{
						cutout.at<Vec3d>(0, counter) = ((Vec3d)hsvImages[i].at<Vec3b>(y, x)) / 255.0; //add the normalized double pixel value to the cutout
						counter++;
					}
				}
			}

			//contains three columns (h s and v), rows as many pixels as there were in the mask, only 1 channel
			Mat reshaped_cutout = cutout.reshape(1, 3).t(); //put 1 pixel value per row (each sample is 1 pixel) 

			//those two are not in use, but just nice to look at
			cv::Mat likelyhoods; //do log(likelyhood[nr] to find the likelyhood it belongs to the cluster (log(1.7) = 0.25 for example)
			cv::Mat labels; //labels to which cluster each pixel belongs
			ems[i][j]->trainEM(reshaped_cutout, likelyhoods, labels, cv::noArray());

			cam_cutouts.push_back(reshaped_cutout);
		}
		reshaped_cutouts.push_back(cam_cutouts);

		
	}
}

#pragma optimize("", on)