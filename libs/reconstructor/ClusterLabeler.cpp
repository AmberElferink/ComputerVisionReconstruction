#include "ClusterLabeler.h"
#include "ForegroundOptimizer.h"

#include <opencv2/highgui.hpp> //imshow debug
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp> //EM include, use with cv::ml::EM

#include <unordered_set> //to keep track of which masks have been matched

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

void ClusterLabeler::SaveEMS(const std::filesystem::path& dataPath)
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

void ClusterLabeler::LoadEMS(const std::filesystem::path& dataPath)
{
	InitializeEMS();

	for (int i = 0; i < m_numCameras; i++)
	{
		auto cameraPath = std::filesystem::path("cam" + std::to_string(i + 1));
		for (int j = 0; j < m_numClusters; j++)
		{
			auto maskPath = std::filesystem::path("maskEM" + std::to_string(j + 1) + ".yml");
			ems[i][j] = cv::ml::EM::load((dataPath / cameraPath / maskPath).u8string());
		}
	}
}

void ClusterLabeler::InitializeEMS()
{

	using namespace cv;
	using namespace cv::ml;
	using namespace std;

	ems.resize(m_numCameras);

	//initialize the EMS
	for (int i = 0; i < m_numCameras; i++) //loop over cameras
	{
		for (int j = 0; j < m_numClusters; j++) //loop over masks
		{
			Ptr<EM>& em = ems[i].emplace_back(EM::create()); //transfer ownership of the Ptr to ems
			em->setClustersNumber(m_numClusters);
			em->setCovarianceMatrixType(EM::COV_MAT_SPHERICAL);
			em->setTermCriteria(TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 300, 0.1));
		}
	}
}

cv::Mat ClusterLabeler::GetCutout(const cv::Mat& mask, const cv::Mat& hsv_image)
{

	using namespace cv;
	using namespace cv::ml;
	using namespace std;

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
				cutout.at<Vec3d>(0, counter) = ((Vec3d)hsv_image.at<Vec3b>(y, x)) / 255.0; //add the normalized double pixel value to the cutout
				counter++;
			}
		}
	}

	//contains three columns (h s and v), rows as many pixels as there were in the mask, only 1 channel
	Mat reshaped_cutout = cutout.reshape(1, 3).t(); //put 1 pixel value per row (each sample is 1 pixel) 

	return reshaped_cutout;
}

//most useful opencv example: https://github.com/opencv/opencv/blob/master/samples/cpp/em.cpp
//example EM in use: http://seiya-kumada.blogspot.com/2013/03/em-algorithm-practice-by-opencv.html
//documentation EM: https://docs.opencv.org/3.4/d1/dfb/classcv_1_1ml_1_1EM.html#ae3f12147ba846a53601b60c784ee263d
//opencv samle EM: https://github.com/opencv/opencv/blob/master/samples/cpp/em.cpp
void ClusterLabeler::TrainEMS(std::vector<std::vector<cv::Mat>>& masks, std::vector<cv::Mat>& hsvImages, std::vector<std::vector<cv::Mat>>& reshaped_cutouts)
{

	using namespace cv;
	using namespace cv::ml;
	using namespace std;

	for (int i = 0; i < masks.size(); i++) //loop over cameras
	{
		std::vector<cv::Mat> cam_cutouts;
		for (int j = 0; j < masks[i].size(); j++) //loop over masks
		{
			cv::Mat& mask = masks[i][j];
			

			cv::Mat reshaped_cutout = GetCutout(mask, hsvImages[i]);

			//those two are not in use, but just nice to look at
			cv::Mat likelyhoods; //do log(likelyhood[nr] to find the likelyhood it belongs to the cluster (log(1.7) = 0.25 for example)
			cv::Mat labels; //labels to which cluster each pixel belongs
			ems[i][j]->trainEM(reshaped_cutout, likelyhoods, labels, cv::noArray());

			cam_cutouts.push_back(reshaped_cutout);
		}
		reshaped_cutouts.push_back(cam_cutouts);

		
	}
}

//given a matrix with indices for the sorted rows, reconstruct the sorted matrix from the original
cv::Mat ReconstructFromRowIndices(cv::Mat matrix, cv::Mat matrixRowIndices)
{
	using namespace cv;
	Mat sortedRowMatrix(matrix.rows, matrix.cols, CV_32FC1);
	for (int row = 0; row < matrixRowIndices.rows; row++)
	{
		for (int col = 0; col < matrixRowIndices.cols; col++)
		{
			int puckIDIndex = matrixRowIndices.at<int>(row, col);
			float distance = matrix.at<int>(row, puckIDIndex);
			sortedRowMatrix.at<int>(row, col) = distance;
		}
	}
	return sortedRowMatrix;
}

//input a CV_8UC1 matrix with row indices for instance 3 rows 4 cols
//input the sorted col indices (only 1 column) in this case 4.
//at these indexes is the minimum distance between the new puck and the old puck. 
//it zips those two together to a CV_8UC2 matrix.
cv::Mat ZipIndices(cv::Mat matrix, cv::Mat matrixRowIndices, cv::Mat matrixColIndices)
{
	using namespace cv;
	Mat combinedIndices(matrix.rows, matrix.cols, CV_8UC2);

	//zip the indices together
	for (int i = 0; i < matrixRowIndices.cols; i++)
	{
		for (int j = 0; j < matrixRowIndices.rows; j++)
		{
			//The number corresponds to the indexes that need to be linked to eachother.
			int colSortedIndice = matrixColIndices.at<int>(j, i);
			//since you continue with column sort after the row, you should get the rowindex corresponding to the column index
			int rowSortedIndice = matrixRowIndices.at<int>(colSortedIndice, i);
			combinedIndices.at<Vec2b>(j, i)[0] = rowSortedIndice;
			combinedIndices.at<Vec2b>(j, i)[1] = colSortedIndice;
		}
	}
	return combinedIndices;
}

//Input Mat with CV_8UC2. It will be printed like this:
//[(1,3), (4,6), (6,4);
// (3,4), (4,4), (2,5);
// (2,8), (7,2), (5,2);]
void PrintZippedIndices(cv::Mat combinedIndices)
{
	std::string print = "[";
	for (int j = 0; j < combinedIndices.rows; j++)
	{
		for (int i = 0; i < combinedIndices.cols; i++)
		{
			print += "(" + std::to_string(combinedIndices.at<cv::Vec2b>(j, i)[0]) + ", " + std::to_string(combinedIndices.at<cv::Vec2b>(j, i)[1]) + "), ";
		}
		print += "; \n ";
	}
	print += "]";
	std::cout << print << std::endl;
}

//input matrix must be of type CV_8UC1
//outputs a matrix with type CV_8UC2 with the indices of sortIdx.
cv::Mat GetSortedMatIndices(cv::Mat matrix, bool verbalDebug)
{
	using namespace cv;

	Mat matrixRowIndices(matrix.rows, matrix.cols, CV_8UC1);
	sortIdx(matrix, matrixRowIndices, SORT_EVERY_ROW + SORT_DESCENDING);

	if (verbalDebug)
		std::cout << "row indices" << std::endl << matrixRowIndices << std::endl << std::endl;

	Mat sortedRowMatrix = ReconstructFromRowIndices(matrix, matrixRowIndices);

	if (verbalDebug)
		std::cout << "sorted row matrix" << std::endl << sortedRowMatrix << std::endl << std::endl;

	Mat colIndices(matrix.rows, matrix.cols, CV_8UC1);
	sortIdx(sortedRowMatrix, colIndices, SORT_EVERY_COLUMN + SORT_DESCENDING);

	if (verbalDebug)
		std::cout << "col indices" << std::endl << colIndices << std::endl << std::endl;

	Mat zippedIndices = ZipIndices(matrix, matrixRowIndices, colIndices);

	if (verbalDebug)
	{
		std::cout << "zippedIndices: " << std::endl;
		PrintZippedIndices(zippedIndices);
	}

	return zippedIndices.clone();
}



//checks if a unsigned set contains a certain element
bool uSetContains(std::unordered_set<int> set, int toFind)
{
	if (set.find(toFind) == set.end())
	{
		return false;
	}
	else
	{
		return true;
	}
}

//delete col row: https://stackoverflow.com/questions/29696805/what-is-the-best-way-to-remove-a-row-or-col-from-a-cv-mat
void DeleteRow(const cv::Mat& matIn, cv::Mat& matOut, int row)
{
	matOut = cv::Mat(cv::Size(matIn.cols, matIn.rows - 1), matIn.type());
	if (row > 0) // Copy everything above that one row.
	{
		cv::Rect rect(0, 0, matIn.cols, row);
		matIn(rect).copyTo(matOut(rect));
	}

	if (row < matIn.rows - 1) // Copy everything below that one row.
	{
		cv::Rect rect1(0, row + 1, matIn.cols, matIn.rows - row - 1);
		cv::Rect rect2(0, row, matIn.cols, matIn.rows - row - 1);
		matIn(rect1).copyTo(matOut(rect2));
	}
}

void DeleteCol(const cv::Mat& matIn, cv::Mat& matOut, int col)
{
	matOut = cv::Mat(cv::Size(matIn.cols - 1, matIn.rows), matIn.type());
	if (col > 0) // Copy everything left of that one column.
	{
		cv::Rect rect(0, 0, col, matIn.rows);
		matIn(rect).copyTo(matOut(rect));
	}

	if (col < matIn.cols - 1) // Copy everything right of that one column.
	{
		cv::Rect rect1(col + 1, 0, matIn.cols - col - 1, matIn.rows);
		cv::Rect rect2(col, 0, matIn.cols - col - 1, matIn.rows);
		matIn(rect1).copyTo(matOut(rect2));
	}
}

//increase the mask/emNr accounting for deleted rows
void setCorrectMaskEMnr(const std::unordered_set<int>& usedMasks, int& maskNr, std::unordered_set<int>& usedEMs, int& emNr)
{
	bool updating = true;
	while (updating) // if one of the numbers had been increased, keep going. Otherwise return
	{
		if (uSetContains(usedMasks, maskNr))
		{
			maskNr++;
			continue;
		}
		if (uSetContains(usedEMs, emNr))
		{
			emNr++;
			continue;
		}
		return;
	}
	
}

//matches the highest probable colors found int the image to the color models
//it returns a vector, at index model there is a mask
std::vector<int> MatchMaskToEM(cv::Mat& probabilities)
{
	std::vector<int> output(probabilities.cols);


	std::cout << probabilities << std::endl;
	//each number can only be present once, the numbers are not ordered.
	//faster than set or vector
	std::unordered_set<int> usedMasks;
	std::unordered_set<int> usedEMs;

	cv::Mat zippedIndices = GetSortedMatIndices(probabilities, false);

	//loop through all zipped columns  and rows to link pucks to eachother
	while(probabilities.cols > 0)
	{
		PrintZippedIndices(zippedIndices);

		int maskNr = zippedIndices.at<cv::Vec2b>(0, 0)[0]; //this was sorted per row, which is horizontal, so the sorting gives the COLUMN you need, which is the mask
		int emNr = zippedIndices.at<cv::Vec2b>(0, 0)[1]; //this was sorted per column, which is vertical, so the sorting gives the ROW you need, which is the model

		//delete corresponding mask and em row so they don't get matched anymore
		cv::Mat nextProbabilities1;
		DeleteCol(probabilities, nextProbabilities1, maskNr);
		cv::Mat nextProbabilities2;
		DeleteRow(nextProbabilities1, nextProbabilities2, emNr);

		probabilities = nextProbabilities2;

		if (nextProbabilities2.cols > 0)
		{
			zippedIndices = GetSortedMatIndices(nextProbabilities2, false);
		}

		//update mask and em number accounting for  deleted rows and columns in rounds BEFORE this one
		setCorrectMaskEMnr(usedMasks, maskNr, usedEMs, emNr);

		//std::cout << "mask: " << maskNr << " matched with: emnr: " << emNr << std::endl;
		output[maskNr] = emNr;

		usedMasks.insert(maskNr);
		usedEMs.insert(emNr);

	}

	return output;
}


std::vector<int> ClusterLabeler::PredictEMS(const std::vector<Camera>& cameras, const std::vector<std::vector<cv::Mat>>& masks_per_camera)
{
	using namespace cv;
	using namespace cv::ml;
	using namespace std;

	cv::Mat probs_matching_masks = cv::Mat::zeros(cv::Size(m_numClusters, m_numClusters), CV_32FC1);

	for (int i = 0; i < ems.size(); i++) //loop over out vector (camera's), which is the same for ems and masks
	{
		auto& camera = cameras[i];
		auto& models = ems[i];
		auto& masks = masks_per_camera[i];
		auto& frame = camera.getFrame();
		cv::Mat hsv_image;
		cv::cvtColor(frame, hsv_image, cv::COLOR_BGR2HSV);

		for (int j = 0; j < models.size(); j++) //loop over inner trained EM vector ems
		{
			for (int maskI = 0; maskI < masks.size(); maskI++)
			{
				cv::Mat cutout = GetCutout(masks[maskI], hsv_image);
				int correctVotes = 0;
				for (int pixel = 0; pixel < cutout.rows; pixel++)
				{
					Vec2d results = models[j]->predict2(cutout.at<Vec3d>(pixel), noArray());
					float prob = exp(results[0]);
					if (prob > 0.15)
					{
						correctVotes++;
					}
				}
				int totalPixels = cv::sum(masks[maskI])[0] / 255;
				float normalizedVotes = (float) correctVotes / (float) totalPixels;

				probs_matching_masks.at<float>(j, maskI) += normalizedVotes; //m[0];
			}
		}
	}

	probs_matching_masks = probs_matching_masks / m_numCameras;

	vector<int> modelEmMatch = MatchMaskToEM(probs_matching_masks);

	return modelEmMatch;
}


