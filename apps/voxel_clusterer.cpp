#include <cstdlib>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <Camera.h>
#include <Reconstructor.h>
#include <ForegroundOptimizer.h>
#include <ClusterLabeler.h>

using nl_uu_science_gmt::Camera;
using nl_uu_science_gmt::ClusterLabeler;
using nl_uu_science_gmt::ForegroundOptimizer;
using nl_uu_science_gmt::Reconstructor;

#define SHOW_RESULTS 1

int main(int argc, char* argv[])
{
    bool show_usage = false;

    constexpr uint32_t NUM_CONTOURS = 4;
    constexpr uint32_t NUM_VIEWS = 4;
    constexpr uint32_t NUM_RETRIES = 40;
    constexpr uint32_t COLOR_CALIBRATION_FRAME_NUMBER = 2 * 672; //949;// (1180 / 2);
    std::vector<Camera> cameras;
    std::filesystem::path data_path = "../data";
    std::filesystem::path config_file_path = "config.xml";
    std::filesystem::path background_file_path = "background.png";
    std::filesystem::path video_file_path = "video.avi";

	std::vector<cv::Mat> hsvImages; //one per camera
	hsvImages.reserve(NUM_VIEWS);

    for (uint32_t i = 0; i < NUM_VIEWS; ++i)
    {
        auto full_path = data_path / ("cam" + std::to_string(i + 1));
        auto& camera = cameras.emplace_back(full_path, config_file_path, i);
        if (!camera.initialize(background_file_path, video_file_path)) {
            return EXIT_FAILURE;
        }
        // TODO: Instead of video file, use input from single image in data/
        camera.getVideoFrame(COLOR_CALIBRATION_FRAME_NUMBER);
        if (camera.getFrame().empty()) {
            std::cerr << "[voxel_clusterer] Error: frame " << COLOR_CALIBRATION_FRAME_NUMBER << " is empty." << std::endl;
            return EXIT_FAILURE;
        }
        cv::Mat hsv_image;
        cvtColor(camera.getFrame(), hsv_image, cv::COLOR_BGR2HSV);  // from BGR to HSV color space
		hsvImages.push_back(hsv_image.clone()); //deep copy

        std::vector<cv::Mat> channels;
        cv::split(hsv_image, channels);  // Split the HSV-channels for further analysis

        uint8_t h_threshold = 0;
        uint8_t s_threshold = 19; //255;
        uint8_t v_threshold = 48; //255;
        int thresholdMaxNoise = 15;

        ForegroundOptimizer foregroundOptimizer(NUM_CONTOURS);

        cv::Mat foreground = foregroundOptimizer.runHSVThresholding(
            camera.getBgHsvChannels().at(0),
            camera.getBgHsvChannels().at(1),
            camera.getBgHsvChannels().at(2),
            channels,
            h_threshold,
            s_threshold,
            v_threshold
        );

        foregroundOptimizer.FindContours(foreground);
        foregroundOptimizer.SaveMaxContours(1000, 500);
		
		foregroundOptimizer.DrawMaxContours(foreground);
        // Improve the foreground image
        camera.setForegroundImage(foreground);

        //cv::imshow("foreground", foreground);
        //cv::waitKey();
    }

    Reconstructor reconstructor(cameras);
    reconstructor.update();

    ClusterLabeler labeler(NUM_CONTOURS, NUM_VIEWS);
    auto [centers, labels] = labeler.FindClusters(
        NUM_CONTOURS,
        NUM_RETRIES,
        reconstructor.getVoxels(),
        reconstructor.getVisibleVoxelIndices());

    std::vector<std::vector<cv::Mat>> masks;
    for (const auto& camera : cameras)
    {
        masks.push_back(labeler.ProjectTShirt(
            NUM_CONTOURS,
            camera,
            reconstructor.getVoxelSize() * 0.5f,
            reconstructor.getVoxels(),
            reconstructor.getVisibleVoxelIndices(),
            labels));
    }

	labeler.CleanupMasks(masks);
	labeler.InitializeEMS();
	std::vector<std::vector<cv::Mat>> reshaped_cutouts; // vector per camera, vector per mask, cut out color pixels in a list
	labeler.TrainEMS(masks, hsvImages, reshaped_cutouts);

	labeler.SaveEMS(data_path);

#if SHOW_RESULTS
	labeler.CheckEMS(reshaped_cutouts);
	std::vector<std::vector<cv::Mat>> cutouts; //black surrounded masked images
	labeler.ShowMaskCutouts(masks, hsvImages, cutouts);
#endif // SHOW_RESULTS

    return EXIT_SUCCESS;
}
