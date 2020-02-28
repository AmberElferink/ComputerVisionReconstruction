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

#define SHOW_RESULTS 0

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

        std::vector<cv::Mat> channels;
        cv::split(hsv_image, channels);  // Split the HSV-channels for further analysis

        uint8_t h_threshold = 0;
        uint8_t s_threshold = 19; //255;
        uint8_t v_threshold = 48; //255;
        int thresholdMaxNoise = 15;

        ForegroundOptimizer foregroundOptimizer(NUM_CONTOURS);
        // TODO(amber): Debug this
//        foregroundOptimizer.optimizeThresholds(
//            thresholdMaxNoise,
//            thresholdMaxNoise,
//            camera.getBgHsvChannels().at(0),
//            camera.getBgHsvChannels().at(1),
//            camera.getBgHsvChannels().at(2),
//            channels,
//            h_threshold,
//            s_threshold,
//            v_threshold
//        );
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
        foregroundOptimizer.SaveMaxContours();

        // Improve the foreground image
        camera.setForegroundImage(foreground);

//        cv::imshow("foreground", foreground);
//        cv::waitKey();
    }

    Reconstructor reconstructor(cameras);
    reconstructor.update();

    ClusterLabeler labeler;
    auto [centers, labels] = labeler.FindClusters(
        NUM_CONTOURS,
        NUM_RETRIES,
        reconstructor.getVoxels(),
        reconstructor.getVisibleVoxelIndices());

    auto masks = labeler.ProjectTShirt(
        NUM_CONTOURS,
        cameras,
        reconstructor.getVoxelSize() * 0.5f,
        reconstructor.getVoxels(),
        reconstructor.getVisibleVoxelIndices(),
        labels);

    cv::FileStorage fs;
    fs.open(data_path / "centers.xml", cv::FileStorage::WRITE);
    if (fs.isOpened())
    {
        fs << "Centers" << centers;
        fs.release();
    }
    else
    {
        std::cerr << "[extrinsics_configurator] Error: Unable to k-means centers to: " << data_path / "centers.xml" << std::endl;
        return EXIT_FAILURE;
    }

    for (uint32_t i = 0; i < cameras.size(); ++i)
    {
        for (uint32_t j = 0; j < NUM_CONTOURS; ++ j)
        {
            cv::imwrite(data_path / ("cam" + std::to_string(i + 1)) / ("mask" + std::to_string(j + 1) + ".png"),
                        masks[i][j],
                        {cv::IMWRITE_PNG_COMPRESSION, 0});
        }
    }

#if SHOW_RESULTS
    for (uint32_t i = 0; i < cameras.size(); ++i)
    {
        for (uint32_t j = 0; j < NUM_CONTOURS; ++ j)
        {
            cv::imshow("mask #" + std::to_string(j), masks[i][j]);
        }
        cv::imshow("camera image", cameras[i].getFrame());
        cv::waitKey();
    }
#endif // SHOW_RESULTS

    return EXIT_SUCCESS;
}
