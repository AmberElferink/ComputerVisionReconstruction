#include <cstdlib>
#include <Camera.h>
#include <Reconstructor.h>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <ForegroundOptimizer.h>
#include <opencv2/highgui.hpp>

using nl_uu_science_gmt::Camera;
using nl_uu_science_gmt::Reconstructor;

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

    std::vector<cv::Point2f> voxels_2d;
    voxels_2d.reserve(reconstructor.getVisibleVoxelIndices().size());
    for (auto i : reconstructor.getVisibleVoxelIndices()) {
        voxels_2d.emplace_back(reconstructor.getVoxels()[i].coordinate.x, reconstructor.getVoxels()[i].coordinate.y);
    }

    std::vector<int> labels;
    labels.reserve(voxels_2d.size());
    cv::Mat centers;
    cv::kmeans(voxels_2d, NUM_CONTOURS, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 1000, 0.001), NUM_RETRIES, cv::KMEANS_PP_CENTERS, centers);

    {
        std::vector<cv::Mat> masks(cameras.size() * NUM_CONTOURS);
        for (auto& mask : masks)
        {
            mask = cv::Mat::zeros(cameras.front().getSize(), CV_8U);
        }
        auto step = reconstructor.getVoxelSize() * 0.5f;
        for (uint32_t i = 0; i < labels.size(); ++i)
        {
            auto person_index = labels[i];
            auto voxel_index = reconstructor.getVisibleVoxelIndices()[i];
            auto& voxel = reconstructor.getVoxels()[voxel_index];
            for (uint32_t j = 0; j < cameras.size(); ++j)
            {
                auto& camera = cameras[j];
                auto& mask = masks[j * NUM_CONTOURS + person_index];
                mask.at<uint8_t>(camera.projectOnView(voxel.coordinate)) = 0xFF;
                mask.at<uint8_t>(camera.projectOnView(cv::Point3f(voxel.coordinate) + cv::Point3f(step, 0, 0))) = 0xFF;
                mask.at<uint8_t>(camera.projectOnView(cv::Point3f(voxel.coordinate) + cv::Point3f(step, step, 0))) = 0xFF;
                mask.at<uint8_t>(camera.projectOnView(cv::Point3f(voxel.coordinate) + cv::Point3f(0, step, 0))) = 0xFF;
                mask.at<uint8_t>(camera.projectOnView(cv::Point3f(voxel.coordinate) + cv::Point3f(0, step, step))) = 0xFF;
                mask.at<uint8_t>(camera.projectOnView(cv::Point3f(voxel.coordinate) + cv::Point3f(0, 0, step))) = 0xFF;
                mask.at<uint8_t>(camera.projectOnView(cv::Point3f(voxel.coordinate) + cv::Point3f(step, 0, step))) = 0xFF;
                mask.at<uint8_t>(camera.projectOnView(cv::Point3f(voxel.coordinate) + cv::Point3f(step, step, step))) = 0xFF;
            }
        }

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

        for (uint32_t j = 0; j < NUM_CONTOURS; ++ j)
        {
            for (uint32_t i = 0; i < cameras.size(); ++i)
            {
                cv::imwrite(data_path / ("cam" + std::to_string(i + 1)) / ("mask" + std::to_string(j + 1) + ".png"),
                            masks[j + i * NUM_CONTOURS],
                            {cv::IMWRITE_PNG_COMPRESSION, 0});
            }
        }
    }

    return EXIT_FAILURE;
}
