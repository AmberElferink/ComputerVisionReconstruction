#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

int main(int argc, char* argv[])
{
    bool show_usage = false;
    std::string metadata;
    std::string input;
    std::string output;

    // Sanity checks on commandline
    if (argc != 5) {
        std::cerr << "[camera_calibrator] Error: too few arguments: expected 5, got " << argc << "." << std::endl;
        show_usage = true;
    } else {
        metadata = argv[1];
        input = argv[2];
        output = argv[3];
    }
    if (metadata.length() < 4 || metadata.substr(metadata.length() - 4) != ".xml") {
        std::cerr << "[camera_calibrator] Error: checkerboard metadata should end with \".xml\": " << metadata << std::endl;
        show_usage = true;
    }
    if (input.length() < 4 || input.substr(input.length() - 4) != ".avi") {
        std::cerr << "[camera_calibrator] Error: input should end with \".avi\": " << input << std::endl;
        show_usage = true;
    }
    if (output.length() < 4 || output.substr(output.length() - 4) != ".xml") {
        std::cerr << "[camera_calibrator] Error: output should end with \".xml\": " << output << std::endl;
        show_usage = true;
    }
    auto frame_step = std::stol(argv[4], nullptr, 0);
    if (frame_step < 1 || frame_step == ULONG_MAX) {
        std::cerr << "[camera_calibrator] Error: frame step should be at least 1: " << argv[4] << std::endl;
        show_usage = true;
    }
    if (show_usage) {
        std::cerr << "Usage: " << argv[0] << " CHECKERBOARD_METADATA.xml INPUT.avi OUTPUT.xml FRAME_STEP" << std::endl;
        return EXIT_FAILURE;
    }

    // Read metadata describing checkerboard for calibration
    int pattern_x;
    int pattern_y;
    float box_side_length;
    {
        cv::FileStorage file(metadata, cv::FileStorage::READ);
        file["CheckerBoardWidth"] >> pattern_x;
        file["CheckerBoardHeight"] >> pattern_y;
        file["CheckerBoardSquareSize"] >> box_side_length;
        file.release();
    }

    // Read all frames in input video file
    cv::VideoCapture capture;
    capture.open(input);
    cv::Mat frame;
    capture >> frame;

    cv::Size image_size;

    if (!frame.empty()) {
        image_size = cv::Size(frame.rows, frame.cols);
    }

    cv::Mat frame_mono(frame.rows, frame.cols, CV_8U);

    // Generate the world-space points of checkerboard based on known real-world size
    std::vector<cv::Point3f> world_space_points(pattern_x * pattern_y);
    for (int j = 0; j < pattern_y; j++)
    {
        for (int i = 0; i < pattern_x; i++)
        {
            world_space_points[i + j * pattern_x] = cv::Vec3f(
                static_cast<float>(i) * box_side_length,
                static_cast<float>(j) * box_side_length,
                0);
        }
    }

    // Find all image-space points in video for frames where the board is detected
    std::vector<std::vector<cv::Point2f>> all_image_space_points;
    // Duplicate world-space points so the arrays are the same length
    std::vector<std::vector<cv::Point3f>> all_world_space_points;
    while (!frame.empty()) {
        // Discard color information
        cv::cvtColor(frame, frame_mono, cv::COLOR_BGR2GRAY);
        std::vector<cv::Point2f> image_space_points;
        if (cv::findChessboardCorners(
            frame_mono,
            cv::Size(pattern_x, pattern_y),
            image_space_points,
            cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK)) {
            // Checkerboard detected on this frame, add to accumulated points
            all_image_space_points.emplace_back(std::move(image_space_points));
            all_world_space_points.push_back(world_space_points);
        }
        for (uint32_t i = 0; i < frame_step; ++i) {
            capture >> frame;
        }
    }
    capture.release();

    // Fail if no patterns were gotten
    if (all_image_space_points.empty()) {
        std::cerr << "No usable frames found in " << input << std::endl;
        return EXIT_FAILURE;
    }

    // Get intrinsic from detected points
    cv::Mat camera_matrix;
    cv::Mat distortion_coefficients;
    cv::Mat rotation;
    cv::Mat translation;
    cv::calibrateCamera(all_world_space_points, all_image_space_points, image_size,
                        camera_matrix, distortion_coefficients, rotation, translation);

    // Write output
    {
        cv::FileStorage file(output, cv::FileStorage::WRITE);
        file << "CameraMatrix" << camera_matrix;
        file << "DistortionCoeffs" << distortion_coefficients.t();
        file.release();
    }

    return EXIT_SUCCESS;
}
