#include <cstdlib>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

cv::Point projectOnView(
    const cv::Point3f &coords, const cv::Mat &rotation_values, const cv::Mat &translation_values, const cv::Mat &camera_matrix,
    const cv::Mat &distortion_coeffs)
{
    std::vector<cv::Point2f> image_points;
    projectPoints(std::vector<cv::Point3f> {coords}, rotation_values, translation_values, camera_matrix, distortion_coeffs, image_points);
    return image_points.front();
}

int main(int argc, char* argv[])
{
    bool show_usage = false;
    std::string metadata;
    std::string intrinsics;
    std::string image;
    std::string corners;
    std::string output;

    // Sanity checks on commandline
    if (argc != 6) {
        std::cerr << "[extrinsics_configurator] Error: too few arguments: expected 6, got " << argc << "." << std::endl;
        show_usage = true;
    } else {
        metadata = argv[1];
        intrinsics = argv[2];
        image = argv[3];
        corners = argv[4];
        output = argv[5];
        if (metadata.length() < 4 || metadata.substr(metadata.length() - 4)!=".xml") {
            std::cerr << "[hand_calibration] Error: checkerboard metadata should end with \".xml\": " << metadata << std::endl;
            show_usage = true;
        }
        if (image.length() < 4 || image.substr(image.length() - 4)!=".png") {
            std::cerr << "[extrinsics_configurator] Error: background should end with \".png\": " << image << std::endl;
            show_usage = true;
        }
        if (intrinsics.length() < 4 || intrinsics.substr(intrinsics.length() - 4)!=".xml") {
            std::cerr << "[extrinsics_configurator] Error: intrinsics should end with \".xml\": " << intrinsics << std::endl;
            show_usage = true;
        }
        if (corners.length() < 4 || corners.substr(corners.length() - 4)!=".xml") {
            std::cerr << "[extrinsics_configurator] Error: corners file should end with \".xml\": " << corners << std::endl;
            show_usage = true;
        }
        if (output.length() < 4 || output.substr(output.length() - 4)!=".xml") {
            std::cerr << "[extrinsics_configurator] Error: output should end with \".xml\": " << output << std::endl;
            show_usage = true;
        }
    }
    if (show_usage) {
        std::cerr << "Usage: " << argv[0] << " CHECKERBOARD_METADATA.xml INTRINSICS.xml BACKGROUND.png CORNERS.xml config.xml" << std::endl;
        return EXIT_FAILURE;
    }

    cv::Size board_size;
    int side_len = 0;
    // Read the checkerboard properties (XML)
    cv::FileStorage fs;
    fs.open(metadata, cv::FileStorage::READ);
    if (fs.isOpened())
    {
        fs["CheckerBoardWidth"] >> board_size.width;
        fs["CheckerBoardHeight"] >> board_size.height;
        fs["CheckerBoardSquareSize"] >> side_len;
        fs.release();
    }
    else
    {
        std::cerr << "[extrinsics_configurator] Error: Unable to read checkerboard properties: " << output << std::endl;
        return EXIT_FAILURE;
    }

    // Read the board corners
    std::vector<cv::Point> board_corners;
    fs.open(corners, cv::FileStorage::READ);
    if (fs.isOpened())
    {
        int corners_amount;
        fs["CornersAmount"] >> corners_amount;

        for (int b = 0; b < corners_amount; ++b)
        {
            std::string corner_id = "Corner_" + std::to_string(b);

            std::vector<int> corner;
            fs[corner_id] >> corner;
            assert(corner.size() == 2);
            board_corners.emplace_back(corner[0], corner[1]);
        }

        if (board_corners.size() != board_size.area()) {
            std::cerr << "[extrinsics_configurator] Error: not enough border corners." << std::endl;
            return EXIT_FAILURE;
        }

        fs.release();
    }
    else
    {
        std::cerr << "[extrinsics_configurator] Error: Unable to read corners: " << output << std::endl;
        return EXIT_FAILURE;
    }

    cv::Mat camera_matrix, distortion_coeffs;
    fs.open(intrinsics, cv::FileStorage::READ);
    if (fs.isOpened())
    {
        cv::Mat camera_matrix_f, distortion_coeffs_f;
        fs["CameraMatrix"] >> camera_matrix_f;
        fs["DistortionCoeffs"] >> distortion_coeffs_f;

        camera_matrix_f.convertTo(camera_matrix, CV_32F);
        distortion_coeffs_f.convertTo(distortion_coeffs, CV_32F);
        fs.release();
    }
    else
    {
        std::cerr << "[extrinsics_configurator] Unable to read camera intrinsics from: " << intrinsics << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<cv::Point3f> object_points;
    std::vector<cv::Point2f> image_points;

    // save the object points and image points
    for (int s = 0; s < board_size.area(); ++s)
    {
        auto div_mod = std::div(s, board_size.width);
        object_points.emplace_back(div_mod.quot * side_len, div_mod.rem * side_len, 0);
        image_points.push_back(board_corners[s]);
    }

    cv::Mat rotation_values_d, translation_values_d;
    cv::solvePnP(object_points, image_points, camera_matrix, distortion_coeffs, rotation_values_d, translation_values_d);

    cv::Mat rotation_values, translation_values;
    rotation_values_d.convertTo(rotation_values, CV_32F);
    translation_values_d.convertTo(translation_values, CV_32F);

    fs.open(output, cv::FileStorage::WRITE);
    if (fs.isOpened())
    {
        fs << "CameraMatrix" << camera_matrix;
        fs << "DistortionCoeffs" << distortion_coeffs;
        fs << "RotationValues" << rotation_values;
        fs << "TranslationValues" << translation_values;
        fs.release();
    }
    else
    {
        std::cerr << "[extrinsics_configurator] Error: Unable to write camera extrinsics to: " << output << std::endl;
        return EXIT_FAILURE;
    }

    auto canvas = cv::imread(image);
    if (canvas.empty())
    {
        std::cerr << "No data found in " << image << std::endl;
        return EXIT_FAILURE;
    }

    auto x_len = float(side_len * (board_size.height - 1));
    auto y_len = float(side_len * (board_size.width - 1));
    auto z_len = float(side_len * 3);
    cv::Point o = projectOnView(cv::Point3f(0, 0, 0), rotation_values, translation_values, camera_matrix, distortion_coeffs);
    cv::Point x = projectOnView(cv::Point3f(x_len, 0, 0), rotation_values, translation_values, camera_matrix, distortion_coeffs);
    cv::Point y = projectOnView(cv::Point3f(0, y_len, 0), rotation_values, translation_values, camera_matrix, distortion_coeffs);
    cv::Point z = projectOnView(cv::Point3f(0, 0, z_len), rotation_values, translation_values, camera_matrix, distortion_coeffs);

    line(canvas, o, x, cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
    line(canvas, o, y, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    line(canvas, o, z, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    circle(canvas, o, 3, cv::Scalar(0, 255, 255), -1, cv::LINE_AA);

    // Show the origin on the checkerboard
    cv::namedWindow("Origin", cv::WINDOW_KEEPRATIO);
    cv::imshow("Origin", canvas);
    cv::waitKey(1000);


    return EXIT_SUCCESS;
}

