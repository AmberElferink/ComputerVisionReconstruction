#include <cstdlib>
#include <iostream>
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

int main(int argc, char* argv[])
{
    bool show_usage = false;
    std::string input;
    std::string output;

    // Sanity checks on commandline
    if (argc != 4) {
        std::cerr << "[image_extractor] Error: too few arguments: expected 4, got " << argc << "." << std::endl;
        show_usage = true;
    } else {
        input = argv[1];
        output = argv[2];
    }
    if (input.length() < 4 || input.substr(input.length() - 4) != ".avi") {
        std::cerr << "[image_extractor] Error: input should end with \".avi\": " << input << std::endl;
        show_usage = true;
    }
    if (output.length() < 4 || output.substr(output.length() - 4) != ".png") {
        std::cerr << "[image_extractor] Error: output should end with \".png\": " << output << std::endl;
        show_usage = true;
    }
    auto frame_number = std::stol(argv[3], nullptr, 0);
    if (frame_number < 0 || frame_number == ULONG_MAX) {
        std::cerr << "[image_extractor] Error: frame number should be a positive number: " << argv[4] << std::endl;
        show_usage = true;
    }
    if (show_usage) {
        std::cerr << "Usage: " << argv[0] << " INPUT.avi OUTPUT.png FRAME_NUMBER" << std::endl;
        return EXIT_FAILURE;
    }

    // Read all frames in input video file
    cv::VideoCapture capture;
    capture.open(input);
    cv::Mat frame;
    for (uint32_t i = 0; i < frame_number; ++i) {
        capture >> frame;
        if (frame.empty()) {
            capture.release();
            std::cerr << "[image_extractor] Error: frame number larger than number of frames in input file. " << std::endl;
            return EXIT_FAILURE;
        }
    }

    // Write output
    if (cv::imwrite(output, frame, {cv::IMWRITE_PNG_COMPRESSION, 0}))
    {
        capture.release();
        return EXIT_SUCCESS;
    }
    else
    {
        capture.release();
        return EXIT_FAILURE;
    }
}

