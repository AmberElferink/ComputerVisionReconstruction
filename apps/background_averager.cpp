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
    if (argc != 3) {
        std::cerr << "[background_averager] Error: too few arguments: expected 3, got " << argc << "." << std::endl;
        show_usage = true;
    } else {
        input = argv[1];
        output = argv[2];
    }
    if (input.length() < 4 || input.substr(input.length() - 4) != ".avi") {
        std::cerr << "[background_averager] Error: input should end with \".avi\": " << input << std::endl;
        show_usage = true;
    }
    if (output.length() < 4 || output.substr(output.length() - 4) != ".png") {
        std::cerr << "[background_averager] Error: output should end with \".png\": " << output << std::endl;
        show_usage = true;
    }
    if (show_usage) {
        std::cerr << "Usage: " << argv[0] << " INPUT.avi OUTPUT.png" << std::endl;
        return EXIT_FAILURE;
    }

    // Read all frames in input video file
    cv::VideoCapture capture;
    capture.open(input);
    cv::Mat frame;
    capture >> frame;
    cv::Mat sum = cv::Mat::zeros(frame.rows, frame.cols, CV_64FC3);
    uint32_t num_frames = 0;
    while (!frame.empty()) {
        cv::accumulate(frame, sum);
        capture >> frame;
        ++num_frames;
    }
    capture.release();

    // Fail if no frames were gotten
    if (num_frames == 0) {
        std::cerr << "No frames found in " << input << std::endl;
        return EXIT_FAILURE;
    }

    // Average the sum
    cv::Mat average = sum / num_frames;

    // Write output
    if (cv::imwrite(output, average, {cv::IMWRITE_PNG_COMPRESSION, 0}))
    {
        return EXIT_SUCCESS;
    }
    else
    {
        return EXIT_FAILURE;
    }

}
