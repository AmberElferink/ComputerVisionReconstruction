#include <cstdlib>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#define MAIN_WINDOW "Checkerboard Marking"

struct calib_data_t {
  std::vector<cv::Point> board_corners;
  cv::Point mouse_position;
};

void on_mouse(int event, int x, int y, int flags, void* param)
{
    auto data = reinterpret_cast<calib_data_t*>(param);
    switch (event)
    {
    case cv::EVENT_LBUTTONDOWN:
        if (flags == (cv::EVENT_FLAG_LBUTTON + cv::EVENT_FLAG_CTRLKEY))
        {
            if (!data->board_corners.empty())
            {
                std::cout << "Removed corner " << data->board_corners.size() << "... (use Click to add)" << std::endl;
                data->board_corners.pop_back();
            }
        }
        else
        {
            data->board_corners.emplace_back(x, y);
            std::cout << "Added corner " << data->board_corners.size() << "... (use CTRL+Click to remove)" << std::endl;
        }
        break;
    case cv::EVENT_MOUSEMOVE:
        data->mouse_position.x = x;
        data->mouse_position.y = y;
        break;
    default:
        break;
    }
}

int main(int argc, char* argv[])
{
    bool show_usage = false;
    std::string metadata;
    std::string input;
    std::string output;

    // Sanity checks on commandline
    if (argc != 4) {
        std::cerr << "[hand_calibration] Error: too few arguments: expected 4, got " << argc << "." << std::endl;
        show_usage = true;
    } else {
        metadata = argv[1];
        input = argv[2];
        output = argv[3];
    }
    if (metadata.length() < 4 || metadata.substr(metadata.length() - 4) != ".xml") {
        std::cerr << "[hand_calibration] Error: checkerboard metadata should end with \".xml\": " << metadata << std::endl;
        show_usage = true;
    }
    if (input.length() < 4 || input.substr(input.length() - 4) != ".png") {
        std::cerr << "[hand_calibration] Error: input should end with \".png\": " << input << std::endl;
        show_usage = true;
    }
    if (output.length() < 4 || output.substr(output.length() - 4) != ".xml") {
        std::cerr << "[hand_calibration] Error: output should end with \".xml\": " << output << std::endl;
        show_usage = true;
    }
    if (show_usage) {
        std::cerr << "Usage: " << argv[0] << " CHECKERBOARD_METADATA.xml INPUT.png OUTPUT.xml" << std::endl;
        return EXIT_FAILURE;
    }

    int cb_width = 0, cb_height = 0;
    int cb_square_size = 0;
    // Read the checkerboard properties (XML)
    cv::FileStorage fs;
    fs.open(metadata, cv::FileStorage::READ);
    if (fs.isOpened())
    {
        fs["CheckerBoardWidth"] >> cb_width;
        fs["CheckerBoardHeight"] >> cb_height;
        fs["CheckerBoardSquareSize"] >> cb_square_size;
    }
    fs.release();

    const cv::Size board_size(cb_width, cb_height);


    std::cout << "Estimate camera extrinsics by hand..." << std::endl;
    calib_data_t data = {{}, {}};
    namedWindow(MAIN_WINDOW, cv::WINDOW_KEEPRATIO);
    cv::setMouseCallback(MAIN_WINDOW, on_mouse, &data);

    auto frame = cv::imread(input);
    if (frame.empty())
    {
        std::cerr << "No data found in " << input << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Now mark the " << board_size.area() << " interior corners of the checkerboard" << std::endl;
    while ((int) data.board_corners.size() < board_size.area())
    {
        auto canvas = frame.clone();
        if (!data.board_corners.empty())
        {
            for (size_t c = 0; c < data.board_corners.size(); c++)
            {
                cv::circle(canvas, data.board_corners[c], 4, cv::Scalar(255, 0, 255), 1, 8);
                if (c > 0)
                {
                    cv::line(canvas, data.board_corners[c], data.board_corners[c - 1], cv::Scalar(255, 0, 255), 1, 8);
                }
            }
            cv::Point2i vector = data.mouse_position - data.board_corners.back();
            cv::line(canvas, data.board_corners.back() -  10.0f * vector, data.board_corners.back() + 10.0f * vector, cv::Scalar(0, 200, 0), 1, 8);
        }

        int key = cv::waitKey(10);
        if (key == 'q' || key == 'Q')
        {
            return EXIT_FAILURE;
        }
        else if (key == 'c' || key == 'C')
        {
            data.board_corners.pop_back();
        }

        cv::imshow(MAIN_WINDOW, canvas);
    }

    assert((int)data.board_corners.size() == board_size.area());
    std::cout << "Marking finished!" << std::endl;
    cv::destroyAllWindows();

    fs.open(output, cv::FileStorage::WRITE);
    if (fs.isOpened())
    {
        fs << "CornersAmount" << (int) data.board_corners.size();
        for (size_t b = 0; b < data.board_corners.size(); ++b)
        {
            fs << ("Corner_" + std::to_string(b)) << data.board_corners[b];
        }
        fs.release();
    }

    return EXIT_SUCCESS;
}
