#include <iostream>
#include <iomanip>
#include <string>

#include <opencv2/opencv.hpp>

int calc_threshold_count(cv::Mat m, int t) {
    int count = 0;

    for (int i = 0; i < m.rows; i++) {
        int val = m.at<float>(i, 0);

        if (val > t) {
            count++;
        }
    }

    return count;
}

cv::Mat detect(cv::Mat frame, int count) {
    const int THRESHOLD = ((1920*1080)/12)/16;

    cv::medianBlur(frame, frame, 5);

    cv::cvtColor(frame, frame, cv::COLOR_BGR2HSV);

    cv::Mat hsv[3];
    cv::split(frame, hsv);

    #pragma omp parallel for 
    for (int i = 0; i < 2; i++) {
        cv::dilate(hsv[i], hsv[i], cv::Mat(), cv::Point(-1,-1), 3);
        cv::medianBlur(hsv[i], hsv[i], 7);
    }

    int h_histSize[] = {360};
    int s_histSize[] = {256};
    int v_histSize[] = {256};
    float h_rng[] = { 0, 360 };
    float s_rng[] = { 0, 256 };
    float v_rng[] = { 0, 256 };
    const float* hranges[] = { h_rng };
    const float* sranges[] = { s_rng };
    const float* vranges[] = { v_rng };

    cv::Mat hist;
    int channels[] = {0};
    cv::Mat result = cv::Mat::zeros(cv::Size(3, 1), CV_32S);

    cv::calcHist( &hsv[0], 1, channels, cv::Mat(), hist, 1, h_histSize, hranges );
    result.at<int>(0, 0) = calc_threshold_count(hist, THRESHOLD);

    cv::calcHist( &hsv[1], 1, channels, cv::Mat(), hist, 1, s_histSize, sranges );
    result.at<int>(0, 1) = calc_threshold_count(hist, THRESHOLD);

    cv::calcHist( &hsv[2], 1, channels, cv::Mat(), hist, 1, v_histSize, vranges );
    result.at<int>(0, 2) = calc_threshold_count(hist, THRESHOLD);

    return result;
}


static const std::string keys = "{ help h    |   | print help message }"
                                "{ p pattern |   | pattern image }"
                                "{ v video   |   | video for analyze }";

int main(int argc, char ** argv)
{
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Pattern detect");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    std::string pattern_filename = parser.get<std::string>("pattern");
    std::string video_filename = parser.get<std::string>("video");

    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }

    if (pattern_filename == "") {
        std::cout << "Please set pattern image" << std::endl;
        return 1;
    }

    if (video_filename == "") {
        std::cout << "Please set video for analyzing" << std::endl;
        return 1;
    }

    cv::Mat frame = cv::imread(pattern_filename);
    if (frame.empty()) {
        std::cout << "Could not open file: " << pattern_filename << std::endl;
        return 1;
    }

    cv::pyrDown(frame, frame);
    cv::pyrDown(frame, frame);

    cv::Mat f = detect(frame, 0);

    cv::VideoCapture cap(video_filename);

    int count = 0;

    int sceens = 0;
    int prev = 0;

    while (true) {
        count ++;
        cv::Mat frame; 
        cap >> frame;

        if (frame.empty()) {
            break;
        }

        cv::pyrDown(frame, frame);
        cv::pyrDown(frame, frame);
        
        cv::Mat cf = detect(frame, count);

        if ( (prev < f.at<int>(0, 0)) && (cf.at<int>(0, 0) >= f.at<int>(0, 0)) ) {
            std::cout << "Detected on " << count << " frame" << std::endl;
            sceens++;
        }
        prev = cf.at<int>(0, 0);
    }

    std::cout << "Detected sceens: " << sceens << std::endl;
}


