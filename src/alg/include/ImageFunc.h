#pragma once
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;
class ImageFunc
{
public:
	ImageFunc();
	~ImageFunc();
	static vector<Mat> grid_sample(const Mat& srcImage, vector< vector<vector<Point2f>>> batch_grid,int Mode=0);
	static int preprocess(std::vector<cv::Mat> &imgs, cv::Mat &img_floats);
private:
	static cv::Mat blobFromImages(const std::vector<cv::Mat>& images,std::vector<double> means,std::vector<double> scales,bool swapRB);
	static cv::Mat blobFromImage(const cv::Mat& image_,std::vector<double> means,std::vector<double> scales,bool swapRB);
};

