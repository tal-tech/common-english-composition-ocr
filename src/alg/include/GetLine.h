#pragma once
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;
//#include "Detector.h"
class GetLine
{
public:
	GetLine();
	~GetLine();
	float l2_dist(cv::Point2f pt1, cv::Point2f pt2);
	pair<vector<Point2f>, vector<vector<Point2f>>> get_poly(const vector<pair<Point2f, float>>& disks);
	bool polynomial_curve_fit(const std::vector<cv::Point2f>& key_point, int n, cv::Mat& A);
	vector<float> line_leng(const vector<Point2f>& line);
	vector<Point2f> linespace(const Point2f& start, const  Point2f& end, int num, bool open);
	pair<vector<Point2f>, vector<Point2f>> make_line(vector<Point2f> line1, vector<Point2f> line2, vector<float> ls, float gaps = 1024);
	vector<Point2f> moving_average(vector<Point2f> line, int n = 3);
	vector<Point2f> moving_ave(vector<Point2f> line, int sz = 64);
	vector<vector<Point2f>> make_grid(const vector<Point2f>& line1,const vector<Point2f>& line2,int h=64);
};

