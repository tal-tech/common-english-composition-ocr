#pragma once
//#include "include\OpencvInclude.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "GetLine.h"
#include "json.h"

using namespace std;
using namespace cv;
struct DISK_INFO {
	float vp;
	float lp;
	float rp;
	float r;
	vector<pair<Point2f, float>> disk;
	bool corr;
	DISK_INFO(float vp_, float lp_, float rp_, float r_, vector<pair<Point2f, float>> disk_, bool corr_)
	{
		vp = vp_;
		lp = lp_;
		rp = rp_;
		disk.assign(disk_.begin(), disk_.end());
		corr = corr_;
		r = r_;
	}
};
class Detector
{
public:
	Detector(const float tr_thresh_,const float tcl_thresh_,const float post_process_expand_);
	~Detector();
	Point2f find_innerpoint(const vector<Point>& cont);
	Point2f centerlize(const Point2f& shift,const int H,const int W,const float& tan_cos,const float& tan_sin, const std::vector<Point>& tcl_contour,const int& stride=1);
	vector<pair<Point2f, float>> mask_to_tcl(const Mat& pred_sin, const Mat& pred_cos,const Mat& pred_radii,const std::vector<Point>& tcl_contour, const Point2f& init_xy,const int& direct=1);
	vector<vector<pair<Point2f, float>>> build_tcl(const  Mat& tcl_pred,const Mat& sin_pred,const Mat& cos_pred,const Mat& radii_pred);
	Json::Value detector(const Mat& ori_image,  const vector<Mat>& outputs, vector<Mat>& batch_reg_images);
	vector<pair<vector<Point>, vector<pair<Point2f, float>>>> postprocessing(const cv::Size& size, const vector<vector<pair<Point2f, float>>>& disk_list,const Mat& tr_pred_mask);
	vector<DISK_INFO> area_filter(const cv::Size& size, const vector<pair<vector<Point>, vector<pair<Point2f, float>>>>& all_conts);
	vector<Point2f> make_diffs(const vector<DISK_INFO>& diskss);
	
	float tr_thresh;
	float tcl_thresh;
	float post_process_expand;
};
