#pragma once
//#include "include\OpencvInclude.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
// #include "GetLine.h"
#include<vector>
#include "json.h"
using namespace std;
using namespace cv;
typedef struct tagPathDecode
{
	double fScore;
	std::vector<int> vecStatePath;
	std::vector<int> vecStatePos;
	int nCurState;
	std::string strDecodeResult;
} pathDecode;
class Decode
{
public:

	std::vector<std::string> state_map;
	std::vector<std::string> word_map;
	Decode(std::string zidian_file_path,std::string word_file_path);
	~Decode();
	void RunReg_batchimage_topk(std::vector<std::vector<std::vector<float>>> arraypro, Json::Value &obj);
private:
	bool mulDecode(std::vector<std::vector<float>> probAll, int N, float frameP, float expP, float diffP, int decodeFunc, std::vector<std::string> &regStrsUSort, std::vector<std::vector<int>> &regPositionsList, std::vector<float> &regWeightsUSort, const std::string label);
	std::vector<std::string> MulPathSelect(std::vector<std::vector<float>> &arr, int topFlag, int nTopK, std::vector<float> &pathScore, std::vector<std::vector<int>> &pathPos, float frameP, float expP, float diffP);
	int getScoreTopK(std::vector<pathDecode> &vecAllPathDecode, int nK, double fMinFramScore, std::vector<pathDecode> &vecPathDecodeTopK);
	// bool sortDecode(const pathDecode &sc1, const pathDecode &sc2);
	inline int my_max_element(const std::vector<float> &vec, float &max_value);
	int ldistance(const std::string source, const std::string target);
	void get_limited_topk(vector<vector<pair<float, int>>> arraypro, int topK, vector<vector<pair<string, float>>>& values);
	void search_coorect_word(vector<int> list_zb, vector<vector<pair<string, float>>> &values, string &new_str_top1, vector<int> &new_zb_list, vector<float> &new_weight_list, vector<vector<pair<string, float>>> &new_arr);
	vector<vector<Point>> get_char_box(vector<int> pos,vector<Point> pts);
	pair<vector<Point>,vector<int>> get_char_pos(vector<int> pos,vector<Point> pts);
	void normalReplace(string &str, const string &old_value, const string &new_value);
	int STATE_NUM = 71;
	string upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
	string lower = "abcdefghijklmnopqrstuvwxyz";
	string alphabet = upper + lower;
	bool b_fix_J = true;
};