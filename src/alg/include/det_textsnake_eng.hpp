///////////////////////////////////////////////////////////////////////////////////////
///  Copyright (C) 2017, TAL AILab Corporation, all rights reserved.
///
///  @file: cls_num_rec.hpp
///  @brief 数字公式识别
///  @details 最初版本
//
//
///  @version 1.0.0.0
///  @author Qiao Yu
///  @date 2020-05-07
///
///  @see 使用参考：demo.cpp
///
///////////////////////////////////////////////////////////////////////////////////////
#ifndef __FACETHINK_API_TEXTSNAKE_CONFIG_HPP__
#define __FACETHINK_API_TEXTSNAKE_CONFIG_HPP__
#include <math.h>

#include <opencv2/opencv.hpp>
#include <string>
#ifdef WIN32
#ifdef DLL_EXPORTS
#define EXPORT_CLASS __declspec(dllexport)
#define EXPORT_API extern "C" __declspec(dllexport)
#define EXPORT_CLASS_API
#else
#define EXPORT_CLASS __declspec(dllimport)
#define EXPORT_API extern "C" __declspec(dllimport)
#endif
#else
#define EXPORT_CLASS
#define EXPORT_API extern "C" __attribute__((visibility("default")))
#define EXPORT_CLASS_API __attribute__((visibility("default")))
#endif
namespace facethink {
class EXPORT_CLASS TextSnakeDet {
 public:
  EXPORT_CLASS_API virtual ~TextSnakeDet(void);

  /**
		 * @brief SDK初始化函数，必须先于其他函数之前调用
		 * @param [in] model_file 模型文件路径(仅限于uff, trt, onnx格式)
		 * @param [in] config_file SDK对应的ini配置文件路径，详见config.ini
		 * @param [in] input_shapes 模型输入的尺寸
		 * @param [in] max_batch 模型的最大批次，应该和模型的batch相同
		*/
  EXPORT_CLASS_API static TextSnakeDet* create(const std::string& model_file, const std::string& config_file,
                                                 const std::vector<int>& input_shapes, int max_batch);

  /** 
	 * @brief 模型推理接口
	 * @param [in] img 输入的图片，将cv::Mat数组合并为连续的cv::Mat, 数据排列方式为NCHW
	 * @param [in] batch_count 图片的batch数量
	 * @param [out] probs 模型推理的结果
	 * 
	*/
  EXPORT_CLASS_API virtual int doInference(const cv::Mat& img, int batch_count, std::vector<cv::Mat>& output) = 0;
  
 protected:
  EXPORT_CLASS_API explicit TextSnakeDet(void);
};
}  // namespace facethink
#endif
