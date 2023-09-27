//#include "english_ocr.h"
#include "cls_num_rec.hpp"
#include "det_textsnake_eng.hpp"
#include "Detector.h"
#include "ImageFunc.h"
#include "Decode.h"

using namespace facethink;

#include <string>
#include <vector>
#include "opencv2/opencv.hpp"


#include <mutex>
std::mutex det_model_mutex;
std::mutex eng_det_mutex;
std::mutex eng_batch_mutex;
/////////////////////////////////////////////////////
static std::mutex         rotation_mutex;



std::string det_model_;
std::string det_config_;
std::string eng_model_;
std::string eng_config_;
std::string eng_zidian_;
std::string eng_decode_dict_;
std::string gen_model_;
std::string gen_config_;
std::string gen_zidian_;
std::string gen_decode_dict_;
std::unique_ptr<TextSnakeDet> p_det_model_ = nullptr;
std::unique_ptr<NumRecClassify> p_eng_model_ = nullptr;
std::unique_ptr<Decode> p_eng_decode_ = nullptr;
std::unique_ptr<NumRecClassify> p_gen_model_ = nullptr;
std::unique_ptr<Decode> p_gen_decode_ = nullptr;


bool Init() {
    
    std::string program_dir;
   
    //
    det_model_  = "./alg/model/det_model/textsnake_resnet_89_softmax_v1.2.onnx";
    det_config_ = "./alg/model/det_model/det_config.ini";
    //
    eng_model_       = "./alg/model/reg_eng/crnn_Rec_done_5_343_V1_2.pt";
    eng_config_      = "./alg/model/reg_eng/reg_config.ini";
    eng_zidian_      = "./alg/model/reg_eng/dict_eng.txt";
    eng_decode_dict_ = "./alg/model/reg_eng/word_en.txt";
    //
    gen_model_       = "./alg/model/reg_gen/crnn_Rec_swa_445w_9_20_use2zhongwenzuowen_new.pt";
    gen_config_      = "./alg/model/reg_gen/reg_config.ini";
    gen_zidian_      = "./alg/model/reg_gen/zidian_new_5883.txt";
    gen_decode_dict_ = "./alg/model/reg_gen/word_en.txt";

    // init algorithm model
    if (nullptr == p_det_model_) {
        std::vector<int> det_input_shape{3, 640, 640};
        p_det_model_.reset(TextSnakeDet::create(det_model_, det_config_, det_input_shape, 1));
    }
    //
    if (nullptr == p_eng_model_) {
        p_eng_model_.reset(NumRecClassify::create(eng_model_, eng_config_));
    }
    if (nullptr == p_eng_decode_) {
        p_eng_decode_.reset(new Decode(eng_zidian_, eng_decode_dict_));
    }
    //
    if (nullptr == p_gen_model_) {
        p_gen_model_.reset(NumRecClassify::create(gen_model_, gen_config_));
    }
    if (nullptr == p_gen_decode_) {
        p_gen_decode_.reset(new Decode(gen_zidian_, gen_decode_dict_));
    }
    return true;
}

void RotateImg(const cv::Mat &old_mat, cv::Mat &new_mat, const int rotate_angle) {
    if (270 == rotate_angle) {
        cv::transpose(old_mat, new_mat);
        cv::flip(new_mat, new_mat, 0);
    } else if (180 == rotate_angle) {
        cv::flip(old_mat, new_mat, 0);
        cv::flip(new_mat, new_mat, 1);
    } else if (90 == rotate_angle) {
        cv::transpose(old_mat, new_mat);
        cv::flip(new_mat, new_mat, 1);
    } else if (0 == rotate_angle) {
        new_mat = old_mat;
    }
}

std::string ProcessRotation(const std::string &img_base64) {
    std::string result, err_msg, body;
    Json::Value body_json;
    body_json["mode"]         = 0;
    body_json["image_base64"] = img_base64;
    body = Json::FastWriter().write(body_json);
    std::map<std::string, std::string> headers;
   

    return result;
}

int ProcessAlgDet(std::vector<cv::Mat> &vec_reg, NumRecClassify *p_model, Decode *p_decode,
        Json::Value &alg_res) {
    //
    int max_batch = 24;
    std::vector<std::vector<std::vector<float>>> reg_probs;
    for (int epoch = 0; epoch < ceil(vec_reg.size() / float(max_batch)); epoch++) {
        cv::Mat float_image;
        std::vector<std::vector<std::vector<float>>> batch_probs;
        std::vector<Mat> batch_reg(vec_reg.begin() + epoch * max_batch, vec_reg.begin() + std::min((1+epoch) * max_batch, int(vec_reg.size())));
        {
            std::unique_lock<std::mutex> lock(eng_det_mutex);
            int ret = p_model->detection(batch_reg, batch_probs);
            if (ret < 0)
                return -1;
        }
        //
        reg_probs.insert(reg_probs.end(), batch_probs.begin(), batch_probs.end());
    }
    //
    std::unique_lock<std::mutex> lock(eng_batch_mutex);
    p_decode->RunReg_batchimage_topk(reg_probs, alg_res);
    return 0;
}

int main(){
    Init();
    std::vector<cv::Mat>reg;
    Json::Value alg_result;
    cv::Mat cv= cv::imread("./7.jpg");
    reg.emplace_back(cv);
    
    int res = ProcessAlgDet(reg, p_eng_model_.get(), p_eng_decode_.get(), alg_result);
    std::cout<<alg_result<<std::endl;
    std::cout<<"--------0-0-00---"<<std::endl;
    return 0;
}
