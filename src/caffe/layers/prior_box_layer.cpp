#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/prior_box_layer.hpp"

// 参数参考这里
/*
layer {
  name: "conv4_3_norm_mbox_priorbox"
  type: "PriorBox"
  bottom: "conv4_3_norm"
  bottom: "data"
  top: "conv4_3_norm_mbox_priorbox"
  prior_box_param {
    min_size: 30.0
    max_size: 60.0
    aspect_ratio: 2
    flip: true
    clip: false
    variance: 0.1
    variance: 0.1
    variance: 0.2
    variance: 0.2
    step: 8
    offset: 0.5
  }
}
*/

namespace caffe {

template <typename Dtype>
void PriorBoxLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const PriorBoxParameter& prior_box_param =
      this->layer_param_.prior_box_param();
  CHECK_GT(prior_box_param.min_size_size(), 0) << "must provide min_size.";
  // 取min_size数据,这里是30.0
  for (int i = 0; i < prior_box_param.min_size_size(); ++i) {
    min_sizes_.push_back(prior_box_param.min_size(i));
    CHECK_GT(min_sizes_.back(), 0) << "min_size must be positive.";
  }
  aspect_ratios_.clear();
  aspect_ratios_.push_back(1.);
  //先把 1.0 压进 aspect_ratios, 由 min_size * min_size 确定的正方形 prior box正好两个了，最大和最小
  
  flip_ = prior_box_param.flip();
  
  //将 prior_box_param.aspect_ratio 不重复的压入 aspect_ratios. 若flip = true,把其倒数也压进去
  // 先猜测 aspect_ratios_ 是指{1, 2, 3, 1/2, 1/3} + 1 个数？？？ 
  for (int i = 0; i < prior_box_param.aspect_ratio_size(); ++i) {
    float ar = prior_box_param.aspect_ratio(i); // 上边只有一个 aspect_ratio: 2, 是不是生成了{1,2,1/2} 吧
    bool already_exist = false;
    for (int j = 0; j < aspect_ratios_.size(); ++j) {
      if (fabs(ar - aspect_ratios_[j]) < 1e-6) {//浮点数判断是否相等
        already_exist = true;
        break;
      }
    }
    if (!already_exist) {
      aspect_ratios_.push_back(ar);
      if (flip_) { //是否取倒数
        aspect_ratios_.push_back(1./ar);
      }
    }
  } //或者说，1这个比例是一定有的，其他的看配置

  // prior box 数量  注意：还需加上max_size 数量才是最终数量
  num_priors_ = aspect_ratios_.size() * min_sizes_.size();// num_priors_=3 ?????

  if (prior_box_param.max_size_size() > 0) { 
    CHECK_EQ(prior_box_param.min_size_size(), prior_box_param.max_size_size()); //max_size 个数和 min 一样
    for (int i = 0; i < prior_box_param.max_size_size(); ++i) {
      max_sizes_.push_back(prior_box_param.max_size(i));
      CHECK_GT(max_sizes_[i], min_sizes_[i])
          << "max_size must be greater than min_size.";
      num_priors_ += 1; // hahahha 这里因为有一个是 $s^'_k=\sqrt{s_k*s_{k+1}}$ 如果不存在max_size 的话这样不必加1
      // 只有一个min和max_size,故num_priors_=4, 此处的prior box是边长为sqrt(min*max)的正方形
    }
  }

  clip_ = prior_box_param.clip();// ??

  // variance 与后期真实框计算有关，要么给 1 个值，要么给 4 个值
  if (prior_box_param.variance_size() > 1) {
    // Must and only provide 4 variance.
    CHECK_EQ(prior_box_param.variance_size(), 4);
    for (int i = 0; i < prior_box_param.variance_size(); ++i) {
      CHECK_GT(prior_box_param.variance(i), 0);
      variance_.push_back(prior_box_param.variance(i));
    }
  } else if (prior_box_param.variance_size() == 1) {
    CHECK_GT(prior_box_param.variance(0), 0);
    variance_.push_back(prior_box_param.variance(0));
  } else {
    // Set default to 0.1.
    variance_.push_back(0.1);
  }

  // prototxt中一般未给定img_h,img_w和img_size,所以img_h,img_w = 0
  if (prior_box_param.has_img_h() || prior_box_param.has_img_w()) {
    CHECK(!prior_box_param.has_img_size())
        << "Either img_size or img_h/img_w should be specified; not both.";
    img_h_ = prior_box_param.img_h();
    CHECK_GT(img_h_, 0) << "img_h should be larger than 0.";
    img_w_ = prior_box_param.img_w();
    CHECK_GT(img_w_, 0) << "img_w should be larger than 0.";
  } else if (prior_box_param.has_img_size()) {
    const int img_size = prior_box_param.img_size();
    CHECK_GT(img_size, 0) << "img_size should be larger than 0.";
    img_h_ = img_size;
    img_w_ = img_size;
  } else {
    img_h_ = 0;
    img_w_ = 0;
  }

  // step 设定
  if (prior_box_param.has_step_h() || prior_box_param.has_step_w()) {
    CHECK(!prior_box_param.has_step())
        << "Either step or step_h/step_w should be specified; not both.";
    step_h_ = prior_box_param.step_h();
    CHECK_GT(step_h_, 0.) << "step_h should be larger than 0.";
    step_w_ = prior_box_param.step_w();
    CHECK_GT(step_w_, 0.) << "step_w should be larger than 0.";
  } else if (prior_box_param.has_step()) {
    const float step = prior_box_param.step();
    CHECK_GT(step, 0) << "step should be larger than 0.";
    step_h_ = step;
    step_w_ = step;
  } else {
    step_h_ = 0;
    step_w_ = 0;
  }

  offset_ = prior_box_param.offset();
}

//该层输出大小为 [1,2，layer_width * layer_height * num_priors_ * 4]
// c的第一维，存放每个框的四个点
// c的第二维，存放variance(每个框都一样)
template <typename Dtype>
void PriorBoxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // 取feature map大小
  const int layer_width = bottom[0]->width();
  const int layer_height = bottom[0]->height();
  vector<int> top_shape(3, 1);
  // Since all images in a batch has same height and width, we only need to
  // generate one set of priors which can be shared across all images.
  top_shape[0] = 1; // /同一层的priors,feature mp_size,img_size,aspect_ratios都一样,和batch无关,所以为1
  // 2 channels. First channel stores the mean of each prior coordinate.
  // Second channel stores the variance of each prior coordinate.
  top_shape[1] = 2;
  //对于1个prior,不管是prior coordinate还是variance都是4,fc7层有19*19*6个prioi
  top_shape[2] = layer_width * layer_height * num_priors_ * 4;
  CHECK_GT(top_shape[2], 0);
  top[0]->Reshape(top_shape);  // blob的reshape方法
}



template <typename Dtype>
void PriorBoxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int layer_width = bottom[0]->width(); // feature map 的宽
  const int layer_height = bottom[0]->height(); // feature map 的高
  int img_width, img_height;
  if (img_h_ == 0 || img_w_ == 0) {
    img_width = bottom[1]->width(); // 原图输入图像的宽 bottom[1] 为 data 输入，看 prototxt 有
    img_height = bottom[1]->height(); // 原图输入图像的高
  } else {
    img_width = img_w_;
    img_height = img_h_;
  }
  float step_w, step_h;
  if (step_w_ == 0 || step_h_ == 0) {
    // TODO 这里怎么算的？
    // 仔细看下 prototxt，对于 "conv9_2_mbox_priorbox" type: "PriorBox" 这种层
   // 有两个输入 bottom: "conv9_2" bottom: "data"，这样就好理解 step_w，step_h了

    step_w = static_cast<float>(img_width) / layer_width;// 缩放比例
    step_h = static_cast<float>(img_height) / layer_height;
  } else {
    step_w = step_w_;
    step_h = step_h_;
  }
  Dtype* top_data = top[0]->mutable_cpu_data();
  // 最后一维输出大小
  int dim = layer_height * layer_width * num_priors_ * 4;
  int idx = 0;
  for (int h = 0; h < layer_height; ++h) { // 在 fp 上遍历
    for (int w = 0; w < layer_width; ++w) {
      // 取feature map 每个点为中心点，进行处理
      // offset默认值是0.5，可理解为一个小的偏移量
      // 将中心点映射回了原图
      float center_x = (w + offset_) * step_w; // offset_ 作用是啥？
      //feature map上的点对应于原图上的位置,offset_=0.5做到了四舍五入
      float center_y = (h + offset_) * step_h; //将中心点映射回了原图
      float box_width, box_height;

      // 对 ar = 1. 的情况进行处理
      for (int s = 0; s < min_sizes_.size(); ++s) {
        int min_size_ = min_sizes_[s];
        // first prior: aspect_ratio = 1, size = min_size
        box_width = box_height = min_size_;
        // min_size确定的正方形框，大小进行了归一
        // xmin
        top_data[idx++] = (center_x - box_width / 2.) / img_width;
        // ymin
        top_data[idx++] = (center_y - box_height / 2.) / img_height;
        // 将原图上的 prior_box, 映射回特定层的 feature map，找到 feature map 左上角坐标

        // xmax
        top_data[idx++] = (center_x + box_width / 2.) / img_width;
        // ymax
        top_data[idx++] = (center_y + box_height / 2.) / img_height;
        // 将原图上的 prior_box, 映射回特定层的 feature map，找到 feature map 右下角坐标

        if (max_sizes_.size() > 0) {
        //设置了max_size的话,再生成一个sqrt(min*max)为边长的prio box
          CHECK_EQ(min_sizes_.size(), max_sizes_.size());
          int max_size_ = max_sizes_[s];
          // second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
          box_width = box_height = sqrt(min_size_ * max_size_);
          // max_size确定的正方形框
          // xmin
          top_data[idx++] = (center_x - box_width / 2.) / img_width;
          // ymin
          top_data[idx++] = (center_y - box_height / 2.) / img_height;
          // xmax
          top_data[idx++] = (center_x + box_width / 2.) / img_width;
          // ymax
          top_data[idx++] = (center_y + box_height / 2.) / img_height;
        }

        // rest of priors
        for (int r = 0; r < aspect_ratios_.size(); ++r) {
          float ar = aspect_ratios_[r];
          if (fabs(ar - 1.) < 1e-6) { // 1 的情况已经处理过了，所以不用处理了，跳过就行
            continue;
          }
          // 根据定义，由 aspect_ratio 和 min_size 共同确定的矩形框
          box_width = min_size_ * sqrt(ar);
          box_height = min_size_ / sqrt(ar);
          // xmin
          top_data[idx++] = (center_x - box_width / 2.) / img_width;
          // ymin
          top_data[idx++] = (center_y - box_height / 2.) / img_height;
          // xmax
          top_data[idx++] = (center_x + box_width / 2.) / img_width;
          // ymax
          top_data[idx++] = (center_y + box_height / 2.) / img_height;
        }
      }
    }
  }  //top_data依次存储每个prior box坐标信息,因为/300了,范围归一化到[0-1.0]

  //clip 默认false,true 的话让prior box的坐标越左界的置0,越右界置1
  // clip the prior's coordidate such that it is within [0, 1]
  if (clip_) {
    for (int d = 0; d < dim; ++d) {
      top_data[d] = std::min<Dtype>(std::max<Dtype>(top_data[d], 0.), 1.);
    }
  } //至此,top_data第一个channel的dim个坐标信息已写好,接下来写第二个channel的dim个variance_信息


  // 前面提到过，输出 c 维大小是 2，第一部分存放预选框数据，第二部分存放variance
  // set the variance.
  top_data += top[0]->offset(0, 1);// 通过偏移拿到第二部分的地址
  if (variance_.size() == 1) {
    caffe_set<Dtype>(dim, Dtype(variance_[0]), top_data);
  } else {
    int count = 0;
    for (int h = 0; h < layer_height; ++h) {
      for (int w = 0; w < layer_width; ++w) {
        for (int i = 0; i < num_priors_; ++i) {
          for (int j = 0; j < 4; ++j) {
            top_data[count] = variance_[j];
            ++count;
          }
        }
      }
    }
  }
}

INSTANTIATE_CLASS(PriorBoxLayer);
REGISTER_LAYER_CLASS(PriorBox);

}  // namespace caffe

