#pragma once

#include <opencv2/core.hpp> // CV_32FC3 

#include <opencv2/highgui.hpp> // imread
#include <opencv2/imgproc.hpp>

#include <dlib/image_processing.h>

#include <string>
#include <vector>
#include <functional>

class FaceMorph {

public:

  FaceMorph() {}

  int init(int gender, int model);

  //TODO: replace with vector of points
  cv::Mat morph_image(const cv::Mat& image, dlib::full_object_detection& shape);

private:

  int log_func(std::function<int ()> func, std::string desc);

  int load_model_points();

  void get_morph_mask();

  void get_poly_points();

  int get_morph_triangle();

  cv::Point2f get_point(int part_index);

  void get_landmark_points();


  void morph_face();

  void morph_triangle(const std::vector<cv::Point2f>& image_tri, 
                      const std::vector<cv::Point2f>& model_tri,
                      const std::vector<cv::Point2f>& morph_tri);
 
  void apply_affine_transform(cv::Mat& warpImage, 
                              cv::Mat& src, 
                              std::vector<cv::Point2f>& srcTri, 
                              std::vector<cv::Point2f>& dstTri);

  void specifiyHistogram(const cv::Mat& source_image, cv::Mat& target_image, cv::Mat& mask);

  void paste_face_on_image();


  cv::Mat image_;

  dlib::full_object_detection shape_;

  std::vector<cv::Point2f> points_; 


  cv::Mat model_image_;

  std::vector<cv::Point2f> model_points_; 


  std::vector<cv::Point2f> morph_points_;

  std::vector<std::vector<int>> triangle_;

  std::vector<cv::Point> points_poly_; 

  cv::Mat mask_;

  float alpha_;

  cv::Mat morph_face_;

  cv::Mat morph_image_;

  uint8_t LUT[3][256];
  int source_hist_int[3][256];
  int target_hist_int[3][256];
  float source_histogram[3][256];
  float target_histogram[3][256];

  std::string MODEL_DIR;

  std::string MODEL_IMAGE_FILE_PATH;
  std::string MODEL_POINTS_FILE_PATH;
  std::string TRIANGLE_FILE_PATH;

};
