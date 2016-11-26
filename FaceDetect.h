#pragma once

#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp> // imread
#include <opencv2/imgproc.hpp>
#include "opencv2/objdetect/objdetect.hpp"

#include <vector>

class FaceDetect {

public:

  FaceDetect() {}

  int init();

  std::vector<dlib::rectangle> detect_face(const dlib::cv_image<dlib::bgr_pixel>& dlib_image);

  std::vector<cv::Rect> detect_face_opencv(const cv::Mat& opencv_image);

private:

  // dlib frontal face detector
  dlib::frontal_face_detector detector_;

  cv::CascadeClassifier face_cascade_;

};
