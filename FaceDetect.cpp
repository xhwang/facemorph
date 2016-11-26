#include <string>
#include <glog/logging.h>

#include "FaceDetect.h"

int FaceDetect::init() {

  try {
    // detector_ = dlib::get_frontal_face_detector();

    std::string face_cascade_name = "haarcascade_frontalface_alt.xml";
    if( !face_cascade_.load( face_cascade_name ) ) { 
      LOG(ERROR) << "Error loading face cascade";
      return -1;
    }

  } catch (std::exception& e) {
    LOG(ERROR) << e.what();
    return -1;
  }

  return 0;
}

std::vector<dlib::rectangle> FaceDetect::detect_face(const dlib::cv_image<dlib::bgr_pixel>& dlib_image) {

  std::vector<dlib::rectangle> rects = detector_( dlib_image );

  return rects;
}

std::vector<cv::Rect> FaceDetect::detect_face_opencv(const cv::Mat& opencv_image) {

  std::vector<cv::Rect> faces;

  cv::Mat image_gray;
  cvtColor( opencv_image, image_gray, CV_BGR2GRAY );

  // Detect faces
  face_cascade_.detectMultiScale( image_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(30, 30) );

  return faces;

}
