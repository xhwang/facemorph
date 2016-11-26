
#pragma once

#include <dlib/opencv.h>
#include <dlib/image_processing.h>

namespace EchoService {

/**
 * Since we keep one instance of this in each class, 
 * there is no need of synchronization
 */
class ShapePredictor {
 public:
  ShapePredictor () {

    //std::cout<< "init shape predictor" 
    //         << std::endl;

    const std::string predictor_path = 
      "shape_predictor_68_face_landmarks.dat";

    dlib::deserialize(predictor_path) >> sp_;

  }
  ~ShapePredictor () {
  }

  dlib::full_object_detection predictShape(
    dlib::cv_image<dlib::bgr_pixel>& dlib_image,
    dlib::rectangle& face_rect) {

    return sp_(dlib_image, face_rect);

  }

 private:

  dlib::shape_predictor sp_;

};

}
