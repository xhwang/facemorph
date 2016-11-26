
#include <fstream>
#include <iostream>
#include <sys/time.h>

#include <glog/logging.h>

#include "FaceMorph.h"

int FaceMorph::log_func(std::function<int ()> func, std::string desc) {

  long mtime, seconds, useconds;    
  struct timeval start, end;

  gettimeofday(&start, NULL);
  int rel = func();
  gettimeofday(&end, NULL);

  seconds  = end.tv_sec  - start.tv_sec;
  useconds = end.tv_usec - start.tv_usec;
  mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;

  LOG(INFO) << desc << " elapse " << mtime << " ms";

  return rel;
}

int FaceMorph::init(int gender, int model) {

  try {

    if(1 == gender) {
      MODEL_DIR =  "./models/M_" + std::to_string(model) + "/";
    }
    else {
      MODEL_DIR =  "./model/F_" + std::to_string(model) + "/";
    }

    LOG(INFO) << "init() model dir " << MODEL_DIR;

    MODEL_IMAGE_FILE_PATH = MODEL_DIR + "model.jpg";
    MODEL_POINTS_FILE_PATH = MODEL_DIR + "model.jpg.txt";
    TRIANGLE_FILE_PATH = MODEL_DIR + "model.tri.txt";

    //TODO: file exist check
    model_image_ = cv::imread(MODEL_IMAGE_FILE_PATH);
    load_model_points();
    get_morph_triangle();

    morph_face_ = cv::Mat::zeros(model_image_.size(), model_image_.type());
    mask_ = cv::Mat::zeros(model_image_.size(), CV_8UC1);

    alpha_ = 0.5;

  } catch (std::exception& e) {
    LOG(ERROR) << "FaceMorph init()" 
      << " gender:" << gender 
      << " model:" << model
      << e.what();
    return -1;
  }

  return 0;
}


int FaceMorph::load_model_points() {

  std::ifstream ifs(MODEL_POINTS_FILE_PATH);
  float x, y;
  while(ifs >> x >> y) {
    model_points_.push_back(cv::Point2f(x, y));
  }

  return 0;
}

int FaceMorph::get_morph_triangle() {

  std::ifstream ifs(TRIANGLE_FILE_PATH);
  int x, y, z;
  while(ifs >> x >> y >> z) {
    std::vector<int> tri = {x, y, z};
    triangle_.push_back(tri);
  }

  return 0;
}


cv::Mat FaceMorph::morph_image(const cv::Mat& image, dlib::full_object_detection& shape) {

#ifdef DEBUG
  int rel = 0;
#endif

  if(image.type() != CV_8UC3) {
    return cv::Mat::zeros(5,5, CV_8UC1);
  }

  image_ = image;

  shape_ = shape;

  get_landmark_points();

  // for now, morph points equals to model's
  // thus, morph mask can be loaded in init()
  morph_points_ = model_points_;

  // load morph mask
  get_morph_mask();

#ifdef DEBUG
  //cv::imshow("mask", mask_);
  //cv::waitKey(0);
#endif

#ifdef DEBUG
  rel = log_func([this]() -> int {morph_face();}, "morph face");
  //cv::imshow("morphed face", morph_face_);
  //cv::waitKey(0);
#else
  morph_face();
#endif

#ifdef DEBUG
  rel = log_func([this]() -> int {specifiyHistogram(model_image_, morph_face_, mask_);}, "color balance");
  //cv::imshow("color balance", morph_face_);
  //cv::waitKey(0);
#else
  specifiyHistogram(model_image_, morph_face_, mask_);
#endif

  morph_image_ = model_image_.clone();
#ifdef DEBUG
  rel = log_func([this]() -> int {paste_face_on_image();}, "morph image");
  //cv::imshow("image", morph_image_);
  //cv::waitKey(0);
#else
  paste_face_on_image();
#endif

  return morph_image_;
}

cv::Point2f FaceMorph::get_point(int part_index) {
  const auto& p = shape_.part(part_index);
  return cv::Point2f(p.x(), p.y());
}

void FaceMorph::get_landmark_points() {
  for(int i = 0; i < shape_.num_parts(); i++) {
    points_.push_back(get_point(i));
  }
}


void FaceMorph::morph_face() {

  int x, y, z;
  
  for(std::vector<int> tri: triangle_) {

    x = tri[0];
    y = tri[1];
    z = tri[2];

    std::vector<cv::Point2f> image_tri, model_tri, morph_tri;

    image_tri.push_back(points_[x]);
    image_tri.push_back(points_[y]);
    image_tri.push_back(points_[z]);
  
    model_tri.push_back(model_points_[x]);
    model_tri.push_back(model_points_[y]);
    model_tri.push_back(model_points_[z]);

    morph_tri.push_back(morph_points_[x]);
    morph_tri.push_back(morph_points_[y]);
    morph_tri.push_back(morph_points_[z]);
  
    morph_triangle(image_tri, model_tri, morph_tri);
  }

}


void FaceMorph::morph_triangle(const std::vector<cv::Point2f>& image_tri, 
                               const std::vector<cv::Point2f>& model_tri,
                               const std::vector<cv::Point2f>& morph_tri) {

  // Find bouding rectangle for each triangle
  cv::Rect image_rect = cv::boundingRect(image_tri);

  if(image_rect.y + image_rect.height >= image_.rows)
    image_rect.height = image_.rows - image_rect.y - 1;

  if(image_rect.x + image_rect.width >= image_.cols)
    image_rect.width = image_.cols - image_rect.x - 1;

  cv::Rect model_rect = cv::boundingRect(model_tri);
  cv::Rect morph_rect = cv::boundingRect(morph_tri);

  // Offset of triangle points by left top corner of the respective rectangle
  std::vector<cv::Point2f> offset_image, offset_model, offset_morph;
  std::vector<cv::Point> offset_poly;
  for(size_t i=0; i<morph_tri.size(); i++) {

    offset_image.push_back( cv::Point2f(image_tri[i].x - image_rect.x, 
                                        image_tri[i].y - image_rect.y) );
  
    offset_model.push_back( cv::Point2f(model_tri[i].x - model_rect.x, 
                                        model_tri[i].y - model_rect.y) );
  
    offset_morph.push_back( cv::Point2f(morph_tri[i].x - morph_rect.x, 
                                        morph_tri[i].y - morph_rect.y) );
  
    offset_poly.push_back( cv::Point(morph_tri[i].x - morph_rect.x, 
                                     morph_tri[i].y - morph_rect.y) );
  
  }

  // Get mask by fill triangle
  cv::Mat mask = cv::Mat::zeros(morph_rect.height, morph_rect.width, CV_8UC3);
  cv::fillConvexPoly(mask, offset_poly, cv::Scalar(1, 1, 1), 16, 0);

  // Apply affine transform to ROI rectangle patches
  cv::Mat image_patch, model_patch;

  image_(image_rect).copyTo(image_patch);
  model_image_(model_rect).copyTo(model_patch);

  cv::Mat warped_image_patch = cv::Mat::zeros(morph_rect.height, morph_rect.width, morph_image_.type());
  cv::Mat warped_model_patch = cv::Mat::zeros(morph_rect.height, morph_rect.width, morph_image_.type());

  apply_affine_transform(warped_image_patch, image_patch, offset_image, offset_morph);
  apply_affine_transform(warped_model_patch, model_patch, offset_model, offset_morph);

  // Alpha blend rectangle patches
  cv::Mat morph_patch = (1 - alpha_) * warped_image_patch + alpha_ * warped_model_patch;
  
  // Copy triangle region of rectangle back to output image
  cv::multiply(morph_patch, mask, morph_patch);
  cv::multiply(morph_face_(morph_rect), cv::Scalar(1, 1, 1) - mask, morph_face_(morph_rect));

  morph_face_(morph_rect) = morph_face_(morph_rect) + morph_patch;
}


// Apply affine transform calculated using srcTri and dstTri to src
void FaceMorph::apply_affine_transform(cv::Mat& warpImage, 
                                       cv::Mat& src, 
                                       std::vector<cv::Point2f>& srcTri, 
                                       std::vector<cv::Point2f>& dstTri) {
  // Given a pair of triangles, find the affine transform.
  cv::Mat warpMat = getAffineTransform( srcTri, dstTri );
                  
  // Apply the Affine Transform just found to the src image
  warpAffine( src, warpImage, warpMat, warpImage.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
}



void FaceMorph::get_morph_mask() {

  get_poly_points();

  cv::fillConvexPoly(mask_, points_poly_, cv::Scalar(255), 16, 0);

  cv::Size feather_amount;

  // TODO: decide feather amount
  // feather_amount.width = feather_amount.height = 
  // (int)cv::norm(morph_points_[0] - morph_points_[16]) / 8;

  feather_amount.height = 20;
  feather_amount.width = 20;

  cv::erode(mask_, mask_, cv::getStructuringElement(cv::MORPH_RECT, feather_amount), 
            cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, cv::Scalar(0));

  cv::blur(mask_, mask_, feather_amount, cv::Point(-1, -1), cv::BORDER_CONSTANT);

}


void FaceMorph::get_poly_points() {

  auto get_point_int = [&](cv::Point2f pt) -> cv::Point {
    return cv::Point(pt.x, pt.y);
  };

  // TODO: get better poly
  // order matters!!!
  points_poly_.push_back(get_point_int(morph_points_[0]));
  points_poly_.push_back(get_point_int(morph_points_[2]));
  points_poly_.push_back(get_point_int(morph_points_[4]));
  points_poly_.push_back(get_point_int(morph_points_[6]));
  points_poly_.push_back(get_point_int(morph_points_[8]));
  points_poly_.push_back(get_point_int(morph_points_[10]));
  points_poly_.push_back(get_point_int(morph_points_[12]));
  points_poly_.push_back(get_point_int(morph_points_[14]));
  points_poly_.push_back(get_point_int(morph_points_[16]));
  points_poly_.push_back(get_point_int(morph_points_[25]));
  points_poly_.push_back(get_point_int(morph_points_[24]));
  points_poly_.push_back(get_point_int(morph_points_[19]));
  points_poly_.push_back(get_point_int(morph_points_[18]));

  //cv::Point2f nose_length;
}

void FaceMorph::paste_face_on_image() {

  for (int i = 0; i < morph_image_.rows; i++) {
    auto image_pixel = morph_image_.row(i).data;
    auto faces_pixel = morph_face_.row(i).data;
    auto masks_pixel = mask_.row(i).data;

    // Debug: value can be larger than 255
    // int t = ((255 - *masks_pixel) * (*image_pixel) + (*masks_pixel) * (*faces_pixel));
    // std::cout << "value: " << t << std::endl;

    for (int j = 0; j < morph_image_.cols; j++) {
      if (*masks_pixel != 0) {

        *image_pixel = ((255 - *masks_pixel) * (*image_pixel) + (*masks_pixel) * (*faces_pixel)) >> 8; // divide by 256
        *(image_pixel + 1) = ((255 - *(masks_pixel + 1)) * (*(image_pixel + 1)) + (*(masks_pixel + 1)) * (*(faces_pixel + 1))) >> 8;
        *(image_pixel + 2) = ((255 - *(masks_pixel + 2)) * (*(image_pixel + 2)) + (*(masks_pixel + 2)) * (*(faces_pixel + 2))) >> 8;
      }

      image_pixel += 3;
      faces_pixel += 3;
      masks_pixel++;
    }
  }

}

void FaceMorph::specifiyHistogram(const cv::Mat& source_image, cv::Mat& target_image, cv::Mat& mask)
{

    std::memset(source_hist_int, 0, sizeof(int) * 3 * 256);
    std::memset(target_hist_int, 0, sizeof(int) * 3 * 256);

    for (size_t i = 0; i < mask.rows; i++)
    {
        auto current_mask_pixel = mask.row(i).data;
        auto current_source_pixel = source_image.row(i).data;
        auto current_target_pixel = target_image.row(i).data;

        for (size_t j = 0; j < mask.cols; j++)
        {
            if (*current_mask_pixel != 0) {
                source_hist_int[0][*current_source_pixel]++;
                source_hist_int[1][*(current_source_pixel + 1)]++;
                source_hist_int[2][*(current_source_pixel + 2)]++;

                target_hist_int[0][*current_target_pixel]++;
                target_hist_int[1][*(current_target_pixel + 1)]++;
                target_hist_int[2][*(current_target_pixel + 2)]++;
            }

            // Advance to next pixel
            current_source_pixel += 3; 
            current_target_pixel += 3; 
            current_mask_pixel++; 
        }
    }

    // Calc CDF
    for (size_t i = 1; i < 256; i++)
    {
        source_hist_int[0][i] += source_hist_int[0][i - 1];
        source_hist_int[1][i] += source_hist_int[1][i - 1];
        source_hist_int[2][i] += source_hist_int[2][i - 1];

        target_hist_int[0][i] += target_hist_int[0][i - 1];
        target_hist_int[1][i] += target_hist_int[1][i - 1];
        target_hist_int[2][i] += target_hist_int[2][i - 1];
    }

    // Normalize CDF
    for (size_t i = 0; i < 256; i++)
    {
        source_histogram[0][i] = (source_hist_int[0][i] ? (float) source_hist_int[0][i] / source_hist_int[0][255] : 0);
        source_histogram[1][i] = (source_hist_int[1][i] ? (float)source_hist_int[1][i] / source_hist_int[1][255] : 0);
        source_histogram[2][i] = (source_hist_int[2][i] ? (float)source_hist_int[2][i] / source_hist_int[2][255] : 0);

        target_histogram[0][i] = (target_hist_int[0][i] ? (float)target_hist_int[0][i] / target_hist_int[0][255] : 0);
        target_histogram[1][i] = (target_hist_int[1][i] ? (float)target_hist_int[1][i] / target_hist_int[1][255] : 0);
        target_histogram[2][i] = (target_hist_int[2][i] ? (float)target_hist_int[2][i] / target_hist_int[2][255] : 0);
    }

    // Create lookup table

    auto binary_search = [&](const float needle, const float haystack[]) -> uint8_t
    {
        uint8_t l = 0, r = 255, m;
        while (l < r)
        {
            m = (l + r) / 2;
            if (needle > haystack[m])
                l = m + 1;
            else
                r = m - 1;
        }
        // TODO check closest value
        return m;
    };

    for (size_t i = 0; i < 256; i++)
    {
        LUT[0][i] = binary_search(target_histogram[0][i], source_histogram[0]);
        LUT[1][i] = binary_search(target_histogram[1][i], source_histogram[1]);
        LUT[2][i] = binary_search(target_histogram[2][i], source_histogram[2]);
    }

    // repaint pixels
    for (size_t i = 0; i < mask.rows; i++)
    {
        auto current_mask_pixel = mask.row(i).data;
        auto current_target_pixel = target_image.row(i).data;
        for (size_t j = 0; j < mask.cols; j++)
        {
            if (*current_mask_pixel != 0)
            {
                *current_target_pixel = LUT[0][*current_target_pixel];
                *(current_target_pixel + 1) = LUT[1][*(current_target_pixel + 1)];
                *(current_target_pixel + 2) = LUT[2][*(current_target_pixel + 2)];
            }

            // Advance to next pixel
            current_target_pixel += 3;
            current_mask_pixel++;
        }
    }
}

