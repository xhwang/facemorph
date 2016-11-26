/*
 *  Copyright (c) 2016, Facebook, Inc.
 *  All rights reserved.
 *
 *  This source code is licensed under the BSD-style license found in the
 *  LICENSE file in the root directory of this source tree. An additional grant
 *  of patent rights can be found in the PATENTS file in the same directory.
 *
 */
#include "EchoHandler.h"

#include <fstream>
#include <vector>
#include <ctime>
#include <sys/time.h>

#include <glog/logging.h>

#include <proxygen/httpserver/RequestHandler.h>
#include <proxygen/httpserver/ResponseBuilder.h>

#include "FaceDetect.h"
#include "FaceMorph.h"
#include "ShapePredictor.h"

using namespace proxygen;

namespace EchoService {

EchoHandler::EchoHandler(ShapePredictor* sp): sp_(sp) {
}

void EchoHandler::onRequest(std::unique_ptr<HTTPMessage> headers) noexcept {

  //std::cout<< "url " << headers.get()->getURL() << std::endl;
  //std::cout<< "path " << headers.get()->getPath() << std::endl;

  HTTPHeaders h = headers.get()->getHeaders();
  //std::string host = h.getSingleOrEmpty(HTTP_HEADER_HOST);
  //std::cout<< "host " << host << std::endl;

  std::string str_gender = h.rawGet("gender");
  std::string str_model = h.rawGet("model");

  gender = std::stoi(str_gender);

  if(!str_model.empty()) {
    model = std::stoi(str_gender);
  }
  else {
    model = 0;
  }

  LOG(INFO) << "gender:" << str_gender << " model:" << str_model;

}

void EchoHandler::onBody(std::unique_ptr<folly::IOBuf> body) noexcept {

  if (body_) {
    // prepend ?
    body_->prependChain(std::move(body));
  } else {
    body_ = std::move(body);
  }

}


void EchoHandler::getRequestBodyAsWhole() {

  // combine chunk
  uint64_t data_size = body_->computeChainDataLength();
  image_data_ = folly::IOBuf::create(data_size);

  for (auto buf: *body_)
  {
    // std::cout<< "size: " << buf.size() << std::endl;
    // write to tail
    memcpy(image_data_->writableTail(), buf.data(), buf.size());
    image_data_->append(buf.size());
  }

}

void EchoHandler::convertMatToResponse(std::vector<uchar> & buf) {

  size_t buf_size = buf.size();
  response_data_ = folly::IOBuf::create(buf_size);
  memcpy(response_data_->writableData(), buf.data(), buf_size);
  response_data_->append(buf_size);

}

int EchoHandler::getFaceRect(const cv::Mat& decoded_image) {

  long mtime, seconds, useconds;    
  struct timeval start, end;

  gettimeofday(&start, NULL);

  // 1. dlib detect face
  //FaceDetect detector = FaceDetect();

  // TODO: detector init need 200ms ?
  //detector.init();

  //dlib::cv_image<dlib::bgr_pixel> dlib_image = decoded_image;
  //std::vector<dlib::rectangle> rects = detector.detect_face( dlib_image );

  //if(rects.size() == 1) {
  //  face_rect_ = rects[0];
  //  return 0;
  //}
  //else if(rects.size() == 0) {
  //  return -1;
  //}
  //else {
  //  return -2;

  // 2. opencv detect face
  FaceDetect detector = FaceDetect();
  detector.init();

  std::vector<cv::Rect> faces = detector.detect_face_opencv(decoded_image);

  gettimeofday(&end, NULL);
  seconds  = end.tv_sec  - start.tv_sec;
  useconds = end.tv_usec - start.tv_usec;
  mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;

  LOG(INFO) << "detect face elapse " << mtime << " ms";

  if(faces.size() == 1) {
    face_rect_ = dlib::rectangle( faces[0].x, faces[0].y, 
        faces[0].x + faces[0].width,
        faces[0].y + faces[0].height);
    return 0;
  }
  else if(faces.size() == 0) {
    return -1;
  }
  else {
    return -2;
  }

}

void EchoHandler::onEOM() noexcept {

  // No body
  if(!body_)
  {
    ResponseBuilder(downstream_)
      .status(400, "Bad Request")
      .sendWithEOM();
    return;
  }
  
  getRequestBodyAsWhole();

  int size = image_data_->length();
  const char* p = reinterpret_cast<const char*>(image_data_->data());
  // Create a Size(1, size) Mat object of 8-bit, single-byte elements
  cv::Mat raw_data = cv::Mat( 1, size, CV_8UC1, (void*)p);
  cv::Mat decoded_image =  cv::imdecode(raw_data, CV_LOAD_IMAGE_COLOR);
  if(!decoded_image.data)
  {
    ResponseBuilder(downstream_)
      .status(415, "Unsupported Media Type")
      .sendWithEOM();
    return;
  }

  LOG(INFO) << "OnEOM()"
    << " upload image "
    << " rows:" << decoded_image.rows
    << " cols:" << decoded_image.cols
    << " size:" << size;

  try {

    long mtime, seconds, useconds;    
    struct timeval start, end;

    gettimeofday(&start, NULL);

    int face_detect_result = getFaceRect(decoded_image);
    if (-1 == face_detect_result) {
      ResponseBuilder(downstream_)
        .status(421, "No face detected")
        .sendWithEOM();
      return;
    }
    else if (-2 == face_detect_result) {
      ResponseBuilder(downstream_)
        .status(422, "Multiple faces exist")
        .sendWithEOM();
      return;
    }

    dlib::cv_image<dlib::bgr_pixel> dlib_image = decoded_image;
    dlib::full_object_detection shape = sp_->predictShape(dlib_image, face_rect_);

    FaceMorph morph = FaceMorph();
    morph.init(gender, model);

    cv::Mat rel = morph.morph_image(decoded_image, shape);

    gettimeofday(&end, NULL);

    seconds  = end.tv_sec  - start.tv_sec;
    useconds = end.tv_usec - start.tv_usec;
    mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;
    LOG(INFO) << "Elapse " << mtime << " ms";

    std::vector<uchar> buf(rel.rows * rel.cols);
    bool encode_rel = cv::imencode("*.jpg", rel, buf);
    if(!encode_rel)
    {
      LOG(ERROR) << "Result image encode fail";

      ResponseBuilder(downstream_)
        .status(500, "Internal Server Error")
        .sendWithEOM();
      return;
    }

    convertMatToResponse(buf);

    // send response
    ResponseBuilder(downstream_)
      .status(200, "OK")
      .header("Content-Type", "image/jpeg")
      .body(std::move(response_data_))
      .sendWithEOM();

  }
  catch(std::exception & e) {
    LOG(ERROR) << e.what();

    ResponseBuilder(downstream_)
      .status(500, "Internal Server Error")
      .sendWithEOM();
  }

}

void EchoHandler::onUpgrade(UpgradeProtocol protocol) noexcept {
  // handler doesn't support upgrades
}

void EchoHandler::requestComplete() noexcept {
  delete this;
}

void EchoHandler::onError(ProxygenError err) noexcept {
  delete this;
}

}
