/*
 *  Copyright (c) 2016, Facebook, Inc.
 *  All rights reserved.
 *
 *  This source code is licensed under the BSD-style license found in the
 *  LICENSE file in the root directory of this source tree. An additional grant
 *  of patent rights can be found in the PATENTS file in the same directory.
 *
 */
#pragma once

#include <folly/Memory.h>
#include <proxygen/httpserver/RequestHandler.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <dlib/opencv.h>
#include <dlib/image_processing.h>

namespace proxygen {
class ResponseHandler;
}

namespace EchoService {

class ShapePredictor;

class EchoHandler : public proxygen::RequestHandler {

 public:
  explicit EchoHandler(ShapePredictor* sp);

  void onRequest(std::unique_ptr<proxygen::HTTPMessage> headers)
      noexcept override;

  void onBody(std::unique_ptr<folly::IOBuf> body) noexcept override;

  void onEOM() noexcept override;

  void onUpgrade(proxygen::UpgradeProtocol proto) noexcept override;

  void requestComplete() noexcept override;

  void onError(proxygen::ProxygenError err) noexcept override;

 private:

  void getRequestBodyAsWhole();

  int getFaceRect(const cv::Mat& decoded_image);

  void convertMatToResponse(std::vector<uchar> & buf);

  int gender;

  int model;

  std::unique_ptr<folly::IOBuf> body_;

  std::unique_ptr<folly::IOBuf> image_data_;

  dlib::rectangle face_rect_;

  ShapePredictor* const sp_{nullptr};

  std::unique_ptr<folly::IOBuf> response_data_;

};

}
