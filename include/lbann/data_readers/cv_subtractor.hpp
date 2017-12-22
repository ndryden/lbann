////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the LBANN Research Team (B. Van Essen, et al.) listed in
// the CONTRIBUTORS file. <lbann-dev@llnl.gov>
//
// LLNL-CODE-697807.
// All rights reserved.
//
// This file is part of LBANN: Livermore Big Artificial Neural Network
// Toolkit. For details, see http://software.llnl.gov/LBANN or
// https://github.com/LLNL/LBANN.
//
// Licensed under the Apache License, Version 2.0 (the "Licensee"); you
// may not use this file except in compliance with the License.  You may
// obtain a copy of the License at:
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the license.
//
// cv_subtractor .cpp .hpp - subtract channel values of an image (possibly the
// pixel-wise mean of dataset) from the corresponding values of another (input)
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CV_SUBTRACTOR_HPP
#define LBANN_CV_SUBTRACTOR_HPP

#include "cv_transform.hpp"
#include "lbann/base.hpp"

#ifdef __LIB_OPENCV
namespace lbann {

class cv_subtractor : public cv_transform {
 protected:
  // --- configuration variables ---
  /**
   * The image to subtract from an input image.
   * It has channel values of a floating point type, in the scale from 0 to 1.
   * An input image will be mapped into the scale before subtraction.
   */
  cv::Mat m_img_to_sub;

  // --- state variables ---
  bool m_subtracted; ///< has been subtracted

 public:
  cv_subtractor() : cv_transform(), m_subtracted(false) {}
  cv_subtractor(const cv_subtractor& rhs);
  cv_subtractor& operator=(const cv_subtractor& rhs);
  cv_subtractor *clone() const override;

  ~cv_subtractor() override {}

  /**
   * Set the image to subtract from every input image.
   * In case that this image is not in a floating point type, it is converted to
   * one with the depth specified by depth_code.
   */ 
  void set(const cv::Mat& img, const int depth_code = cv_image_type<::DataType>::T());

  /// Load and set the image to subtract from every input image.
  void set(const std::string name_of_img, const int depth_code = cv_image_type<::DataType>::T());

  void reset() override {
    m_enabled = false;
    m_subtracted = false;
  }

  /**
   * If a given image is in grayscale, the tranform is enabled, and not otherwise.
   * @return false if not enabled or unsuccessful.
   */
  bool determine_transform(const cv::Mat& image) override;

  /// convert back to color image if it used to be a grayscale image
  bool determine_inverse_transform() override;

  /**
   * Apply color conversion if enabled.
   * As it is applied, the transform becomes deactivated.
   * @return false if not successful.
   */
  bool apply(cv::Mat& image) override;

  std::string get_type() const override { return "subtractor"; }
  std::string get_description() const override;
  std::ostream& print(std::ostream& os) const override;
};

} // end of namespace lbann
#endif // __LIB_OPENCV

#endif // LBANN_CV_SUBTRACTOR_HPP