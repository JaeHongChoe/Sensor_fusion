#include "KalmanFilter.hpp"

namespace XyCar
{
KalmanFilter::KalmanFilter(const YAML::Node &config)
{
  dt_ = 1.0 / 30;
  slope_derivative_ = config["KALMAN"]["SLOPE_DER"].as<PREC>();
  intercept_derivative_ = config["KALMAN"]["INTERCEPT_DER"].as<PREC>();

  state_matrix_ = cv::Mat_<PREC>(4, 1);
  transition_matrix_ = (cv::Mat_<PREC>(4, 4) << 1, dt_, 0, 0, 0, 1, 0, 0, 0, 0, 1, dt_, 0, 0, 0, 1);
  measurement_matrix_ = (cv::Mat_<PREC>(2, 4) << 1, 0, 0, 0, 0, 0, 1, 0);
  process_noise_matrix_ = cv::Mat::eye(4, 4, CV_64F);
  measurement_noise_matrix_ = (cv::Mat_<PREC>(2, 2) << 50, 0, 0, 50);
  covariance_matrix_ = 100 * cv::Mat::eye(4, 4, CV_64F);
}

void KalmanFilter::kalman_filtering(PREC slope, PREC intercept)
{
  if (is_first_) {
    state_matrix_.at<PREC>(0, 0) = slope;
    state_matrix_.at<PREC>(2, 0) = intercept;
    is_first_ = false;
  }
  predict(state_matrix_.at<PREC>(0, 0), state_matrix_.at<PREC>(2, 0));
  update(slope, intercept);
}

void KalmanFilter::predict(PREC estimation_slope, PREC estimation_intercept)
{
  state_matrix_ = transition_matrix_ * (cv::Mat_<PREC>(4, 1) << estimation_slope, slope_derivative_, estimation_intercept, intercept_derivative_);
  static const auto transition_matrix_t = transition_matrix_.t();
  covariance_matrix_ = transition_matrix_ * covariance_matrix_ * transition_matrix_t + process_noise_matrix_;
}

void KalmanFilter::update(PREC slope, PREC intercept)
{
  // kalman gain
  static const auto measurement_matrix_t = measurement_matrix_.t();
  kalman_gain_ = covariance_matrix_ * measurement_matrix_t * (measurement_matrix_ * covariance_matrix_ * measurement_matrix_t + measurement_noise_matrix_).inv();

  // if (pos > 0 && pos < 640)
  if (std::round(slope) != 0 || std::round(intercept) != 0) {
    cv::Mat measurement = (cv::Mat_<PREC>(2, 1) << slope, intercept);
    state_matrix_ += kalman_gain_ * (measurement - measurement_matrix_ * state_matrix_);
  }

  // Update covariance
  covariance_matrix_ -= kalman_gain_ * measurement_matrix_ * covariance_matrix_;
}

const cv::Mat_<PREC> &KalmanFilter::get_state() const { return state_matrix_; }
} // namespace XyCar
