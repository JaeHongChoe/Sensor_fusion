#ifndef LANE_DETECTION__KALMANFILTER_HPP
#define LANE_DETECTION__KALMANFILTER_HPP

// Third party header
#include "opencv2/opencv.hpp"

namespace XyCar
{
/**
 * @details  Class of the Kalman filter.
 */
class KalmanFilter
{
public:
  using Ptr = KalmanFilter *;
  KalmanFilter(const YAML::Node &config);

  void kalman_filtering(PREC slope, PREC intercept);

  /**
   * @details Get the state matrix.
   *          When you want to receive the slope and intercept, you receive it using the cv::Mat::at() method.
   *          ex) slope = state_mat.at<PREC>(0, 0);
   *          ex) intercept = state_mat.at<PREC>(2, 0);
   * @return const Mat&
   */
  const cv::Mat_<PREC> &get_state() const;

private:
  PREC slope_derivative_;     ///< The differential term of slope of lane.
  PREC intercept_derivative_; ///< The differential term of intercept of lane.
  PREC dt_;
  bool is_first_ = true;

  cv::Mat_<PREC> state_matrix_; ///< x: The State estimates (상태 추정치)
  cv::Mat_<PREC> transition_matrix_; ///< A: The State Transistion Matrix (상태 변환 행렬)
  cv::Mat_<PREC> measurement_matrix_; ///< H: The Measurement Matrix (측정 행렬)
  cv::Mat_<PREC> process_noise_matrix_; ///< Q: The Process Noise Covariance Matrix (과정 노이즈 공분산)
  cv::Mat_<PREC> measurement_noise_matrix_; ///< R: The Measurement Noise Covariance Matrix (측정 노이즈 공분산)
  cv::Mat_<PREC> covariance_matrix_; ///< P: The Error Covariance matrix (오차 공분산)
  cv::Mat_<PREC> kalman_gain_; ///< K: The Kalman gain (칼만 이득)

  /**
   * @details Perform the prediction steps of the Kalman filter.
   *          Predicts the state vector using a given 'slope' and 'intercept'.
   *          Then, calculate the predicted covariance using the transition matrix and the covariance matrix.
   * @param[in] slope The slope of lane.
   * @param[in] intercept The intercept of lane.
   * @return void
   */
  void predict(PREC slope, PREC intercept);

  /**
   * @details Update the internal state of a Kalman filter given the 'average slope', 'average intercept' and 'position of lane'.
              The Kalman gain is computed, and the state matrix and covariance matrix are updated accordingly.
              If the position is less than or equal to 0, or greater than or equal to 640,
              the state matrix is updated with a measurement matrix derived from the average slope and intercept.
   * @param[in] avg_slope The average slope of the lane.
   * @param[in] avg_intercept The intercept slope of the lane.
   * @return void
   */
  void update(PREC avg_slope, PREC avg_intercept);
};
} // namespace XyCar

#endif // LANE_DETECTION__KALMANFILTER_HPP
