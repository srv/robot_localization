/*
 * Copyright (c) 2014, 2015, 2016, Charles River Analytics, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above
 * copyright notice, this list of conditions and the following
 * disclaimer in the documentation and/or other materials provided
 * with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived
 * from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "robot_localization/ekf.h"
#include "robot_localization/filter_common.h"

#include <xmlrpcpp/XmlRpcException.h>

#include <iomanip>
#include <limits>
#include <sstream>
#include <vector>

namespace RobotLocalization
{
  Ekf::Ekf(std::vector<double>) :
    FilterBase()  // Must initialize filter base!
  {
  }

  Ekf::~Ekf()
  {
  }

  void Ekf::correct(const Measurement &measurement)
  {
    FB_DEBUG("---------------------- Ekf::correct ----------------------\n" <<
             "State is:\n" << state_ << "\n"
             "Topic is:\n" << measurement.topicName_ << "\n"
             "Measurement is:\n" << measurement.measurement_ << "\n"
             "Measurement topic name is:\n" << measurement.topicName_ << "\n\n"
             "Measurement covariance is:\n" << measurement.covariance_ << "\n");

    // We don't want to update everything, so we need to build matrices that only update
    // the measured parts of our state vector. Throughout prediction and correction, we
    // attempt to maximize efficiency in Eigen.

    // First, determine how many state vector values we're updating
    std::vector<size_t> updateIndices;
    for (size_t i = 0; i < measurement.updateVector_.size(); ++i)
    {
      if (measurement.updateVector_[i])
      {
        // Handle nan and inf values in measurements
        if (std::isnan(measurement.measurement_(i)))
        {
          FB_DEBUG("Value at index " << i << " was nan. Excluding from update.\n");
        }
        else if (std::isinf(measurement.measurement_(i)))
        {
          FB_DEBUG("Value at index " << i << " was inf. Excluding from update.\n");
        }
        else
        {
          updateIndices.push_back(i);
        }
      }
    }

    FB_DEBUG("Update indices are:\n" << updateIndices << "\n");

    size_t updateSize = updateIndices.size();

    // Now set up the relevant matrices
    Eigen::VectorXd stateSubset(updateSize);                              // x (in most literature)
    Eigen::VectorXd measurementSubset(updateSize);                        // z
    Eigen::MatrixXd measurementCovarianceSubset(updateSize, updateSize);  // R
    Eigen::MatrixXd stateToMeasurementSubset(updateSize, state_.rows());  // H
    Eigen::MatrixXd kalmanGainSubset(state_.rows(), updateSize);          // K
    Eigen::VectorXd innovationSubset(updateSize);                         // z - Hx

    stateSubset.setZero();
    measurementSubset.setZero();
    measurementCovarianceSubset.setZero();
    stateToMeasurementSubset.setZero();
    kalmanGainSubset.setZero();
    innovationSubset.setZero();

    // Now build the sub-matrices from the full-sized matrices
    for (size_t i = 0; i < updateSize; ++i)
    {
      measurementSubset(i) = measurement.measurement_(updateIndices[i]);
      stateSubset(i) = state_(updateIndices[i]);

      for (size_t j = 0; j < updateSize; ++j)
      {
        measurementCovarianceSubset(i, j) = measurement.covariance_(updateIndices[i], updateIndices[j]);
      }

      // Handle negative (read: bad) covariances in the measurement. Rather
      // than exclude the measurement or make up a covariance, just take
      // the absolute value.
      if (measurementCovarianceSubset(i, i) < 0)
      {
        FB_DEBUG("WARNING: Negative covariance for index " << i <<
                 " of measurement (value is" << measurementCovarianceSubset(i, i) <<
                 "). Using absolute value...\n");

        measurementCovarianceSubset(i, i) = ::fabs(measurementCovarianceSubset(i, i));
      }

      // If the measurement variance for a given variable is very
      // near 0 (as in e-50 or so) and the variance for that
      // variable in the covariance matrix is also near zero, then
      // the Kalman gain computation will blow up. Really, no
      // measurement can be completely without error, so add a small
      // amount in that case.
      if (measurementCovarianceSubset(i, i) < 1e-9)
      {
        FB_DEBUG("WARNING: measurement had very small error covariance for index " << updateIndices[i] <<
                 ". Adding some noise to maintain filter stability.\n");

        measurementCovarianceSubset(i, i) = 1e-9;
      }
    }

    // The state-to-measurement function, h, will now be a measurement_size x full_state_size
    // matrix, with ones in the (i, i) locations of the values to be updated
    for (size_t i = 0; i < updateSize; ++i)
    {
      stateToMeasurementSubset(i, updateIndices[i]) = 1;
    }

    FB_DEBUG("Current state subset is:\n" << stateSubset <<
             "\nMeasurement subset is:\n" << measurementSubset <<
             "\nMeasurement covariance subset is:\n" << measurementCovarianceSubset <<
             "\nState-to-measurement subset is:\n" << stateToMeasurementSubset << "\n");

    // (1) Compute the Kalman gain: K = (PH') / (HPH' + R)
    Eigen::MatrixXd pht = estimateErrorCovariance_ * stateToMeasurementSubset.transpose();
    Eigen::MatrixXd hphrInv  = (stateToMeasurementSubset * pht + measurementCovarianceSubset).inverse();
    kalmanGainSubset.noalias() = pht * hphrInv;

    innovationSubset = (measurementSubset - stateSubset);

    // Wrap angles in the innovation
    for (size_t i = 0; i < updateSize; ++i)
    {
      if (updateIndices[i] == StateMemberRoll  ||
          updateIndices[i] == StateMemberPitch ||
          updateIndices[i] == StateMemberYaw)
      {
        while (innovationSubset(i) < -PI)
        {
          innovationSubset(i) += TAU;
        }

        while (innovationSubset(i) > PI)
        {
          innovationSubset(i) -= TAU;
        }
      }
    }

    // (2) Check Mahalanobis distance between mapped measurement and state.
    if (checkMahalanobisThreshold(innovationSubset, hphrInv, measurement.mahalanobisThresh_))
    {
      // (3) Apply the gain to the difference between the state and measurement: x = x + K(z - Hx)
      state_.noalias() += kalmanGainSubset * innovationSubset;

      // (4) Update the estimate error covariance using the Joseph form: (I - KH)P(I - KH)' + KRK'
      Eigen::MatrixXd gainResidual = identity_;
      gainResidual.noalias() -= kalmanGainSubset * stateToMeasurementSubset;
      estimateErrorCovariance_ = gainResidual * estimateErrorCovariance_ * gainResidual.transpose();
      estimateErrorCovariance_.noalias() += kalmanGainSubset *
                                            measurementCovarianceSubset *
                                            kalmanGainSubset.transpose();

      // Handle wrapping of angles
      wrapStateAngles();

      // (5) Update the state for posterior smoothing
      if (past_states_.size() > 0)
      {
        past_states_[past_states_.size() - 1].Set(state_, estimateErrorCovariance_);
      }

      FB_DEBUG("Kalman gain subset is:\n" << kalmanGainSubset <<
               "\nInnovation is:\n" << innovationSubset <<
               "\nCorrected full state is:\n" << state_ <<
               "\nCorrected full estimate error covariance is:\n" << estimateErrorCovariance_ <<
               "\n\n---------------------- /Ekf::correct ----------------------\n");
    }
  }

  Eigen::MatrixXd Ekf::computeTransferFunction(const double& delta,
                                               const Eigen::VectorXd& state)
  {
    Eigen::MatrixXd transferFunction(state.size(), state.size());

    // Prepare the invariant parts of the transfer
    // function
    transferFunction.setIdentity();

    double roll = state(StateMemberRoll);
    double pitch = state(StateMemberPitch);
    double yaw = state(StateMemberYaw);
    double xVel = state(StateMemberVx);
    double yVel = state(StateMemberVy);
    double zVel = state(StateMemberVz);
    double rollVel = state(StateMemberVroll);
    double pitchVel = state(StateMemberVpitch);
    double yawVel = state(StateMemberVyaw);
    double xAcc = state(StateMemberAx);
    double yAcc = state(StateMemberAy);
    double zAcc = state(StateMemberAz);

    // We'll need these trig calculations a lot.
    double sp = ::sin(pitch);
    double cp = ::cos(pitch);

    double sr = ::sin(roll);
    double cr = ::cos(roll);

    double sy = ::sin(yaw);
    double cy = ::cos(yaw);

    // Prepare the transfer function
    transferFunction(StateMemberX, StateMemberVx) = cy * cp * delta;
    transferFunction(StateMemberX, StateMemberVy) = (cy * sp * sr - sy * cr) * delta;
    transferFunction(StateMemberX, StateMemberVz) = (cy * sp * cr + sy * sr) * delta;
    transferFunction(StateMemberX, StateMemberAx) = 0.5 * transferFunction(StateMemberX, StateMemberVx) * delta;
    transferFunction(StateMemberX, StateMemberAy) = 0.5 * transferFunction(StateMemberX, StateMemberVy) * delta;
    transferFunction(StateMemberX, StateMemberAz) = 0.5 * transferFunction(StateMemberX, StateMemberVz) * delta;
    transferFunction(StateMemberY, StateMemberVx) = sy * cp * delta;
    transferFunction(StateMemberY, StateMemberVy) = (sy * sp * sr + cy * cr) * delta;
    transferFunction(StateMemberY, StateMemberVz) = (sy * sp * cr - cy * sr) * delta;
    transferFunction(StateMemberY, StateMemberAx) = 0.5 * transferFunction(StateMemberY, StateMemberVx) * delta;
    transferFunction(StateMemberY, StateMemberAy) = 0.5 * transferFunction(StateMemberY, StateMemberVy) * delta;
    transferFunction(StateMemberY, StateMemberAz) = 0.5 * transferFunction(StateMemberY, StateMemberVz) * delta;
    transferFunction(StateMemberZ, StateMemberVx) = -sp * delta;
    transferFunction(StateMemberZ, StateMemberVy) = cp * sr * delta;
    transferFunction(StateMemberZ, StateMemberVz) = cp * cr * delta;
    transferFunction(StateMemberZ, StateMemberAx) = 0.5 * transferFunction(StateMemberZ, StateMemberVx) * delta;
    transferFunction(StateMemberZ, StateMemberAy) = 0.5 * transferFunction(StateMemberZ, StateMemberVy) * delta;
    transferFunction(StateMemberZ, StateMemberAz) = 0.5 * transferFunction(StateMemberZ, StateMemberVz) * delta;
    transferFunction(StateMemberRoll, StateMemberVroll) = transferFunction(StateMemberX, StateMemberVx);
    transferFunction(StateMemberRoll, StateMemberVpitch) = transferFunction(StateMemberX, StateMemberVy);
    transferFunction(StateMemberRoll, StateMemberVyaw) = transferFunction(StateMemberX, StateMemberVz);
    transferFunction(StateMemberPitch, StateMemberVroll) = transferFunction(StateMemberY, StateMemberVx);
    transferFunction(StateMemberPitch, StateMemberVpitch) = transferFunction(StateMemberY, StateMemberVy);
    transferFunction(StateMemberPitch, StateMemberVyaw) = transferFunction(StateMemberY, StateMemberVz);
    transferFunction(StateMemberYaw, StateMemberVroll) = transferFunction(StateMemberZ, StateMemberVx);
    transferFunction(StateMemberYaw, StateMemberVpitch) = transferFunction(StateMemberZ, StateMemberVy);
    transferFunction(StateMemberYaw, StateMemberVyaw) = transferFunction(StateMemberZ, StateMemberVz);
    transferFunction(StateMemberVx, StateMemberAx) = delta;
    transferFunction(StateMemberVy, StateMemberAy) = delta;
    transferFunction(StateMemberVz, StateMemberAz) = delta;

    return transferFunction;
  }

  Eigen::MatrixXd Ekf::computeTransferFunctionJacobian(
      const double& delta,
      const Eigen::VectorXd& state,
      const Eigen::MatrixXd& transferFunction) {
    Eigen::MatrixXd transferFunctionJacobian(transferFunction);

    double roll = state(StateMemberRoll);
    double pitch = state(StateMemberPitch);
    double yaw = state(StateMemberYaw);
    double xVel = state(StateMemberVx);
    double yVel = state(StateMemberVy);
    double zVel = state(StateMemberVz);
    double rollVel = state(StateMemberVroll);
    double pitchVel = state(StateMemberVpitch);
    double yawVel = state(StateMemberVyaw);
    double xAcc = state(StateMemberAx);
    double yAcc = state(StateMemberAy);
    double zAcc = state(StateMemberAz);

    // We'll need these trig calculations a lot.
    double sp = 0.0;
    double cp = 0.0;
    ::sincos(pitch, &sp, &cp);

    double sr = 0.0;
    double cr = 0.0;
    ::sincos(roll, &sr, &cr);

    double sy = 0.0;
    double cy = 0.0;
    ::sincos(yaw, &sy, &cy);

    // Prepare the transfer function Jacobian. This function is analytically derived from the
    // transfer function.
    double xCoeff = 0.0;
    double yCoeff = 0.0;
    double zCoeff = 0.0;
    double oneHalfATSquared = 0.5 * delta * delta;

    yCoeff = cy * sp * cr + sy * sr;
    zCoeff = -cy * sp * sr + sy * cr;
    double dFx_dR = (yCoeff * yVel + zCoeff * zVel) * delta +
                    (yCoeff * yAcc + zCoeff * zAcc) * oneHalfATSquared;
    double dFR_dR = 1 + (yCoeff * pitchVel + zCoeff * yawVel) * delta;

    xCoeff = -cy * sp;
    yCoeff = cy * cp * sr;
    zCoeff = cy * cp * cr;
    double dFx_dP = (xCoeff * xVel + yCoeff * yVel + zCoeff * zVel) * delta +
                    (xCoeff * xAcc + yCoeff * yAcc + zCoeff * zAcc) * oneHalfATSquared;
    double dFR_dP = (xCoeff * rollVel + yCoeff * pitchVel + zCoeff * yawVel) * delta;

    xCoeff = -sy * cp;
    yCoeff = -sy * sp * sr - cy * cr;
    zCoeff = -sy * sp * cr + cy * sr;
    double dFx_dY = (xCoeff * xVel + yCoeff * yVel + zCoeff * zVel) * delta +
                    (xCoeff * xAcc + yCoeff * yAcc + zCoeff * zAcc) * oneHalfATSquared;
    double dFR_dY = (xCoeff * rollVel + yCoeff * pitchVel + zCoeff * yawVel) * delta;

    yCoeff = sy * sp * cr - cy * sr;
    zCoeff = -sy * sp * sr - cy * cr;
    double dFy_dR = (yCoeff * yVel + zCoeff * zVel) * delta +
                    (yCoeff * yAcc + zCoeff * zAcc) * oneHalfATSquared;
    double dFP_dR = (yCoeff * pitchVel + zCoeff * yawVel) * delta;

    xCoeff = -sy * sp;
    yCoeff = sy * cp * sr;
    zCoeff = sy * cp * cr;
    double dFy_dP = (xCoeff * xVel + yCoeff * yVel + zCoeff * zVel) * delta +
                    (xCoeff * xAcc + yCoeff * yAcc + zCoeff * zAcc) * oneHalfATSquared;
    double dFP_dP = 1 + (xCoeff * rollVel + yCoeff * pitchVel + zCoeff * yawVel) * delta;

    xCoeff = cy * cp;
    yCoeff = cy * sp * sr - sy * cr;
    zCoeff = cy * sp * cr + sy * sr;
    double dFy_dY = (xCoeff * xVel + yCoeff * yVel + zCoeff * zVel) * delta +
                    (xCoeff * xAcc + yCoeff * yAcc + zCoeff * zAcc) * oneHalfATSquared;
    double dFP_dY = (xCoeff * rollVel + yCoeff * pitchVel + zCoeff * yawVel) * delta;

    yCoeff = cp * cr;
    zCoeff = -cp * sr;
    double dFz_dR = (yCoeff * yVel + zCoeff * zVel) * delta +
                    (yCoeff * yAcc + zCoeff * zAcc) * oneHalfATSquared;
    double dFY_dR = (yCoeff * pitchVel + zCoeff * yawVel) * delta;

    xCoeff = -cp;
    yCoeff = -sp * sr;
    zCoeff = -sp * cr;
    double dFz_dP = (xCoeff * xVel + yCoeff * yVel + zCoeff * zVel) * delta +
                    (xCoeff * xAcc + yCoeff * yAcc + zCoeff * zAcc) * oneHalfATSquared;
    double dFY_dP = (xCoeff * rollVel + yCoeff * pitchVel + zCoeff * yawVel) * delta;

    // Much of the transfer function Jacobian is identical to the transfer function
    transferFunctionJacobian(StateMemberX, StateMemberRoll) = dFx_dR;
    transferFunctionJacobian(StateMemberX, StateMemberPitch) = dFx_dP;
    transferFunctionJacobian(StateMemberX, StateMemberYaw) = dFx_dY;
    transferFunctionJacobian(StateMemberY, StateMemberRoll) = dFy_dR;
    transferFunctionJacobian(StateMemberY, StateMemberPitch) = dFy_dP;
    transferFunctionJacobian(StateMemberY, StateMemberYaw) = dFy_dY;
    transferFunctionJacobian(StateMemberZ, StateMemberRoll) = dFz_dR;
    transferFunctionJacobian(StateMemberZ, StateMemberPitch) = dFz_dP;
    transferFunctionJacobian(StateMemberRoll, StateMemberRoll) = dFR_dR;
    transferFunctionJacobian(StateMemberRoll, StateMemberPitch) = dFR_dP;
    transferFunctionJacobian(StateMemberRoll, StateMemberYaw) = dFR_dY;
    transferFunctionJacobian(StateMemberPitch, StateMemberRoll) = dFP_dR;
    transferFunctionJacobian(StateMemberPitch, StateMemberPitch) = dFP_dP;
    transferFunctionJacobian(StateMemberPitch, StateMemberYaw) = dFP_dY;
    transferFunctionJacobian(StateMemberYaw, StateMemberRoll) = dFY_dR;
    transferFunctionJacobian(StateMemberYaw, StateMemberPitch) = dFY_dP;

    return transferFunctionJacobian;
  }

  void Ekf::predict(const double referenceTime, const double delta)
  {
    FB_DEBUG("---------------------- Ekf::predict ----------------------\n" <<
             "delta is " << delta << "\n" <<
             "state is " << state_ << "\n");

    double roll = state_(StateMemberRoll);
    double pitch = state_(StateMemberPitch);
    double yaw = state_(StateMemberYaw);
    double xVel = state_(StateMemberVx);
    double yVel = state_(StateMemberVy);
    double zVel = state_(StateMemberVz);
    double rollVel = state_(StateMemberVroll);
    double pitchVel = state_(StateMemberVpitch);
    double yawVel = state_(StateMemberVyaw);
    double xAcc = state_(StateMemberAx);
    double yAcc = state_(StateMemberAy);
    double zAcc = state_(StateMemberAz);

    // We'll need these trig calculations a lot.
    double sp = ::sin(pitch);
    double cp = ::cos(pitch);

    double sr = ::sin(roll);
    double cr = ::cos(roll);

    double sy = ::sin(yaw);
    double cy = ::cos(yaw);

    prepareControl(referenceTime, delta);

    // Prepare the transfer function
    transferFunction_(StateMemberX, StateMemberVx) = cy * cp * delta;
    transferFunction_(StateMemberX, StateMemberVy) = (cy * sp * sr - sy * cr) * delta;
    transferFunction_(StateMemberX, StateMemberVz) = (cy * sp * cr + sy * sr) * delta;
    transferFunction_(StateMemberX, StateMemberAx) = 0.5 * transferFunction_(StateMemberX, StateMemberVx) * delta;
    transferFunction_(StateMemberX, StateMemberAy) = 0.5 * transferFunction_(StateMemberX, StateMemberVy) * delta;
    transferFunction_(StateMemberX, StateMemberAz) = 0.5 * transferFunction_(StateMemberX, StateMemberVz) * delta;
    transferFunction_(StateMemberY, StateMemberVx) = sy * cp * delta;
    transferFunction_(StateMemberY, StateMemberVy) = (sy * sp * sr + cy * cr) * delta;
    transferFunction_(StateMemberY, StateMemberVz) = (sy * sp * cr - cy * sr) * delta;
    transferFunction_(StateMemberY, StateMemberAx) = 0.5 * transferFunction_(StateMemberY, StateMemberVx) * delta;
    transferFunction_(StateMemberY, StateMemberAy) = 0.5 * transferFunction_(StateMemberY, StateMemberVy) * delta;
    transferFunction_(StateMemberY, StateMemberAz) = 0.5 * transferFunction_(StateMemberY, StateMemberVz) * delta;
    transferFunction_(StateMemberZ, StateMemberVx) = -sp * delta;
    transferFunction_(StateMemberZ, StateMemberVy) = cp * sr * delta;
    transferFunction_(StateMemberZ, StateMemberVz) = cp * cr * delta;
    transferFunction_(StateMemberZ, StateMemberAx) = 0.5 * transferFunction_(StateMemberZ, StateMemberVx) * delta;
    transferFunction_(StateMemberZ, StateMemberAy) = 0.5 * transferFunction_(StateMemberZ, StateMemberVy) * delta;
    transferFunction_(StateMemberZ, StateMemberAz) = 0.5 * transferFunction_(StateMemberZ, StateMemberVz) * delta;
    transferFunction_(StateMemberRoll, StateMemberVroll) = transferFunction_(StateMemberX, StateMemberVx);
    transferFunction_(StateMemberRoll, StateMemberVpitch) = transferFunction_(StateMemberX, StateMemberVy);
    transferFunction_(StateMemberRoll, StateMemberVyaw) = transferFunction_(StateMemberX, StateMemberVz);
    transferFunction_(StateMemberPitch, StateMemberVroll) = transferFunction_(StateMemberY, StateMemberVx);
    transferFunction_(StateMemberPitch, StateMemberVpitch) = transferFunction_(StateMemberY, StateMemberVy);
    transferFunction_(StateMemberPitch, StateMemberVyaw) = transferFunction_(StateMemberY, StateMemberVz);
    transferFunction_(StateMemberYaw, StateMemberVroll) = transferFunction_(StateMemberZ, StateMemberVx);
    transferFunction_(StateMemberYaw, StateMemberVpitch) = transferFunction_(StateMemberZ, StateMemberVy);
    transferFunction_(StateMemberYaw, StateMemberVyaw) = transferFunction_(StateMemberZ, StateMemberVz);
    transferFunction_(StateMemberVx, StateMemberAx) = delta;
    transferFunction_(StateMemberVy, StateMemberAy) = delta;
    transferFunction_(StateMemberVz, StateMemberAz) = delta;

    // Prepare the transfer function Jacobian. This function is analytically derived from the
    // transfer function.
    double xCoeff = 0.0;
    double yCoeff = 0.0;
    double zCoeff = 0.0;
    double oneHalfATSquared = 0.5 * delta * delta;

    yCoeff = cy * sp * cr + sy * sr;
    zCoeff = -cy * sp * sr + sy * cr;
    double dFx_dR = (yCoeff * yVel + zCoeff * zVel) * delta +
                    (yCoeff * yAcc + zCoeff * zAcc) * oneHalfATSquared;
    double dFR_dR = 1 + (yCoeff * pitchVel + zCoeff * yawVel) * delta;

    xCoeff = -cy * sp;
    yCoeff = cy * cp * sr;
    zCoeff = cy * cp * cr;
    double dFx_dP = (xCoeff * xVel + yCoeff * yVel + zCoeff * zVel) * delta +
                    (xCoeff * xAcc + yCoeff * yAcc + zCoeff * zAcc) * oneHalfATSquared;
    double dFR_dP = (xCoeff * rollVel + yCoeff * pitchVel + zCoeff * yawVel) * delta;

    xCoeff = -sy * cp;
    yCoeff = -sy * sp * sr - cy * cr;
    zCoeff = -sy * sp * cr + cy * sr;
    double dFx_dY = (xCoeff * xVel + yCoeff * yVel + zCoeff * zVel) * delta +
                    (xCoeff * xAcc + yCoeff * yAcc + zCoeff * zAcc) * oneHalfATSquared;
    double dFR_dY = (xCoeff * rollVel + yCoeff * pitchVel + zCoeff * yawVel) * delta;

    yCoeff = sy * sp * cr - cy * sr;
    zCoeff = -sy * sp * sr - cy * cr;
    double dFy_dR = (yCoeff * yVel + zCoeff * zVel) * delta +
                    (yCoeff * yAcc + zCoeff * zAcc) * oneHalfATSquared;
    double dFP_dR = (yCoeff * pitchVel + zCoeff * yawVel) * delta;

    xCoeff = -sy * sp;
    yCoeff = sy * cp * sr;
    zCoeff = sy * cp * cr;
    double dFy_dP = (xCoeff * xVel + yCoeff * yVel + zCoeff * zVel) * delta +
                    (xCoeff * xAcc + yCoeff * yAcc + zCoeff * zAcc) * oneHalfATSquared;
    double dFP_dP = 1 + (xCoeff * rollVel + yCoeff * pitchVel + zCoeff * yawVel) * delta;

    xCoeff = cy * cp;
    yCoeff = cy * sp * sr - sy * cr;
    zCoeff = cy * sp * cr + sy * sr;
    double dFy_dY = (xCoeff * xVel + yCoeff * yVel + zCoeff * zVel) * delta +
                    (xCoeff * xAcc + yCoeff * yAcc + zCoeff * zAcc) * oneHalfATSquared;
    double dFP_dY = (xCoeff * rollVel + yCoeff * pitchVel + zCoeff * yawVel) * delta;

    yCoeff = cp * cr;
    zCoeff = -cp * sr;
    double dFz_dR = (yCoeff * yVel + zCoeff * zVel) * delta +
                    (yCoeff * yAcc + zCoeff * zAcc) * oneHalfATSquared;
    double dFY_dR = (yCoeff * pitchVel + zCoeff * yawVel) * delta;

    xCoeff = -cp;
    yCoeff = -sp * sr;
    zCoeff = -sp * cr;
    double dFz_dP = (xCoeff * xVel + yCoeff * yVel + zCoeff * zVel) * delta +
                    (xCoeff * xAcc + yCoeff * yAcc + zCoeff * zAcc) * oneHalfATSquared;
    double dFY_dP = (xCoeff * rollVel + yCoeff * pitchVel + zCoeff * yawVel) * delta;

    // Much of the transfer function Jacobian is identical to the transfer function
    transferFunctionJacobian_ = transferFunction_;
    transferFunctionJacobian_(StateMemberX, StateMemberRoll) = dFx_dR;
    transferFunctionJacobian_(StateMemberX, StateMemberPitch) = dFx_dP;
    transferFunctionJacobian_(StateMemberX, StateMemberYaw) = dFx_dY;
    transferFunctionJacobian_(StateMemberY, StateMemberRoll) = dFy_dR;
    transferFunctionJacobian_(StateMemberY, StateMemberPitch) = dFy_dP;
    transferFunctionJacobian_(StateMemberY, StateMemberYaw) = dFy_dY;
    transferFunctionJacobian_(StateMemberZ, StateMemberRoll) = dFz_dR;
    transferFunctionJacobian_(StateMemberZ, StateMemberPitch) = dFz_dP;
    transferFunctionJacobian_(StateMemberRoll, StateMemberRoll) = dFR_dR;
    transferFunctionJacobian_(StateMemberRoll, StateMemberPitch) = dFR_dP;
    transferFunctionJacobian_(StateMemberRoll, StateMemberYaw) = dFR_dY;
    transferFunctionJacobian_(StateMemberPitch, StateMemberRoll) = dFP_dR;
    transferFunctionJacobian_(StateMemberPitch, StateMemberPitch) = dFP_dP;
    transferFunctionJacobian_(StateMemberPitch, StateMemberYaw) = dFP_dY;
    transferFunctionJacobian_(StateMemberYaw, StateMemberRoll) = dFY_dR;
    transferFunctionJacobian_(StateMemberYaw, StateMemberPitch) = dFY_dP;

    FB_DEBUG("Transfer function is:\n" << transferFunction_ <<
             "\nTransfer function Jacobian is:\n" << transferFunctionJacobian_ <<
             "\nProcess noise covariance is:\n" << processNoiseCovariance_ <<
             "\nCurrent state is:\n" << state_ << "\n");

    Eigen::MatrixXd *processNoiseCovariance = &processNoiseCovariance_;

    if (useDynamicProcessNoiseCovariance_)
    {
      computeDynamicProcessNoiseCovariance(state_, delta);
      processNoiseCovariance = &dynamicProcessNoiseCovariance_;
    }

    // (1) Apply control terms, which are actually accelerations
    state_(StateMemberVroll) += controlAcceleration_(ControlMemberVroll) * delta;
    state_(StateMemberVpitch) += controlAcceleration_(ControlMemberVpitch) * delta;
    state_(StateMemberVyaw) += controlAcceleration_(ControlMemberVyaw) * delta;

    state_(StateMemberAx) = (controlUpdateVector_[ControlMemberVx] ?
      controlAcceleration_(ControlMemberVx) : state_(StateMemberAx));
    state_(StateMemberAy) = (controlUpdateVector_[ControlMemberVy] ?
      controlAcceleration_(ControlMemberVy) : state_(StateMemberAy));
    state_(StateMemberAz) = (controlUpdateVector_[ControlMemberVz] ?
      controlAcceleration_(ControlMemberVz) : state_(StateMemberAz));

    // (2) Project the state forward: x = Ax + Bu (really, x = f(x, u))
    state_ = transferFunction_ * state_;

    // Handle wrapping
    wrapStateAngles();

    FB_DEBUG("Predicted state is:\n" << state_ <<
             "\nCurrent estimate error covariance is:\n" <<  estimateErrorCovariance_ << "\n");

    // (3) Project the error forward: P = J * P * J' + Q
    estimateErrorCovariance_ = (transferFunctionJacobian_ *
                                estimateErrorCovariance_ *
                                transferFunctionJacobian_.transpose());
    estimateErrorCovariance_.noalias() += delta * (*processNoiseCovariance);

    // (3) Save the state for posterior smoothing
    EkfState s(referenceTime, state_, estimateErrorCovariance_);
    past_states_.push_back(s);

    FB_DEBUG("Predicted estimate error covariance is:\n" << estimateErrorCovariance_ <<
             "\n\n--------------------- /Ekf::predict ----------------------\n");
  }

  void Ekf::smooth()
  {
    if (past_states_.size() < 2)
    {
      return;
    }
    smoothed_states_.clear();
    smoothed_states_ = past_states_;
    for (size_t i = 0; i < past_states_.size() - 1; i++)
    {
      EkfState sf(smoothed_states_[past_states_.size() - 1 - i]);
      EkfState s(past_states_[past_states_.size() - 2 - i]);
      Eigen::VectorXd x_prior = s.GetState();
      Eigen::VectorXd x_smoothed = sf.GetState();
      double delta = sf.GetTime() - s.GetTime();
      Eigen::MatrixXd P_prior = s.GetCov();
      Eigen::MatrixXd P_smoothed = sf.GetCov();

      Eigen::MatrixXd f = computeTransferFunction(delta, x_prior);
      Eigen::MatrixXd A = computeTransferFunctionJacobian(delta, x_prior, f);

      Eigen::MatrixXd P_prior_pred = A*P_prior*A.transpose() + processNoiseCovariance_ * delta;
      Eigen::MatrixXd J = P_prior * A.transpose() * P_prior_pred.inverse();

      Eigen::VectorXd x_prior_smoothed = x_prior + J*(x_smoothed - f*x_prior);
      Eigen::MatrixXd P_prior_smoothed = P_prior + J*(P_smoothed - P_prior_pred)*J.transpose();
      smoothed_states_[past_states_.size()-2-i].Set(x_prior_smoothed, P_prior_smoothed);
    }
  }
}  // namespace RobotLocalization
