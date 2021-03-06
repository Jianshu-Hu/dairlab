#include "examples/goldilocks_models/planning/FoM_stride_length_constraint.h"


namespace dairlib {
namespace goldilocks_models {
namespace planning {

FomStrideLengthConstraint::FomStrideLengthConstraint(
  bool left_stance, int n_q, VectorXd stride_length,
  const std::string& description):
  Constraint(1,
             2 * n_q,
             VectorXd::Zero(1),
             VectorXd::Zero(1),
             description),
  left_stance_(left_stance),
  stride_length_(stride_length) {
}

void FomStrideLengthConstraint::DoEval(const Eigen::Ref<const Eigen::VectorXd>& x,
                                     Eigen::VectorXd* y) const {
  EvaluateConstraint(x, y);
}

void FomStrideLengthConstraint::DoEval(const Eigen::Ref<const AutoDiffVecXd>& x,
                                     AutoDiffVecXd* y) const {
  // forward differencing
  /*double dx = 1e-8;

  VectorXd x_val = autoDiffToValueMatrix(x);
  VectorXd y0, yi;
  EvaluateConstraint(x_val, &y0);

  MatrixXd dy = MatrixXd(y0.size(), x_val.size());
  for (int i = 0; i < x_val.size(); i++) {
    x_val(i) += dx;
    EvaluateConstraint(x_val, &yi);
    x_val(i) -= dx;
    dy.col(i) = (yi - y0) / dx;
  }
  drake::math::initializeAutoDiffGivenGradientMatrix(y0, dy, *y);*/

  // central differencing
  double dx = 1e-8;

  VectorXd x_val = autoDiffToValueMatrix(x);
  VectorXd y0, yi;
  EvaluateConstraint(x_val, &y0);

  MatrixXd dy = MatrixXd(y0.size(), x_val.size());
  for (int i = 0; i < x_val.size(); i++) {
    x_val(i) -= dx / 2;
    EvaluateConstraint(x_val, &y0);
    x_val(i) += dx;
    EvaluateConstraint(x_val, &yi);
    x_val(i) -= dx / 2;
    dy.col(i) = (yi - y0) / dx;
  }
  EvaluateConstraint(x_val, &y0);
  drake::math::initializeAutoDiffGivenGradientMatrix(y0, dy, *y);
}

void FomStrideLengthConstraint::DoEval(const
                                     Eigen::Ref<const VectorX<Variable>>& x,
                                     VectorX<Expression>*y) const {
  throw std::logic_error(
    "This constraint class does not support symbolic evaluation.");
}

void FomStrideLengthConstraint::EvaluateConstraint(
  const Eigen::Ref<const VectorX<double>>& x, VectorX<double>* y) const {
  VectorX<double> q0 = x.head(7);
  VectorX<double> qf = x.tail(7);

  if (left_stance_) {
    VectorX<double> right_foot_pos_x_0(1);
    right_foot_pos_x_0 <<
                        q0(0) - 0.5 * sin(q0(2) + q0(4)) - 0.5 * sin(q0(2) + q0(4) + q0(6));
    VectorX<double> right_foot_pos_x_f(1);
    right_foot_pos_x_f <<
                        qf(0) - 0.5 * sin(qf(2) + qf(4)) - 0.5 * sin(qf(2) + qf(4) + qf(6));
    *y = right_foot_pos_x_f - right_foot_pos_x_0 - stride_length_;
  } else {
    VectorX<double> left_foot_pos_x_0(1);
    left_foot_pos_x_0 <<
                       q0(0) - 0.5 * sin(q0(2) + q0(3)) - 0.5 * sin(q0(2) + q0(3) + q0(5));
    VectorX<double> left_foot_pos_x_f(1);
    left_foot_pos_x_f <<
                       qf(0) - 0.5 * sin(qf(2) + qf(3)) - 0.5 * sin(qf(2) + qf(3) + qf(5));
    *y = left_foot_pos_x_f - left_foot_pos_x_0 - stride_length_;
  }
}


}  // namespace planning
}  // namespace goldilocks_models
}  // namespace dairlib
