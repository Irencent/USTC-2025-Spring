#include <Rcpp.h>
#include <cmath> // 用于sqrt函数

using namespace Rcpp;

// [[Rcpp::export]]
double max_row_norm(NumericMatrix mat) {
  int nrow = mat.nrow();
  int ncol = mat.ncol();
  double max_norm = 0.0;

  for (int i = 0; i < nrow; ++i) {
    double row_sum = 0.0;
    for (int j = 0; j < ncol; ++j) {
      row_sum += mat(i, j) * mat(i, j); // 计算平方和
    }
    double current_norm = sqrt(row_sum);  // 计算模（欧几里得范数）
    if (current_norm > max_norm) {
      max_norm = current_norm;
    }
  }
  return max_norm;
}
