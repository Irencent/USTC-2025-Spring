#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
NumericVector convolve_cpp(NumericVector x, NumericVector y) {
  int n = x.size();
  int m = y.size();
  int k_max = n + m - 1;  // 结果向量长度
  NumericVector z(k_max);

  for (int k = 0; k < k_max; ++k) {
    double sum = 0.0;
    // 确定i的范围：i >= max(0, k - m + 1) 且 i <= min(k, n-1)
    int i_start = std::max(0, k - m + 1);
    int i_end = std::min(k, n - 1);
    for (int i = i_start; i <= i_end; ++i) {
      int j = k - i;  // j = k - i，确保 j >= 0 且 j < m
      sum += x[i] * y[j];
    }
    z[k] = sum;
  }
  return z;
}
