#include <Rcpp.h>
#include <cmath>

using namespace Rcpp;

// [[Rcpp::export]]
double monte_carlo_integral(double a, double b, int B, int seed) {
  if (a >= b) {
    stop("Invalid interval: a must be less than b.");
  }

  // 设置随机种子
  Environment base = Environment("package:base");
  Function set_seed = base["set.seed"];
  set_seed(seed);

  // 生成均匀分布样本
  NumericVector u = runif(B, a, b);

  // 计算积分估计值
  double sum = 0.0;
  for (int i = 0; i < B; ++i) {
    double x = u[i];
    sum += exp(-0.5 * x * x) / sqrt(2 * M_PI);
  }
  return (b - a) * sum / B;
}
