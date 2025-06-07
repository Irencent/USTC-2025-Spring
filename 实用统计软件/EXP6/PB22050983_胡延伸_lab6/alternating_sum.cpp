#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
double alternating_sum(NumericVector x) {
  double sum = 0.0;
  for (int i = 0; i < x.size(); ++i) {
    sum += (i % 2 == 0) ? -x[i] : x[i];  // 奇数索引（从1开始）乘-1，偶数乘+1
  }
  return sum;
}
