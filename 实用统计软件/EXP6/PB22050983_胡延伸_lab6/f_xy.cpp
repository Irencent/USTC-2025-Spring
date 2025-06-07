#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
NumericVector f_xy(NumericVector x, NumericVector y) {
  if (x.size() != y.size()) {
    stop("x and y must have the same length.");
  }
  int n = x.size();
  NumericVector result(n);
  for (int i = 0; i < n; ++i) {
    if (x[i] < y[i]) {
      result[i] = pow(x[i], 2);
    } else {
      result[i] = -pow(x[i] - y[i], 2);
    }
  }
  return result;
}
