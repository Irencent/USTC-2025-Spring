#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
DataFrame create_df() {
  // 创建数值列 x = (6, 6, 6)
  NumericVector x = NumericVector::create(6, 6, 6);

  // 创建字符列 s = ("xyz", "ABC", "456")
  CharacterVector s = CharacterVector::create("xyz", "ABC", "456");

  // 组合为数据框并命名列
  return DataFrame::create(
    Named("x") = x,
    Named("s") = s
  );
}
