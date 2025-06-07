#include <Rcpp.h>
using namespace Rcpp;

//' Multiply numeric vector by two
 //'
 //' @param x A numeric vector
 //' @return Numeric vector with doubled values
 //' @name timesTwo
 //' @export
 // [[Rcpp::export]]
 NumericVector timesTwo(NumericVector x) {
   return x * 2;
 }
