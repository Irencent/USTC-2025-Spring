#' Monte Carlo Integration
#'
#' @title Monte Carlo Integration
#' @description Estimate the integral of a function over a specified interval using Monte Carlo methods.
#' @param x_range A numeric vector of length 2 defining the integration interval [a, b].
#' @param fun A character string representing the function to integrate (e.g., "x^2").
#' @param B Number of Monte Carlo samples (positive integer).
#' @param seed Random seed (default: 123).
#' @return A list containing the estimated integral value (`I`) and variance (`var`).
#' @export
#' @importFrom stats runif
mc_int <- function(x_range, fun, B, seed = 123) {
  # 参数检查
  if (length(x_range) != 2 || x_range[1] >= x_range[2]) {
    stop("x_range must be a vector of length 2 with a < b.")
  }
  if (!is.character(fun)) {
    stop("fun must be a character string.")
  }
  if (B < 1) {
    stop("B must be a positive integer.")
  }

  # 解析函数表达式
  fun_expr <- parse(text = fun)

  # 验证函数可执行
  test_fun <- try(eval(fun_expr, envir = list(x = 0)), silent = TRUE)
  if (inherits(test_fun, "try-error")) {
    stop("Function cannot be evaluated. Check syntax.")
  }

  # 蒙特卡洛积分
  set.seed(seed)
  a <- x_range[1]
  b <- x_range[2]
  u <- runif(B, min = a, max = b)
  f_u <- eval(fun_expr, envir = list(x = u))  # 使用解析后的表达式
  I <- (b - a) * mean(f_u)
  var <- (b - a)^2 * var(f_u) / B

  # 返回结果（包含解析后的表达式）
  result <- structure(
    list(
      I = I,
      var = var,
      x_range = x_range,
      fun = fun_expr  # 存储表达式，而非字符串
    ),
    class = "MCI"
  )
  return(result)
}
