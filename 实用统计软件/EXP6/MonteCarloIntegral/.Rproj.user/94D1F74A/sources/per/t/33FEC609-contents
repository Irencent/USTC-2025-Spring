#' @title Plot Monte Carlo Integration Result
#' @description Visualize the function and integration area.
#' @param x An object of class `MCI` from `mc_int`.
#' @param ... Additional graphical parameters.
#' @method plot MCI
#' @export
plot.MCI <- function(x, ...) {
  if (!inherits(x, "MCI")) {
    stop("Input must be of class 'MCI'.")
  }

  a <- x$x_range[1]
  b <- x$x_range[2]
  delta <- b - a

  # 生成绘图数据
  x_vals <- seq(a - 0.15 * delta, b + 0.15 * delta, length.out = 1000)
  f_vals <- eval(x$fun, envir = list(x = x_vals))  # 显式绑定变量名 x

  # 检查数据是否为数值向量
  if (!is.numeric(x_vals) || !is.numeric(f_vals)) {
    stop("x_vals or f_vals is not numeric.")
  }

  # 绘制函数曲线
  plot(x_vals, f_vals, type = "l", xlab = "x", ylab = "f(x)",
       main = paste("Estimated Integral:", round(x$I, 4)), ...)
  grid()

  # 填充积分区域
  x_shade <- seq(a, b, length.out = 1000)
  f_shade <- eval(x$fun, envir = list(x = x_shade))  # 显式绑定变量名 x
  polygon(
    c(x_shade, rev(x_shade)),
    c(rep(0, 1000), rev(f_shade)),
    col = rgb(0.2, 0.5, 0.8, 0.5),
    border = NA
  )
  abline(v = c(a, b), lty = 2)
}
