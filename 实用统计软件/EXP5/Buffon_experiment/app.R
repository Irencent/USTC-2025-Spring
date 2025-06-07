library(shiny)
library(ggplot2)
library(reshape2)

# 修改投针函数支持针长参数
cast_needle <- function(plane_width = 20, L = 1) {
  available_range <- plane_width/2 - L
  x_start <- runif(2, -available_range, available_range)
  angle <- runif(1, 0, 2*pi)
  x_end <- x_start + c(L * cos(angle), L * sin(angle))
  cross <- floor(x_start[2]) != floor(x_end[2])
  list(start = x_start, end = x_end, cross = cross)
}

# 修改实验函数支持针长参数
buffon_experiment <- function(B = 2084, plane_width = 10, L = 1, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  X_start <- X_end <- matrix(NA, B, 2)
  cross <- logical(B)
  for (i in 1:B) {
    needle <- cast_needle(plane_width = plane_width, L = L)
    X_start[i, ] <- needle$start
    X_end[i, ] <- needle$end
    cross[i] <- needle$cross
  }
  structure(
    list(start = X_start, end = X_end, cross = cross, plane = plane_width, L = L),
    class = "buffon_experiment"
  )
}

# 修改绘图函数支持针长显示
plot.buffon_experiment <- function(obj) {
  cross <- obj$cross
  X_start <- obj$start
  X_end <- obj$end
  B <- length(cross)
  cols <- rev(hcl(h = seq(15, 375, length = 3), l = 65, c = 100)[1:2])
  
  pi_hat <- round(2*obj$L/mean(cross), 6)
  title <- bquote("Buffon's Needle (L="~.(obj$L)~"): " ~ hat(pi)[B] ~ "=" ~ .(pi_hat))
  
  plot(NA, xlab = "x", ylab = "y",
       xlim = c(-obj$plane/2, obj$plane/2),
       ylim = c(-obj$plane/2, obj$plane/2),
       main = title)
  abline(h = (-obj$plane):obj$plane, lty = 3)
  for (i in 1:B) {
    lines(c(X_start[i,1], X_end[i,1]), 
          c(X_start[i,2], X_end[i,2]),
          col = cols[cross[i] + 1])
  }
}

# 修改收敛函数支持针长参数
converge <- function(B = 2084, plane_width = 10, L = 1, seed = 123, M = 10) {
  if (B < 10) {
    warning("Number of needles too small, using B=10")
    B <- 10
  }
  pi_hat <- matrix(NA, B, M)
  trials <- 1:B
  set.seed(seed)
  
  for (i in 1:M) {
    cross <- buffon_experiment(B = B, plane_width = plane_width, L = L)$cross
    pi_hat[,i] <- 2*L*trials/cumsum(cross)
  }
  
  pi_hat_long <- melt(pi_hat)
  ggplot(pi_hat_long, aes(x = Var1, y = value)) +
    geom_line(aes(color = factor(Var2)), show.legend = FALSE, alpha = 0.5) +
    stat_summary(fun = mean, geom = "line", size = 1, color = "black") +
    geom_hline(yintercept = pi, color = "gray50", size = 1.5, linetype = 2) +
    coord_cartesian(ylim = pi + c(-0.75, 0.75)) +
    labs(x = "Number of Needles", y = expression(hat(pi))) +
    theme_minimal()
}

# 更新UI界面
ui <- fluidPage(
  titlePanel(h4("Buffon's Needle Experiment")),
  sidebarLayout(
    sidebarPanel(
      sliderInput("L", "Needle Length (L)", 0.5, 2.0, 1.0, 0.1),
      numericInput("plane", "Plane Width", 10, 6, 100),
      numericInput("B", "Number of Needles", 100, 20, 1e6),
      sliderInput("conf_level", "Confidence Level", 0.8, 0.99, 0.95, 0.01),
      numericInput("M", "Number of Trials", 1, 1, 100),
      numericInput("seed", "Random Seed", 123, 1, 1e6),
      actionButton("cast", "Cast Needles!", icon = icon("arrow-circle-down"))
    ),
    mainPanel(
      tabsetPanel(
        tabPanel("Experiment Results",
                 plotOutput("exp"),
                 h4("Real-time Estimation:"),
                 verbatimTextOutput("pi_ci")),
        tabPanel("Convergence Plot", plotOutput("conv"))
      )
    )
  )
)

server <- function(input, output, session) {
  observeEvent(input$cast, {
    updateNumericInput(session, "seed", value = round(runif(1, 1, 1e4)))
  })
  
  experiment <- eventReactive(input$cast, {
    buffon_experiment(B = input$B, plane_width = input$plane, 
                      L = input$L, seed = input$seed)
  })
  
  convergence <- eventReactive(input$cast, {
    converge(B = input$B, plane_width = input$plane, 
             L = input$L, seed = input$seed, M = input$M)
  })
  
  # 新增置信区间计算
  ci_calculation <- reactive({
    req(experiment())
    cross <- experiment()$cross
    n <- length(cross)
    k <- sum(cross)
    L <- input$L
    alpha <- 1 - input$conf_level
    
    # 使用Clopper-Pearson精确区间
    p_lower <- qbeta(alpha/2, k, n - k + 1)
    p_upper <- qbeta(1 - alpha/2, k + 1, n - k)
    
    pi_hat <- 2*L*n/k
    pi_lower <- 2*L/(p_upper)
    pi_upper <- 2*L/(p_lower)
    
    list(
      estimate = pi_hat,
      ci = c(pi_lower, pi_upper),
      coverage = input$conf_level
    )
  })
  
  output$exp <- renderPlot({
    plot(experiment())
  }, height = 620)
  
  output$conv <- renderPlot({
    convergence()
  }, height = 620)
  
  # 新增实时显示输出
  output$pi_ci <- renderPrint({
    ci <- ci_calculation()
    cat(sprintf("Current π estimate: %.4f\n", ci$estimate))
    cat(sprintf("%.0f%% Confidence Interval: (%.4f, %.4f)",
                ci$coverage*100, ci$ci[1], ci$ci[2]))
  })
}

shinyApp(ui, server)