
do_plot <- function(path, LENGTH, type, main_text, xaxis, yaxis) {
  
  data <- read.table(path)
  x = c()
  y = c()
  
  for (i in 1:LENGTH) {
    x = c(x, data[1,i])
    y = c(y, data[2,i])
  }
  
  if (type == "regress") {
    x2 = x * x
    fit = lm(y ~ x + x2) 
    summary(fit)
    y_hat = predict(fit)
    jpeg(paste(main_text, ".jpg"), width = 450, height = 300)
    plot(x,y, xlab = xaxis, ylab = yaxis)
    lines(x,y_hat)
    dev.off()
  } else if (type == "line") {
    x = x[-1]
    y = y[-1]
    jpeg(paste(main_text, ".jpg"), width = 450, height = 300)
    plot(x,y, type="b", xlab = xaxis, ylab = yaxis)
    dev.off()
  }
}
do_plot("change_C_linear.txt", 20, "line", "Changing C Parameter (linear Kernel)", "C", "Mean of Variance")
do_plot("change_C_rbf.txt", 20, "line", "Changing C Parameter (gaussian Kernel)", "C", "Mean of Variance")

do_plot("change_Balances_linear.txt", 52, "regress", "Changing Balance (linear Kernel)", "Balance", "Variance")
do_plot("change_Balances_rbf.txt", 52, "regress", "Changing Balance (gaussian Kernel)", "Blanace", "Variance")

do_plot("change_Support_Vectors_linear.txt", 40, "regress", "Changing N Support (linear Kernel)", "#supportvectors", "Variance")
do_plot("change_Support_Vectors_rbf.txt", 40, "regress", "Changing N Support (gaussian Kernel)", "#supportvectors", "Variance")

