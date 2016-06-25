data <- read.table("test5.txt")

x = c()
y = c()
for (i in 1:40) {
  x = c(x, data[1,i])
  y = c(y, data[2,i])
}


fit = lm(y ~ x) 
summary(fit)

plot(x,y)
abline(fit)
