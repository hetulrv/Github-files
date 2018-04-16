library(forecast)
plot(AirPassengers)
lAirpassengers <- log(AirPassengers)
plot(lAirpassengers, ylab = "Log(Airpassengers)")
fit1 = stl(lAirpassengers, s.window = "period")  
# period is used in the s.window to keep the seasonal components as same
plot(fit1)
fit1$time.series
exp(fit1$time.series)
d <- as.data.frame(data.matrix(fit1$time.series, exp(fit1$time.series)))
#d
#par(mfrow = c(2,1))
monthplot(AirPassengers)
seasonplot(AirPassengers, year.labels = TRUE, main = "Seasonal Plot")
plot(fit1)
mods<- c('ANN', 'AAN','ZZZ')
d<-NULL
g<-NULL
h<-NULL
rmse <- NULL
for (i in mods){
  print(i)
  fit2 <- ets(nhtemp,model = i)
  plot(fit2)
  print(fit2)
  forecast(fit2,1)
  plot(forecast(fit2,1),xlab = "Year", ylab = expression(paste("Temperature(",degree*F, ")",)),main = "New Haven Annual Mean Temperature")
  accuracy(fit2)
  print(fit2$aic)
  d[i] = fit2$aic
  g[i] = fit2$bic
  h[i] = fit2$mse
  rmse[i] <- sqrt(fit2$mse)
  }
print("AIC values")
print(d)
print("BIC values")
print(g)
print("MSE values")
print(h)
print(fit2$mse)
rmse <- sqrt(fit2$mse)
print(rmse)