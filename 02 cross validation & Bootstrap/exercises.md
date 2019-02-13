Exercises - CV & Bootstrap
================
Amin Yakubu
2/12/2019

``` r
library(ISLR)
```

Applied

Question 5

``` r
data("Default")
```

Fit a logistic regression model that uses income and balance to predict default.

``` r
attach(Default)
glm.fit = glm(default ~ income + balance, data = Default, family = 'binomial')
```

Using the validation set approach, estimate the test error of this model.

``` r
validation = function() {
  #Split the sample set into a training set and a validation set.
  train = sample(dim(Default)[1], dim(Default)[1]/2)
  
  #Fit a multiple logistic regression model using only the train- ing observations.
  glm.fit = glm(default ~ income + balance, data = Default, family = 'binomial', subset = train)
  
  #Obtain a prediction of default status for each individual in the validation set by computing the posterior
  #probability of default for that individual, and classifying the individual to the default category if the posterior
  #probability is greater than 0.5.
  
  glm.pred = rep("No", dim(Default)[1]/2)
  glm.probs = predict(glm.fit, Default[-train, ], type = "response")
  glm.pred[glm.probs > 0.5] = "Yes"
  
  #Compute the validation set error, which is the fraction of the observations in the validation set that are 
  #misclassified.
  return(mean(glm.pred != Default[-train, ]$default))
}
```

``` r
validation()
```

    ## [1] 0.0264

Repeat the process in (b) three times, using three different splits of the observations into a training set and a validation set. Comment on the results obtained

``` r
error = rep(0, 3)

for (i in 1:3) {
  value = validation()
  error[i] = value
}

sum(error)/3
```

    ## [1] 0.0258

Now consider a logistic regression model that predicts the probability of default using income, balance, and a dummy variable for student. Estimate the test error for this model using the validation set approach. Comment on whether or not including a dummy variable for student leads to a reduction in the test error rate.

``` r
glm.fit = glm(default ~ income + balance + student, data = Default, family = 'binomial')
```

Income is no more significant

Let's estimate the error rate

``` r
train = sample(nrow(Default), nrow(Default)/2)
glm.fit = glm(default ~ income + balance + student, data = Default, family = 'binomial', subset = train)

glm.probs = predict(glm.fit, Default[-train,], type = 'response')

glm.pred = rep("No", nrow(Default)/2)
glm.pred[glm.probs > 0.5] = 'Yes'

error = mean(glm.pred != Default[-train, ]$default)
```

Using the validation set approach, it doesn't appear adding the student dummy variable leads to a reduction in the test error rate.

Question 6 We continue to consider the use of a logistic regression model to predict the probability of default using income and balance on the Default data set. In particular, we will now compute estimates for the standard errors of the income and balance logistic regression co- efficients in two different ways: (1) using the bootstrap, and (2) using the standard formula for computing the standard errors in the glm() function.

``` r
glm.fit = glm(default ~ income + balance, data = Default, family = 'binomial')
summary(glm.fit)
```

    ## 
    ## Call:
    ## glm(formula = default ~ income + balance, family = "binomial", 
    ##     data = Default)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -2.4725  -0.1444  -0.0574  -0.0211   3.7245  
    ## 
    ## Coefficients:
    ##               Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept) -1.154e+01  4.348e-01 -26.545  < 2e-16 ***
    ## income       2.081e-05  4.985e-06   4.174 2.99e-05 ***
    ## balance      5.647e-03  2.274e-04  24.836  < 2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 2920.6  on 9999  degrees of freedom
    ## Residual deviance: 1579.0  on 9997  degrees of freedom
    ## AIC: 1585
    ## 
    ## Number of Fisher Scoring iterations: 8

``` r
library(boot)
```

``` r
boot.fn = function(data, index) {
  
  return(coef(glm(default ~ income + balance, data = data, subset = index, family = 'binomial'))) 
}

# Testing the function
boot.fn(Default, 1:nrow(Default))
```

    ##   (Intercept)        income       balance 
    ## -1.154047e+01  2.080898e-05  5.647103e-03

``` r
# Bootstrap
set.seed(1)
boot.fn(Default, sample(nrow(Default), nrow(Default), replace = T)) 
```

    ##   (Intercept)        income       balance 
    ## -1.115930e+01  2.134762e-05  5.419225e-03

``` r
# Using the boot() to repeat the step above 1000 times and get the standard errors

boot(Default, boot.fn, 1000)
```

    ## 
    ## ORDINARY NONPARAMETRIC BOOTSTRAP
    ## 
    ## 
    ## Call:
    ## boot(data = Default, statistic = boot.fn, R = 1000)
    ## 
    ## 
    ## Bootstrap Statistics :
    ##          original        bias     std. error
    ## t1* -1.154047e+01 -8.648704e-03 4.240313e-01
    ## t2*  2.080898e-05  5.360985e-08 4.588898e-06
    ## t3*  5.647103e-03  2.800720e-06 2.268846e-04

Similar answers to the second and third significant digits.

Question 7

we saw that the cv.glm() function can be used in order to compute the LOOCV test error estimate. Alterna- tively, one could compute those quantities using just the glm() and predict.glm() functions, and a for loop. You will now take this ap- proach in order to compute the LOOCV error for a simple logistic regression model on the Weekly data set

``` r
data("Weekly")
```

Fit a logistic regression model that predicts Direction using Lag1 and Lag2

``` r
glm.fit = glm(Direction ~ Lag1 + Lag2, data = Weekly, family = 'binomial')
summary(glm.fit)
```

    ## 
    ## Call:
    ## glm(formula = Direction ~ Lag1 + Lag2, family = "binomial", data = Weekly)
    ## 
    ## Deviance Residuals: 
    ##    Min      1Q  Median      3Q     Max  
    ## -1.623  -1.261   1.001   1.083   1.506  
    ## 
    ## Coefficients:
    ##             Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)  0.22122    0.06147   3.599 0.000319 ***
    ## Lag1        -0.03872    0.02622  -1.477 0.139672    
    ## Lag2         0.06025    0.02655   2.270 0.023232 *  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 1496.2  on 1088  degrees of freedom
    ## Residual deviance: 1488.2  on 1086  degrees of freedom
    ## AIC: 1494.2
    ## 
    ## Number of Fisher Scoring iterations: 4

Fit a logistic regression model that predicts Direction using Lag1 and Lag2 using all but the first observation

``` r
glm.fit2 = glm(Direction ~ Lag1 + Lag2, data = Weekly[2:nrow(Weekly), ], family = 'binomial')
summary(glm.fit2)
```

    ## 
    ## Call:
    ## glm(formula = Direction ~ Lag1 + Lag2, family = "binomial", data = Weekly[2:nrow(Weekly), 
    ##     ])
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -1.6258  -1.2617   0.9999   1.0819   1.5071  
    ## 
    ## Coefficients:
    ##             Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)  0.22324    0.06150   3.630 0.000283 ***
    ## Lag1        -0.03843    0.02622  -1.466 0.142683    
    ## Lag2         0.06085    0.02656   2.291 0.021971 *  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 1494.6  on 1087  degrees of freedom
    ## Residual deviance: 1486.5  on 1085  degrees of freedom
    ## AIC: 1492.5
    ## 
    ## Number of Fisher Scoring iterations: 4

Use the model from glm.fit model to predict the direction of the first observation. You can do this by predicting that the first observation will go up if P(Direction="Up"|Lag1, Lag2) &gt; 0.5. Was this ob- servation correctly classified?

``` r
glm.prob1 = predict(glm.fit, Weekly[1,], type = 'response')

glm.pred1 = list('Down')

glm.pred1[glm.prob1 > 0.5] = 'Up'

glm.pred1
```

    ## [[1]]
    ## [1] "Up"

``` r
glm.pred1 == Weekly[1, ]$Direction
```

    ## [1] FALSE

Prediction was UP, true Direction was DOWN.

Writeaforloopfromi=1toi=n,wherenisthenumberof observations in the data set, that performs each of the following steps:

1.  Fit a logistic regression model using all but the ith obser- vation to predict Direction using Lag1 and Lag2.
2.  Compute the posterior probability of the market moving up for the ith observation.
3.  Use the posterior probability for the ith observation in order to predict whether or not the market moves up.
4.  Determine whether or not an error was made in predicting the direction for the ith observation. If an error was made, then indicate this as a 1, and otherwise indicate it as a 0.

``` r
count = rep(0, nrow(Weekly))

for (i in 1:nrow(Weekly)) {
  glm.fit = glm(Direction ~ Lag1 + Lag2, family = 'binomial', data = Weekly[-i,])
  is_up = predict(glm.fit, Weekly[i,], type = 'response') > 0.5
  is_true_up = Weekly[i,]$Direction == "Up"
  if (is_up != is_true_up)
    count[i] = 1
  
}

sum(count)
```

    ## [1] 490

490 errors.

``` r
mean(count)
```

    ## [1] 0.4499541

The error tate is 0.45

Question 8

We will now perform cross-validation on a simulated data set.

``` r
set.seed(1)
x = rnorm(100)
y = x - 2 * x^2  + rnorm(100)
```

n = 100, p = 2.

Y=X−2X2+ϵ.

``` r
plot(x, y)
```

![](exercises_files/figure-markdown_github/unnamed-chunk-19-1.png) Quadratic plot. X from about -2 to 2. Y from about -8 to 2.

Set a random seed, and then compute the LOOCV errors that result from fitting the following four models using least squares:

``` r
library(rlist)
df = data.frame(x = x,
                y = y)

delta1 = list()
delta2 = list()

for (i in 1:4) {
  glm.fit = glm(y ~ poly(x, i), data = df)
  error1 = cv.glm(df, glm.fit)$delta[1]
  delta1[i] = error1
  error2 = cv.glm(df, glm.fit)$delta[2]
  delta2[i] = error2
}
```

The results will be same if we rerun the bootstrap

The quadratic model had the smallest LOOCV error because from the simulated data we can see that the relationship between x and y is quadratic.

Comment on the statistical significance of the coefficient esti- mates that results from fitting each of the models in (c) using least squares. Do these results agree with the conclusions drawn based on the cross-validation results?

``` r
summary(glm(y ~ poly(x, 4), data = df))
```

    ## 
    ## Call:
    ## glm(formula = y ~ poly(x, 4), data = df)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -2.0550  -0.6212  -0.1567   0.5952   2.2267  
    ## 
    ## Coefficients:
    ##              Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)  -1.55002    0.09591 -16.162  < 2e-16 ***
    ## poly(x, 4)1   6.18883    0.95905   6.453 4.59e-09 ***
    ## poly(x, 4)2 -23.94830    0.95905 -24.971  < 2e-16 ***
    ## poly(x, 4)3   0.26411    0.95905   0.275    0.784    
    ## poly(x, 4)4   1.25710    0.95905   1.311    0.193    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for gaussian family taken to be 0.9197797)
    ## 
    ##     Null deviance: 700.852  on 99  degrees of freedom
    ## Residual deviance:  87.379  on 95  degrees of freedom
    ## AIC: 282.3
    ## 
    ## Number of Fisher Scoring iterations: 2

The p-values show that the linear and quadratic terms are statistically significants and that the cubic and 4th degree terms are not statistically significants. This agree strongly with our cross-validation results which were minimum for the quadratic model.

Question 9

``` r
library(MASS)
data("Boston")
```

Based on this data set, provide an estimate for the population mean of medv. Call this estimate μˆ.

``` r
mu_hat = mean(Boston$medv)
```

Provide an estimate of the standard error of μˆ. Interpret this result We can compute the standard error of the sample mean by dividing the sample standard deviation by the square root of the number of observations.

``` r
standard.error = sd(Boston$medv)/sqrt(nrow(Boston))
```

``` r
boot.sample.mean = function(data, index){
  return(mean(data[index]))
}

boot.sample.mean(Boston$medv, sample(nrow(Boston), nrow(Boston), replace = T))
```

    ## [1] 22.24229

``` r
set.seed(1)
boot.results = boot(Boston$medv, boot.sample.mean, 1000)
```

The bootstrap estimated standard error of μ̂ of 0.4119 is very close to the estimate found in (b) of 0.4089.

Based on your bootstrap estimate from (c), provide a 95 % con- fidence interval for the mean of medv. Compare it to the results obtained using t.test(Boston$medv). Hint: You can approximate a 95 % confidence interval using the formula \[μˆ − 2SE(μˆ), μˆ + 2SE(μˆ)\].

``` r
CI.mu.hat <- c(22.53 - 2 * 0.4119, 22.53 + 2 * 0.4119)

CI.mu.hat
```

    ## [1] 21.7062 23.3538

``` r
t.test(Boston$medv)
```

    ## 
    ##  One Sample t-test
    ## 
    ## data:  Boston$medv
    ## t = 55.111, df = 505, p-value < 2.2e-16
    ## alternative hypothesis: true mean is not equal to 0
    ## 95 percent confidence interval:
    ##  21.72953 23.33608
    ## sample estimates:
    ## mean of x 
    ##  22.53281

The bootstrap confidence interval is very close to the one provided by the t.test() function.

Basedonthisdataset,provideanestimate,μˆmed,forthemedian value of medv in the population.

``` r
median_value = median(Boston$medv)
```

Wenowwouldliketoestimatethestandarderrorofμˆmed.Unfor- tunately, there is no simple formula for computing the standard error of the median. Instead, estimate the standard error of the median using the bootstrap. Comment on your findings.

``` r
boot.sample.median = function(data, index){
  return(median(data[index]))
}

boot.sample.median(Boston$medv, sample(nrow(Boston), nrow(Boston), replace = T))
```

    ## [1] 22

``` r
set.seed(1)
boot(Boston$medv, boot.sample.median, 1000)
```

    ## 
    ## ORDINARY NONPARAMETRIC BOOTSTRAP
    ## 
    ## 
    ## Call:
    ## boot(data = Boston$medv, statistic = boot.sample.median, R = 1000)
    ## 
    ## 
    ## Bootstrap Statistics :
    ##     original   bias    std. error
    ## t1*     21.2 -0.01615   0.3801002

We get a standard error of 0.38 which is relatively small compared to median value.

Based on this data set, provide an estimate for the tenth percentile of medv in Boston suburbs. Call this quantity μˆ0.1.

``` r
tenth.percentile <- quantile(Boston$medv, c(0.1))
tenth.percentile
```

    ##   10% 
    ## 12.75

Use the bootstrap to estimate the standard error of μˆ0.1. Com- ment on your findings.

``` r
boot.tenth.perct = function(data, index){
  return(quantile(data[index], c(0.1)))
}

boot.tenth.perct(Boston$medv, sample(nrow(Boston), nrow(Boston), replace = T))
```

    ##   10% 
    ## 12.75

``` r
set.seed(1)
boot(Boston$medv, boot.tenth.perct, 1000)
```

    ## 
    ## ORDINARY NONPARAMETRIC BOOTSTRAP
    ## 
    ## 
    ## Call:
    ## boot(data = Boston$medv, statistic = boot.tenth.perct, R = 1000)
    ## 
    ## 
    ## Bootstrap Statistics :
    ##     original  bias    std. error
    ## t1*    12.75 0.01005    0.505056

We get standard error of 0.505 which is relatively small compared to percentile value.
