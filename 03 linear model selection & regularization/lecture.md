Ridge Regression & Lasso
================
Amin Yakubu
2/13/2019

``` r
library(ISLR)
library(glmnet) # For lasso, ridge & elastic net
library(caret) # for train function
library(corrplot) # For correlation plot
library(plotmo) # for trace plot. glmnet can also produce plots but this one is better
```

Predict a baseball player’s salary on the basis of various statistics associated with performance in the previous year. Use `?Hitters` for more details.

``` r
data(Hitters)
# delete rows containing the missing data
Hitters <- na.omit(Hitters)

# matrix of predictors (glmnet uses input matrix). We specify the response and use . the specify all predictors
x <- model.matrix(Salary ~ . , Hitters)[,-1] # we delete the intercept (which is the first column)

# vector of response
y <- Hitters$Salary

corrplot(cor(x), method = 'square') # method is optional
```

![](lecture_files/figure-markdown_github/unnamed-chunk-2-1.png)

Ridge regression using `glmnet()`
---------------------------------

`alpha` is the elasticnet mixing parameter. The penalty on the coefficient vetor for predictor *j* is (1 − *α*)/2||*β*<sub>*j*</sub>||<sub>2</sub><sup>2</sup> + *α*||*β*<sub>*j*</sub>||<sub>1</sub>. `alpha=1` is the lasso penalty, and `alpha=0` the ridge penalty. `glmnet()` function standardizes the variables by default. `ridge.mod` contains the coefficient estimates for a set of lambda values. The grid for lambda is in `ridge.mod$lambda`.

``` r
# fit the ridge regression (alpha = 0) with a sequence of lambdas
ridge.mod <- glmnet(x, y, alpha = 0, lambda = exp(seq(-1, 10, length = 100))) # 100 you will have 100 solutions. 
# default here is standardized. 
# set alpha = 0 for ridge
# set alpha = 1 for lasso
# lambda is trying different values. Alternatively use nlambda = 100. It will choose the values automatically
# another function standardize = TRUE meaning you will center you predictors and outcome

ridge.mod$lambda
```

    ##   [1] 2.202647e+04 1.971015e+04 1.763742e+04 1.578265e+04 1.412294e+04
    ##   [6] 1.263776e+04 1.130876e+04 1.011953e+04 9.055351e+03 8.103084e+03
    ##  [11] 7.250958e+03 6.488442e+03 5.806113e+03 5.195539e+03 4.649172e+03
    ##  [16] 4.160262e+03 3.722766e+03 3.331277e+03 2.980958e+03 2.667478e+03
    ##  [21] 2.386965e+03 2.135950e+03 1.911332e+03 1.710335e+03 1.530475e+03
    ##  [26] 1.369529e+03 1.225508e+03 1.096633e+03 9.813105e+02 8.781152e+02
    ##  [31] 7.857720e+02 7.031397e+02 6.291970e+02 5.630302e+02 5.038216e+02
    ##  [36] 4.508394e+02 4.034288e+02 3.610039e+02 3.230405e+02 2.890694e+02
    ##  [41] 2.586706e+02 2.314687e+02 2.071272e+02 1.853456e+02 1.658545e+02
    ##  [46] 1.484132e+02 1.328059e+02 1.188400e+02 1.063427e+02 9.515961e+01
    ##  [51] 8.515256e+01 7.619786e+01 6.818484e+01 6.101447e+01 5.459815e+01
    ##  [56] 4.885657e+01 4.371878e+01 3.912128e+01 3.500726e+01 3.132588e+01
    ##  [61] 2.803162e+01 2.508380e+01 2.244597e+01 2.008554e+01 1.797333e+01
    ##  [66] 1.608324e+01 1.439192e+01 1.287845e+01 1.152415e+01 1.031226e+01
    ##  [71] 9.227814e+00 8.257411e+00 7.389056e+00 6.612018e+00 5.916694e+00
    ##  [76] 5.294490e+00 4.737718e+00 4.239496e+00 3.793668e+00 3.394723e+00
    ##  [81] 3.037732e+00 2.718282e+00 2.432425e+00 2.176630e+00 1.947734e+00
    ##  [86] 1.742909e+00 1.559623e+00 1.395612e+00 1.248849e+00 1.117519e+00
    ##  [91] 1.000000e+00 8.948393e-01 8.007374e-01 7.165313e-01 6.411804e-01
    ##  [96] 5.737534e-01 5.134171e-01 4.594258e-01 4.111123e-01 3.678794e-01

`coef(ridge.mod)` gives the coefficient matrix. Each column is the fit corresponding to one lambda value.

``` r
mat.coef <- coef(ridge.mod)
dim(mat.coef)
```

    ## [1]  20 100

### Cross-validation

We use cross-validation to determine the optimal value of lambda. The two vertical lines are the for minimal MSE and 1SE rule. The 1SE rule gives the model with fewest coefficients that's less than one SE away from the sub-model with the lowest error.

``` r
set.seed(2)
cv.ridge <- cv.glmnet(x, y, 
                      alpha = 0, 
                      lambda = exp(seq(-1, 10, length = 100)), 
                      type.measure = "mse")

# here alpha = 0, meaning ridge
# type.measure the metric to decide the best model. 

plot(cv.ridge)
```

![](lecture_files/figure-markdown_github/unnamed-chunk-5-1.png)

``` r
# the redline is the mse for each lambda value
# we also see the mse + SE and mse - SE
# the first dashed line shows the optimal value for lambda
# second dashed line is the 1 standared error rule - gives the model with the fewest coefficients. Another good model esp if you want few parameters. It depends on your own choice. They are both good.
```

### Trace plot

There are two functions for generating the trace plot.

``` r
plot(ridge.mod, xvar = "lambda", label = TRUE)
```

![](lecture_files/figure-markdown_github/unnamed-chunk-6-1.png)

``` r
plot_glmnet(ridge.mod, xvar = "rlambda", label = 19) # label = 10 is the defaul
```

![](lecture_files/figure-markdown_github/unnamed-chunk-6-2.png)

### Coefficients of the final model

Get the coefficients of the optimal model. `s` is value of the penalty parameter `lambda` at which predictions are required.

``` r
best.lambda <- cv.ridge$lambda.min # cv.ridge$labmda.1se gives you the other optimal value on the other side of the plot
best.lambda
```

    ## [1] 4.239496

``` r
predict(ridge.mod, s = best.lambda, type = "coefficients") # the coefficents are on the original scale -- not scaled
```

    ## 20 x 1 sparse Matrix of class "dgCMatrix"
    ##                         1
    ## (Intercept)  149.22633062
    ## AtBat         -1.60997670
    ## Hits           5.61921653
    ## HmRun          0.63498887
    ## Runs          -0.32293926
    ## RBI            0.06725632
    ## Walks          5.22312846
    ## Years        -10.42026714
    ## CAtBat        -0.05557923
    ## CHits          0.20316697
    ## CHmRun         0.72476255
    ## CRuns          0.69350591
    ## CRBI           0.36360275
    ## CWalks        -0.60569491
    ## LeagueN       61.37840710
    ## DivisionW   -122.78299563
    ## PutOuts        0.27944521
    ## Assists        0.28974207
    ## Errors        -3.76586590
    ## NewLeagueN   -27.92044299

``` r
# use type = 'response' for homework. and also use newx = to a matrix. So it will return a vector of predicted responses
```

Lasso using `glmnet()`
----------------------

The syntax is along the same line as ridge regression. Now we use `alpha = 1`.

``` r
cv.lasso <- cv.glmnet(x,y, alpha = 1, lambda = exp(seq(-1, 5, length = 100)))
cv.lasso$lambda.min
```

    ## [1] 2.888121

``` r
# alpha = 1 is lasso and alpha = 0 is ridge
```

``` r
plot(cv.lasso)
```

![](lecture_files/figure-markdown_github/unnamed-chunk-9-1.png)

``` r
# shows the number of non zero coefficents. Look at the numbers on top of the model
```

``` r
# cv.lasso$glmnet.fit is a fitted glmnet object for the full data
# You can also plot the result obtained from glmnet()
plot(cv.lasso$glmnet.fit, xvar = "lambda", label = TRUE)
```

![](lecture_files/figure-markdown_github/unnamed-chunk-10-1.png)

``` r
plot_glmnet(cv.lasso$glmnet.fit)
```

![](lecture_files/figure-markdown_github/unnamed-chunk-10-2.png)

``` r
predict(cv.lasso, s = "lambda.min", type = "coefficients")
```

    ## 20 x 1 sparse Matrix of class "dgCMatrix"
    ##                        1
    ## (Intercept)  116.5367637
    ## AtBat         -1.4948043
    ## Hits           5.5615474
    ## HmRun          .        
    ## Runs           .        
    ## RBI            .        
    ## Walks          4.6148464
    ## Years         -8.8197352
    ## CAtBat         .        
    ## CHits          .        
    ## CHmRun         0.5367614
    ## CRuns          0.6455687
    ## CRBI           0.3723109
    ## CWalks        -0.5058885
    ## LeagueN       32.2153588
    ## DivisionW   -119.0062145
    ## PutOuts        0.2714146
    ## Assists        0.1626281
    ## Errors        -1.9397895
    ## NewLeagueN     .

``` r
# you can also use the actual number for s
```

Ridge and lasso using `caret`
-----------------------------

``` r
ctrl1 <- trainControl(method = "repeatedcv", number = 10, repeats = 5)
# you can try other options

set.seed(2)
ridge.fit <- train(x, y,
                     method = "glmnet",
                     tuneGrid = expand.grid(alpha = 0, 
                                            lambda = exp(seq(-1, 10, length = 100))),
                   # preProc = c("center", "scale"),
                     trControl = ctrl1)
# preProc was not used because that's the default for glmnet. Other methods it may not be the default

plot(ridge.fit)
```

![](lecture_files/figure-markdown_github/unnamed-chunk-12-1.png)

``` r
plot(ridge.fit, xTrans = function(x) log(x)) # here were are plotting log lambda so it looks like the previous plots
```

![](lecture_files/figure-markdown_github/unnamed-chunk-12-2.png)

``` r
names(ridge.fit)
```

    ##  [1] "method"       "modelInfo"    "modelType"    "results"     
    ##  [5] "pred"         "bestTune"     "call"         "dots"        
    ##  [9] "metric"       "control"      "finalModel"   "preProcess"  
    ## [13] "trainingData" "resample"     "resampledCM"  "perfNames"   
    ## [17] "maximize"     "yLimits"      "times"        "levels"

``` r
ridge.fit$bestTune
```

    ##    alpha   lambda
    ## 38     0 22.44597

``` r
coef(ridge.fit$finalModel, ridge.fit$bestTune$lambda)
```

    ## 20 x 1 sparse Matrix of class "dgCMatrix"
    ##                         1
    ## (Intercept)  8.112693e+01
    ## AtBat       -6.815959e-01
    ## Hits         2.772312e+00
    ## HmRun       -1.365680e+00
    ## Runs         1.014826e+00
    ## RBI          7.130225e-01
    ## Walks        3.378558e+00
    ## Years       -9.066800e+00
    ## CAtBat      -1.199478e-03
    ## CHits        1.361029e-01
    ## CHmRun       6.979958e-01
    ## CRuns        2.958896e-01
    ## CRBI         2.570711e-01
    ## CWalks      -2.789666e-01
    ## LeagueN      5.321272e+01
    ## DivisionW   -1.228345e+02
    ## PutOuts      2.638876e-01
    ## Assists      1.698796e-01
    ## Errors      -3.685645e+00
    ## NewLeagueN  -1.810510e+01

``` r
# you can use predict on the ridge.fit$finalModel. Because ridge.fit$finalModel is a glmnet object
```

``` r
set.seed(2)
lasso.fit <- train(x, y,
                     method = "glmnet",
                     tuneGrid = expand.grid(alpha = 1, 
                                            lambda = exp(seq(-1, 5, length = 100))),
                   # preProc = c("center", "scale"),
                     trControl = ctrl1)

plot(lasso.fit, xTrans = function(x) log(x))
```

![](lecture_files/figure-markdown_github/unnamed-chunk-13-1.png)

``` r
lasso.fit$bestTune
```

    ##    alpha   lambda
    ## 34     1 2.718282

``` r
coef(lasso.fit$finalModel,lasso.fit$bestTune$lambda)
```

    ## 20 x 1 sparse Matrix of class "dgCMatrix"
    ##                        1
    ## (Intercept)  122.7036652
    ## AtBat         -1.5350416
    ## Hits           5.6337077
    ## HmRun          .        
    ## Runs           .        
    ## RBI            .        
    ## Walks          4.7077121
    ## Years         -9.5278084
    ## CAtBat         .        
    ## CHits          .        
    ## CHmRun         0.5057445
    ## CRuns          0.6554434
    ## CRBI           0.3932329
    ## CWalks        -0.5241348
    ## LeagueN       31.9907784
    ## DivisionW   -119.2910396
    ## PutOuts        0.2720718
    ## Assists        0.1708952
    ## Errors        -2.0389969
    ## NewLeagueN     .

``` r
set.seed(2)
enet.fit <- train(x, y,
                     method = "glmnet",
                     tuneGrid = expand.grid(alpha = seq(0, 1, length = 5), 
                                            # We are seeing if alpha between 0 and 1 is better
                                            lambda = exp(seq(-2, 4, length = 50))),
                   # preProc = c("center", "scale"),
                     trControl = ctrl1)

# for each alpha we have 50 lambdas. 
enet.fit$bestTune
```

    ##     alpha   lambda
    ## 225     1 2.556849

``` r
ggplot(enet.fit)
```

![](lecture_files/figure-markdown_github/unnamed-chunk-14-1.png)

``` r
set.seed(2)
lm.fit <- train(x, y,
                method = "lm",
                trControl = ctrl1)

resamp <- resamples(list(lasso = lasso.fit, ridge = ridge.fit, lm = lm.fit, enet = enet.fit)) # here we can compare the models
summary(resamp)
```

    ## 
    ## Call:
    ## summary.resamples(object = resamp)
    ## 
    ## Models: lasso, ridge, lm, enet 
    ## Number of resamples: 50 
    ## 
    ## MAE 
    ##           Min.  1st Qu.   Median     Mean  3rd Qu.     Max. NA's
    ## lasso 167.8420 205.6732 236.3658 233.6655 254.1476 331.5538    0
    ## ridge 160.1742 209.1149 229.7622 233.5423 258.4432 328.8789    0
    ## lm    179.7526 219.1404 237.5740 239.4876 261.3989 331.6590    0
    ## enet  168.6762 206.3211 236.2156 233.8372 254.5370 330.9309    0
    ## 
    ## RMSE 
    ##           Min.  1st Qu.   Median     Mean  3rd Qu.     Max. NA's
    ## lasso 208.7781 282.9478 311.3332 329.8454 348.8886 542.4902    0
    ## ridge 199.6564 281.2640 311.9637 330.7488 353.7262 542.2358    0
    ## lm    231.5831 286.7221 318.2538 337.1979 354.3965 594.9582    0
    ## enet  209.3123 282.2664 311.7742 329.8630 348.1674 543.5909    0
    ## 
    ## Rsquared 
    ##              Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
    ## lasso 0.005094987 0.3447779 0.5133295 0.4774190 0.6315335 0.7653965    0
    ## ridge 0.001370537 0.3430458 0.5154361 0.4753980 0.6292286 0.7998328    0
    ## lm    0.019695547 0.3275225 0.4922025 0.4591266 0.5924145 0.8122387    0
    ## enet  0.005565354 0.3442027 0.5134468 0.4773154 0.6287449 0.7635855    0

``` r
parallelplot(resamp, metric = "RMSE") # There are 50 curves showing the performances of the 4 models
```

![](lecture_files/figure-markdown_github/unnamed-chunk-15-1.png)

``` r
bwplot(resamp, metric = "RMSE") # here we used a box plot
```

![](lecture_files/figure-markdown_github/unnamed-chunk-15-2.png)
