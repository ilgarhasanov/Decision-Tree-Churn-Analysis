library(tidyverse)
library(rstudioapi)
library(skimr)

# Importing the dataset
path <- dirname(getSourceEditorContext()$path)
setwd(path)

df <- read_csv('Churn_Modelling (1).csv')

df %>% skim()

df <- df[,4:14]

df.num <- df %>%select_if(is.numeric) 

df.num$Exited <- df.num$Exited %>% as.factor()

df.chr <- df %>% mutate_if(is.character,as.factor) %>% 
  select_if(is.factor) 

df <- cbind(df.chr, df.num)



# Splitting the df into the Train set and Test set
library(caTools)
set.seed(123)
split <- df$Exited %>% sample.split(SplitRatio = 0.8)
train <- df %>% subset(split == TRUE)
test <- df %>% subset(split == FALSE)


# Fitting XGBoost model ----
library(xgboost)
library(parsnip)
set.seed(123)
clf <- boost_tree(mode = "classification", 
                         mtry = 30,
                         learn_rate = 0.35,  
                         tree_depth = 7) %>% 
  set_engine(engine = "xgboost") %>%
  fit(Exited ~ ., data = train)


# Predicting the Test set results
y_pred <- clf %>% predict(new_data = test %>% select(-Exited))


# Model evaluation ----
residuals = test$Exited - y_pred$.pred

RMSE = sqrt(mean(residuals^2))

y_test_mean = mean(test$Exited)

tss =  sum((test$Exited - y_test_mean)^2)
rss =  sum(residuals^2)

R2  =  1 - (rss/tss)

n <- test %>% nrow() 
k <- test %>% ncol() - 1
Adjusted_R2 = 1-(1-R2)*((n-1)/(n-k-1))


#Cross Validation
library(h2o)
h2o.init()

target = "Exited"
features = df %>% select (-Exited) %>% names() 

train_h2o <- train %>% select(target,features) %>% as.h2o()
test_h2o <- test %>% select(target,features) %>% as.h2o()

fit <- h2o.kmeans(training_frame = train_h2o,
                  k = 10,
                  x = features,
                  nfolds = 5,  
                  keep_cross_validation_predictions = TRUE)

# This is where list of cv preds are stored (one element per fold):
fit@model[["cross_validation_predictions"]]

# However you most likely want a single-column frame including all cv preds
cvpreds <- h2o.getFrame(fit@model[["cross_validation_holdout_predictions_frame_id"]][["name"]])

