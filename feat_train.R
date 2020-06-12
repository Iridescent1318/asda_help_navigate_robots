library(e1071)
library(party)
library(randomForest)
library(mlbench)
library(caret)
library(caretEnsemble)

TRAIN_SOURCE = 'x_train_feats.csv'
x_train = read.csv(TRAIN_SOURCE)
y_train = read.csv('y_train.csv')
names(x_train)[1] <- 'series_id'

NO_PCA = 1
MANUAL_TRAIN = 0

if(NO_PCA == 0){
  x.pca <- princomp(x_train[2:length(x_train)])
  summary(x.pca, loadings=TRUE)
}

if(MANUAL_TRAIN){
  x_train_trans <- cbind(x_train[1], x.pca$scores)
  y <- factor(y_train$surface)
  
  N <- length(x_train$series_id)
  cv_train_percent <- 0.9
  cv_times <- 10
  acc_list_ctree <- array(1:cv_times)/10
  acc_list_svm <- array(1:cv_times)/10
  acc_list_rf <- array(1:cv_times)/10
  
  for(i in 1:cv_times){
    idx_shuffle <- sample(1:N, N, replace=FALSE)
    train_end_idx <- round(N*cv_train_percent)
    
    if(NO_PCA){
      x_cv_train <- x_train[idx_shuffle[1:train_end_idx],]
      x_cv_test <- x_train[idx_shuffle[(train_end_idx+1):N],]
    }else{
      x_cv_train <- x_train_trans[idx_shuffle[1:train_end_idx],]
      x_cv_test <- x_train_trans[idx_shuffle[(train_end_idx+1):N],]
    }
    
    y_cv_train <- factor(y[x_cv_train$series_id+1], )
    y_cv_test <- factor(y[x_cv_test$series_id+1], )
    
    cv_train_merge <- cbind(x_cv_train[2:length(x_train)], y_cv_train)
    
    s = proc.time()
    robot.ctree <- ctree(y_cv_train~., data = cv_train_merge)
    y_cv_pred_ctree <- predict(robot.ctree, x_cv_test)
    acc_ctree <- sum(y_cv_pred_ctree == y_cv_test) / length(y_cv_test)
    acc_list_ctree[i] <- acc_ctree
    t = proc.time() - s
    print(sprintf("No.: %d, ctree time used: %.4f", i, t[3]))
    
    s = proc.time()
    robot.svm <- svm(y_cv_train~., data = cv_train_merge)
    y_cv_pred_svm <- predict(robot.svm, x_cv_test)
    acc_svm <- sum(y_cv_pred_svm == y_cv_test) / length(y_cv_test)
    acc_list_svm[i] <- acc_svm
    t = proc.time() - s
    print(sprintf("No.: %d, svm time used: %.4f", i, t[3]))
    
    s = proc.time()
    robot.rf <- randomForest(y_cv_train~., data=cv_train_merge, ntree = 500)
    y_cv_pred_rf <- predict(robot.rf, x_cv_test)
    acc_rf <- sum(y_cv_pred_rf == y_cv_test) / length(y_cv_test)
    acc_list_rf[i] <- acc_rf
    t = proc.time() - s
    print(sprintf("No.: %d, rf time used: %.4f", i, t[3]))
    
    print(sprintf("No.: %d CTree accuracy: %.4f  SVM accuracy: %.4f RF accuracy: %.4f", i, acc_list_ctree[i], acc_list_svm[i], acc_list_rf[i]))
  }
  
  print(sprintf("AVERAGE CTree accuracy: %.4f  SVM accuracy: %.4f RF accuracy: %.4f", mean(acc_list_ctree), mean(acc_list_svm), mean(acc_list_rf)))
  
}else{
  y <- factor(y_train$surface)
  sprintf("Test")
  train_merge = cbind(x_train[2:length(x_train)], y)
  t_control <- trainControl(method="repeatedcv", number = 10, repeats = 3)
  t_metric = "Accuracy"
  robot.c50 <- train(y~., data=train_merge, method="C5.0", metric=t_metric, trControl = t_control)
  summary(robot.c50)
  
  # data(Ionosphere)
  # dataset <- Ionosphere
  # dataset <- dataset[,-2]
  # dataset$V1 <- as.numeric(as.character(dataset$V1))
  # control <- trainControl(method="repeatedcv", number=10, repeats=3)
  # seed <- 7
  # metric <- "Accuracy"
  # # C5.0
  # set.seed(seed)
  # fit.c50 <- train(Class~., data=dataset, method="C5.0", metric=metric, trControl=control)
  # # Stochastic Gradient Boosting
  # set.seed(seed)
  # fit.gbm <- train(Class~., data=dataset, method="gbm", metric=metric, trControl=control, verbose=FALSE)
  # # summarize results
  # boosting_results <- resamples(list(c5.0=fit.c50, gbm=fit.gbm))
  # summary(boosting_results)
  # dotplot(boosting_results)
}
