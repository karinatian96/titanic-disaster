suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
})

data_dir <- Sys.getenv("DATA_DIR", "/app/data")

cat("=== Step 13: Loading train.csv ===\n")
train <- read_csv(file.path(data_dir, "train.csv"), show_col_types = FALSE)
cat("Train dims:", dim(train), "\n")
print(head(train))

cat("\n=== Step 14: Cleaning data ===\n")
print(colSums(is.na(train)))
age_med  <- median(train$Age, na.rm = TRUE)
fare_med <- median(train$Fare, na.rm = TRUE)
emb_mode <- names(sort(table(train$Embarked), decreasing = TRUE))[1]
cat("Filled Age NA with median =", age_med, "\n")
cat("Filled Fare NA with median =", fare_med, "\n")
cat("Filled Embarked NA with mode =", emb_mode, "\n")
cat("Converted Sex and Embarked to factors\n")
cat("Added FamilySize = SibSp + Parch + 1\n")
cat("Excluded Cabin, Name, Ticket from model features (too many missing / not numeric)\n")

prepare <- function(df) {
  df %>%
    mutate(
      Age = ifelse(is.na(Age), age_med, Age),
      Fare = ifelse(is.na(Fare), fare_med, Fare),
      Embarked = ifelse(is.na(Embarked) | Embarked == "", emb_mode, Embarked),
      Sex = factor(Sex, levels = c("male", "female")),
      Embarked = factor(Embarked, levels = c("C", "Q", "S")),
      FamilySize = SibSp + Parch + 1
    )
}
train_p <- prepare(train)

cat("\n=== Step 15: Training logistic regression ===\n")
set.seed(42)
idx <- sample(seq_len(nrow(train_p)), size = floor(0.8 * nrow(train_p)))
tr  <- train_p[idx, ]
val <- train_p[-idx, ]
model <- glm(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare +
               FamilySize + Embarked,
             data = tr, family = binomial)
print(summary(model))

cat("\n=== Step 16: Training accuracy ===\n")
tr_pred <- ifelse(predict(model, tr, type = "response") > 0.5, 1, 0)
cat("Training accuracy:", round(mean(tr_pred == tr$Survived), 4), "\n")

cat("\n=== Step 17: Predicting on test.csv ===\n")
test <- read_csv(file.path(data_dir, "test.csv"), show_col_types = FALSE)
cat("Test dims:", dim(test), "\n")
test_p <- prepare(test)
test_pred <- ifelse(predict(model, test_p, type = "response") > 0.5, 1, 0)
cat("Predicted survivors:", sum(test_pred), "/", length(test_pred), "\n")
print(head(data.frame(PassengerId = test$PassengerId, Survived = test_pred), 10))

cat("\n=== Step 18: Test accuracy ===\n")
if ("Survived" %in% names(test)) {
  cat("Test accuracy:", round(mean(test_pred == test$Survived), 4), "\n")
} else {
  val_pred <- ifelse(predict(model, val, type = "response") > 0.5, 1, 0)
  cat("test.csv has no 'Survived' labels, so reporting accuracy on a",
      "20% hold-out split of train.csv instead.\n")
  cat("Hold-out accuracy:", round(mean(val_pred == val$Survived), 4), "\n")
}
cat("\nDone!\n")
