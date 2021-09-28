
# This script contains the following parts to get a clear overview:

# Part 1: The basic code provided by Harvard University to build the algorithms on
# Part 2: Observation of the dataset (dimension, best movies, most rated movie, etc)
# Part 3: The recommendation system (the training procedure)
# Part 4: The final prediction on the validation set
# Part 5: In usage: Making predictions

# Part 1

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


# Part 2

# How many movies are in the edx dataset
length(unique(edx$movieId))

# How many users rated movies
length(unique(edx$userId))

# 10 best rated genres which are rated more than 100 times
genres <- edx %>% group_by(genres) %>% 
  filter(table(genres) > 100) %>%
  arrange(desc(rating))

genres$genres[1:10]

# Sample 20 random genres and plot them against the ratings
set.seed(11, sample.kind = "Rounding")

random_genres <- sample(genres$genres, 20)

ind<- edx %>% filter(genres %in% random_genres) %>%
  group_by(genres) %>%
  summarize(rating_mean = mean(rating))

genre_rating <- (unique(ind))

genre_rating %>% ggplot(aes(rating_mean, genres))+
  geom_point()

# 10 of the best rated movies
best_ratings <- edx %>% group_by(movieId) %>% 
  arrange(desc(rating))

head(best_ratings, 10)

# Most rated movie in the dataset
most_rated <- edx %>% group_by(movieId) %>%
  summarize(title = title, n = n()) %>% 
  arrange (desc(n))

most_rated$title[1]

# Histogram of distribution of user ratings to show the user effect
edx %>% group_by(userId) %>% 
  summarize(avg_rating = mean(rating)) %>% 
  filter(n() > 10) %>%
  ggplot(aes(avg_rating)) + 
  geom_histogram(bins = 30, color = "orange") +
  scale_y_log10() +
  ggtitle("Average rating of users who rated over 100 times")

# Overall rating average
mean(edx$rating)

# Heatmap of a random subset with 100 users and movies to see the missing values
edx_sample_ind<- sample(edx$userId,100)
edx_sample <- edx %>% filter(userId %in% edx_sample_ind)

user_movie_matrix <- as.matrix(dcast(edx_sample, userId~movieId, value.var = "rating", na.rm = FALSE)[,-1])
image(user_movie_matrix, main = "Heatmap of 100 users and 100 movies")


# Part 3

# Create a data partition of the edx set to have a train and a test set
set.seed(7, sample.kind = "Rounding")

test_index <- createDataPartition(edx$rating, times = 1, p = 0.8, list = FALSE)
train_set <- edx %>% slice(test_index)
test_set <- edx %>% slice(-test_index)

# Only keep users with over 20 ratings to avoid noisy estimates.
# This is only for training the algorithm, later the complete edx set 
# and the complete validation set will be used for testing.
train_set <- train_set %>% group_by(userId) %>% filter(table(userId) > 20)

# Make sure that all movies and users in the test set are also in the train set so that there'll be no NAs
test_set <- test_set %>% 
  semi_join(train_set, "userId") %>%
  semi_join(train_set, "movieId")

# Define the average of all ratings
rating_avg <- mean(train_set$rating)

# Define the movie to movie effect, the distance each movie has to the average of all movies
movie_effect <- train_set %>% group_by(movieId) %>%
  summarize(movie_effect = mean(rating - rating_avg))

# Define user to user effect, the distance each user rates from the average of all users
user_effect <- train_set %>% 
  left_join(movie_effect, by = "movieId") %>% 
  group_by(userId) %>% 
  summarize(user_effect = mean(rating - rating_avg - movie_effect))

# See how this effect looks like
head(user_effect)

# Predict the ratings on the test set
predictions <- test_set %>%
  left_join(movie_effect, by = "movieId") %>%
  left_join(user_effect, by = "userId") %>%
  mutate(predictions = rating_avg + movie_effect + user_effect)

# Look at the prediction table
head(predictions)

# Calculate the RMSE on the test set
RMSE(test_set$rating, predictions$predictions)

# Using regularization term lambda to shrink the estimates when movies have only few ratings
# and train the new regulated movie and user effects.
# Find the best lambda using cross-validation
lambdas <- seq(1, 6, 0.1)
rmses <- sapply(lambdas, function(lambda){
  rating_avg <- mean(train_set$rating)
  reg_movie_effect <- train_set %>%
    group_by(movieId) %>%
    summarize(reg_movie_effect = sum(rating - rating_avg)/(n()+lambda))
  reg_user_effect <- train_set %>% 
    left_join(reg_movie_effect, by="movieId") %>%
    group_by(userId) %>%
    summarize(reg_user_effect = sum(rating - reg_movie_effect - rating_avg)/(n()+lambda))
  predictions <- test_set %>% 
    left_join(reg_movie_effect, by = "movieId") %>%
    left_join(reg_user_effect, by = "userId") %>%
    mutate(predictions = rating_avg + reg_movie_effect + reg_user_effect) %>%
    .$predictions
  return(RMSE(predictions, test_set$rating))
})

RMSE <- min(rmses)
RMSE

lambda_index <- which.min(rmses)
lambda <- lambdas[lambda_index]
lambda


# Part 4

# Final training with the regulated algorithm on the complete edx set to get all users and 
# movies back
reg_movie_effect <- edx %>%
  group_by(movieId) %>%
  summarize(reg_movie_effect = sum(rating - rating_avg)/(n()+lambda))
reg_user_effect <- edx %>% 
  left_join(reg_movie_effect, by="movieId") %>%
  group_by(userId) %>%
  summarize(reg_user_effect = sum(rating - rating_avg - reg_movie_effect)/(n()+lambda))

# Left-join the users and movies of the validation set just to get the same users and movies 
# The ratings of the validation set are not used or even touched
final_predictions <- validation %>% 
  left_join(reg_movie_effect, by = "movieId") %>%
  left_join(reg_user_effect, by = "userId") %>%
  mutate(predictions = rating_avg + reg_movie_effect + reg_user_effect)

# Now the final-predictions table has the same length as the validation set
dim(final_predictions)
dim(validation)

# Final test on the validation set
RMSE(final_predictions$predictions, validation$rating)


# Part 5: In usage: Making predictions

# Picking user 212 randomly and see which 5 movies he liked most
user_212 <- validation %>% filter(userId == 212)
Top_user_212 <- user_212 %>% filter(rating == 5) %>% select(title)
Top5_user_212 <- Top_user_212[1:5]
Top5_user_212

# Show movies, the algorithm recommends for this user
Recommend_user_212 <- final_predictions %>% 
  filter(userId == 212) %>%
  anti_join(Top5_user_212, by = "title") %>% 
  arrange(desc(predictions)) %>%
  select(title)
head(Recommend_user_212)

# Picking user 1724 randomly and see which 5 movies he liked most
user_1724 <- validation %>% filter(userId == 1724)
Top_user_1724 <- user_1724 %>% filter(rating == 5) %>% select(title)
Top5_user_1724 <- Top_user_1724[1:5]
Top5_user_1724

# Show movies, the algorithm recommends for this user
Recommend_user_1724 <- final_predictions %>% 
  filter(userId == 1724) %>%
  anti_join(Top5_user_1724, by = "title") %>% 
  arrange(desc(predictions)) %>%
  select(title)
head(Recommend_user_1724)
