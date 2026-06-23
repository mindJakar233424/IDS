
# Install needed packages once (uncomment if not installed)
# install.packages(c("tidyverse", "tidytext", "tm", "SnowballC",
#                    "wordcloud", "e1071", "caret", "textclean"))

library(tidyverse)
library(tidytext)
library(tm)
library(SnowballC)
library(wordcloud)
library(e1071)      # Naive Bayes
library(caret)      # Evaluation
library(textclean)

#=============================
# A. DATA COLLECTION
#=============================

data_url <- "https://raw.githubusercontent.com/mindJakar233424/IDS/refs/heads/main/cleaned_reviews.csv"
reviews_raw <- read_csv(data_url)

# View structure to identify text and label columns
glimpse(reviews_raw)

# --- IMPORTANT: set these to your actual column names ---
text_col_name  <- "cleaned_review"
label_col_name <- "sentiments"   # change if your label column has a different name
# ---------------------------------------------------------

# Keep only needed columns
reviews <- reviews_raw %>%
  select(cleaned_review, sentiments) %>%
  rename(
    text  = cleaned_review,
    label = sentiments
  ) %>%
  filter(!is.na(text), !is.na(label))

# Ensure at least 500 documents
nrow(reviews)

#=============================
# B. TEXT UNDERSTANDING & EXPLORATION
#=============================

# Tokenization for stats
reviews_tokens <- reviews %>%
  mutate(doc_id = row_number()) %>%
  unnest_tokens(word, text)

# Total documents
total_docs <- nrow(reviews)
total_docs

# Average text length (in tokens)
avg_length <- reviews_tokens %>%
  count(doc_id) %>%
  summarise(avg_tokens = mean(n)) %>%
  pull(avg_tokens)
avg_length

# Vocabulary size
vocab_size <- reviews_tokens %>%
  distinct(word) %>%
  nrow()
vocab_size

# Most frequent words
top_words <- reviews_tokens %>%
  count(word, sort = TRUE) %>%
  filter(n > 10)
head(top_words, 20)

# Most frequent bigrams
bigrams <- reviews %>%
  mutate(doc_id = row_number()) %>%
  unnest_tokens(bigram, text, token = "ngrams", n = 2) %>%
  count(bigram, sort = TRUE)
head(bigrams, 20)

# Word cloud (quick visualization)
set.seed(123)
wordcloud(words = top_words$word,
          freq  = top_words$n,
          max.words = 100,
          colors = brewer.pal(8, "Dark2"))

# Sentiment / label distribution
reviews %>%
  count(label) %>%
  ggplot(aes(x = label, y = n, fill = label)) +
  geom_col(show.legend = FALSE) +
  labs(title = "Class / Sentiment Distribution",
       x = "Label",
       y = "Count")

#=============================
# C. TEXT PREPROCESSING PIPELINE
#=============================

# Automatic text preprocessing
preprocess_text <- function(df, text_col = "text") {
  df %>%
    mutate(
      text_clean = .[[text_col]] %>%
        tolower() %>%         # lowercase
        replace_url() %>%     # remove URLs
        replace_tag() %>%     # remove @mentions
        replace_emoji() %>%   # normalize emojis
        replace_contraction() %>%
        replace_number() %>%  # remove numbers
        replace_non_ascii()
    )
}

reviews_clean <- preprocess_text(reviews, text_col = "text")

# Create corpus
corpus <- VCorpus(VectorSource(reviews_clean$text_clean))

# TM transformations
corpus <- corpus %>%
  tm_map(content_transformer(tolower)) %>%
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords("en")) %>%
  tm_map(stripWhitespace) %>%
  tm_map(stemDocument)

# DTM and TF-IDF
dtm <- DocumentTermMatrix(corpus)
dtm_tfidf <- weightTfIdf(dtm)

# Remove sparse terms for modeling
dtm_reduced <- removeSparseTerms(dtm_tfidf, 0.99)

# Convert to data frame
dtm_df <- as.data.frame(as.matrix(dtm_reduced))
dtm_df$label <- as.factor(reviews_clean$label)

#=============================
# D. MODELING: NAIVE BAYES CLASSIFIER
#=============================

set.seed(123)
train_index <- createDataPartition(dtm_df$label, p = 0.8, list = FALSE)

train_data <- dtm_df[train_index, ]
test_data  <- dtm_df[-train_index, ]

# Train Naive Bayes model
nb_model <- naiveBayes(label ~ ., data = train_data)

# Predict on test set
pred <- predict(nb_model, newdata = test_data)

#=============================
# E. EVALUATION & INTERPRETATION
#=============================

# Confusion matrix, Accuracy, F1-score
conf_mat <- confusionMatrix(pred, test_data$label)
conf_mat

# Accuracy
conf_mat$overall["Accuracy"]

# F1-score (per class)
f1_scores <- conf_mat$byClass[, "F1"]
f1_scores

# Optional: inspect class-conditional probabilities for first few features
nb_model$tables[1:5]
