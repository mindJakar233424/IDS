
# install.packages(c("tidyverse", "tidytext", "tm", "topicmodels",
#                    "SnowballC", "wordcloud", "textmineR", "broom","irlba"))

library(tidyverse)
library(tidytext)
library(tm)
library(topicmodels)
library(SnowballC)
library(wordcloud)
library(irlba)
library(textmineR)
library(broom)      # tidy()
library(ggplot2)

#=============================
# A. DATA COLLECTION
#=============================

data_url <- "https://raw.githubusercontent.com/mindJakar233424/IDS/refs/heads/main/cleaned_reviews.csv"
reviews_raw <- read_csv(data_url)

glimpse(reviews_raw)
# Columns: sentiments (label), cleaned_review (text), cleaned_review_length, review_score

# For topic modeling, we only need the text
docs <- reviews_raw %>%
  select(cleaned_review) %>%
  rename(text = cleaned_review) %>%
  filter(!is.na(text))

nrow(docs)   # should be >= 500

#=============================
# B. TEXT UNDERSTANDING & EXPLORATION
#=============================

tokens <- docs %>%
  mutate(doc_id = row_number()) %>%
  unnest_tokens(word, text)

# Total documents
total_docs <- nrow(docs)
total_docs

# Average text length (in tokens)
avg_len <- tokens %>%
  count(doc_id) %>%
  summarise(avg_tokens = mean(n)) %>%
  pull(avg_tokens)
avg_len

# Vocabulary size
vocab_size <- tokens %>%
  distinct(word) %>%
  nrow()
vocab_size

# Most frequent words
top_words <- tokens %>%
  count(word, sort = TRUE) %>%
  filter(n > 50)   # adjust threshold if needed
head(top_words, 20)

# Most frequent bigrams
bigrams <- docs %>%
  mutate(doc_id = row_number()) %>%
  unnest_tokens(bigram, text, token = "ngrams", n = 2) %>%
  count(bigram, sort = TRUE)
head(bigrams, 20)

# Word cloud visualization
set.seed(123)
wordcloud(words = top_words$word,
          freq  = top_words$n,
          max.words = 100,
          colors = brewer.pal(8, "Dark2"))

#=============================
# C. TEXT PREPROCESSING & DTM
#=============================

# Automatic text preprocessing (lowercase, remove URLs, etc.)
preprocess_text <- function(df, text_col = "text") {
  df %>%
    mutate(
      text_clean = .[[text_col]] %>%
        tolower() %>%
        gsub("https?://\\S+|www\\.\\S+", " ", .) %>%  # remove URLs
        gsub("[0-9]+", " ", .) %>%                    # remove digits
        gsub("[[:punct:]]", " ", .) %>%              # remove punctuation
        gsub("[\\s]+", " ", .)                       # extra spaces
    )
}

docs_clean <- preprocess_text(docs, text_col = "text")

# Create corpus
corpus <- VCorpus(VectorSource(docs_clean$text_clean))

corpus <- corpus %>%
  tm_map(content_transformer(tolower)) %>%
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords("en")) %>%
  tm_map(stripWhitespace) %>%
  tm_map(stemDocument)

# Document-Term Matrix (term frequency for LDA)
dtm <- DocumentTermMatrix(corpus)

# Remove very sparse terms
dtm <- removeSparseTerms(dtm, 0.99)

# Remove empty documents
row_sums <- slam::row_sums(dtm)
dtm <- dtm[row_sums > 0, ]

dtm

#=============================
# D. TOPIC MODELING WITH LDA
#=============================

set.seed(123)


# For assignment, select a fixed k (e.g., 4 or 5)
k <- 5

lda_model <- LDA(dtm, k = k, control = list(seed = 1234))
lda_model

#=============================
# E. EVALUATION & INTERPRETATION
#=============================

# 1) Extract word-topic probabilities (beta)
topics_terms <- tidy(lda_model, matrix = "beta")   # term probabilities per topic
head(topics_terms)

# 2) Top words per topic visualization
top_terms <- topics_terms %>%
  group_by(topic) %>%
  slice_max(beta, n = 10) %>%
  ungroup() %>%
  arrange(topic, desc(beta))

top_terms

# Visualize top words per topic
top_terms %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip() +
  scale_x_reordered() +
  labs(title = "Top Words per Topic", x = NULL, y = "Beta (probability)")

# 3) Topic distribution per document (gamma)
doc_topics <- tidy(lda_model, matrix = "gamma")   # topic probabilities per document
head(doc_topics)

# 4) Simple topic coherence-like score (optional, for reporting)

coherence_inspect <- top_terms %>%
  group_by(topic) %>%
  summarise(
    top_terms = paste(term, collapse = ", ")
  )
coherence_inspect
