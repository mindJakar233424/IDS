# 1. Install Required Packages
# --------------------------
install.packages(c(
  "readr", "dplyr", "tm", "SnowballC","textstem",
  "tidytext", "factoextra", "stringr", "quanteda","ggplot2",
  "proxy", "data.table", "Matrix", "irlba","dbscan","cluster"
))


## ================== Libraries ==================
library(dplyr)
library(tm)
library(readr)
library(SnowballC)
library(textstem)
library(tidytext)
library(stringr)
library(factoextra)
library(quanteda)
library(ggplot2)
library(proxy)
library(data.table)
library(Matrix)
library(irlba)
library(dbscan)
library(cluster)

## ================== 1. Read data ==================
url <- "https://raw.githubusercontent.com/mindJakar233424/IDS/main/Final%20data.csv"
data <- read.csv(url, stringsAsFactors = FALSE)

head(data)
stopifnot("review_full" %in% names(data))

## ================== 2. Corpus + preprocessing ==================
corpus <- VCorpus(VectorSource(data$review_full))

# Preprocessing helpers
toLower_tr   <- content_transformer(function(x, ...) tolower(x))
rm_url_tr    <- content_transformer(function(x, ...) gsub("https?://\\S+|www\\.\\S+", " ", x))
lemmatize_tr <- content_transformer(function(x, ...) lemmatize_strings(x))
stem_tr      <- content_transformer(function(x, ...) SnowballC::wordStem(x, language = "en"))

# Apply preprocessing
corpus <- tm_map(corpus, rm_url_tr)
corpus <- tm_map(corpus, toLower_tr)
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords("en"))
corpus <- tm_map(corpus, stripWhitespace)
corpus <- tm_map(corpus, stem_tr)       # choose stemming
# corpus <- tm_map(corpus, lemmatize_tr) # or lemmatization instead of stemming

# Back to text
data$cleaned_review <- sapply(corpus, as.character, USE.NAMES = FALSE)
head(data$cleaned_review, 10)

## ================== 3. Tokenization ==================
data$tokens <- lapply(
  data$cleaned_review,
  function(txt) {
    if (is.na(txt) || !nzchar(txt)) return(character(0))
    unlist(strsplit(trimws(txt), "\\s+"), use.names = FALSE)
  }
)
head(data$tokens)

## ================== 4. TF–IDF (compact, no DTM) ==================
N <- nrow(data)

dt <- data.table(
  doc  = rep(seq_len(N), lengths(data$tokens)),
  term = tolower(unlist(data$tokens, use.names = FALSE))
)[nzchar(term)]

tf <- dt[, .N, by = .(doc, term)]; setnames(tf, "N", "tf")

df_tbl <- unique(tf[, .(doc, term)])[, .N, by = term]; setnames(df_tbl, "N", "df")
df_tbl[, idf := log((N + 1) / (df + 1)) + 1]

tfidf <- merge(tf, df_tbl[, .(term, idf)], by = "term")[, tfidf := tf * idf]
terms <- data.table(term = sort(unique(tfidf$term)))[, j := .I]
tfidf <- merge(tfidf, terms, by = "term")

Xsp <- sparseMatrix(
  i    = as.integer(tfidf$doc),
  j    = as.integer(tfidf$j),
  x    = tfidf$tfidf,
  dims = c(N, nrow(terms))
)

## ================== 5. Drop empties + SVD (X) ==================
nz <- which(Matrix::rowSums(Xsp) > 0)
Xin <- Xsp[nz, ]
rk  <- max(2, min(50, nrow(Xin) - 1, ncol(Xin) - 1))

set.seed(42)
svd_res <- irlba(Xin, nv = rk, nu = rk)
X <- svd_res$u %*% diag(svd_res$d)    # SVD-reduced representation

## ================== 6. K-means + elbow + silhouette ==================
ks <- 2:10
wss <- sapply(ks, function(k) {
  set.seed(123)
  kmeans(X, centers = k, nstart = 10, iter.max = 100)$tot.withinss
})
plot(ks, wss, type = "b", pch = 19, xlab = "K", ylab = "WSS", main = "Elbow")

print(data.frame(K = ks, WSS = wss))

K <- 3
set.seed(123)
km <- kmeans(X, centers = K, nstart = 20, iter.max = 200)

data$cluster <- { cl <- rep(NA_integer_, N); cl[nz] <- km$cluster; cl }
head(data.frame(row_id = seq_len(N), cluster = data$cluster), 10)

# Silhouette for K-means (on SVD space X)
sil_km <- silhouette(km$cluster, dist(X))
plot(sil_km, main = "Silhouette plot – K-means (SVD space)")
mean_sil_km <- mean(sil_km[, "sil_width"])
mean_sil_km

## ================== 7. Hierarchical clustering + silhouette ==================
K_hc <- K
hc <- hclust(dist(X), method = "ward.D2")

plot(hc, labels = FALSE, hang = -1,
     main = paste("Hierarchical Dendrogram (Ward.D2) | K =", K_hc),
     xlab = "Documents (non-empty)", ylab = "Height")
rect.hclust(hc, k = K_hc, border = "gray40")

lab <- cutree(hc, k = K_hc)
data$hclust_cluster <- { z <- rep(NA_integer_, N); z[nz] <- lab; z }

head(data.frame(row_id = seq_len(N), hclust_cluster = data$hclust_cluster), 10)
cat("\nHierarchical cluster counts:\n")
print(addmargins(table(factor(data$hclust_cluster, levels = seq_len(K_hc)), useNA = "ifany")))

# Silhouette for hierarchical clustering
sil_hc <- silhouette(lab, dist(X))
plot(sil_hc, main = "Silhouette plot – Hierarchical (Ward.D2, SVD space)")
mean_sil_hc <- mean(sil_hc[, "sil_width"])
mean_sil_hc





## ================== 8. PCA + visual clustering (K-means, HC) ==================
if (!requireNamespace("factoextra", quietly = TRUE)) install.packages("factoextra")
if (!requireNamespace("cluster", quietly = TRUE)) install.packages("cluster")

library(factoextra)
library(cluster)

# PCA reduction
X_pca <- prcomp(scale(X), center = TRUE, scale. = TRUE)$x[, 1:2]

# K-means on PCA
set.seed(123)
k <- 3
km_fit <- kmeans(X_pca, centers = k, nstart = 25)

fviz_cluster(km_fit, data = X_pca,
             geom = "point",
             ellipse.type = "norm",
             main = "K-means Clustering (PCA reduced)")

# Hierarchical on PCA
hc_pca <- hclust(dist(X_pca), method = "ward.D2")
hc_clusters <- cutree(hc_pca, k = k)

fviz_cluster(list(data = X_pca, cluster = hc_clusters),
             geom = "point",
             ellipse.type = "norm",
             main = "Hierarchical Clustering (PCA reduced)")

fviz_dend(hc_pca, k = k, rect = TRUE, main = "Hierarchical Dendrogram (PCA)")

