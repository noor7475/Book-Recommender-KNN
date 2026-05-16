# Book Recommendation System – KNN Classifier

## Overview

This project implements a **K-Nearest Neighbors (KNN)** algorithm to build a book recommendation engine. By treating books as vectors in a multi-dimensional space, the model identifies and suggests titles that are mathematically similar based on user ratings, book length, and metadata.

The notebook covers a comprehensive machine learning pipeline:

* **Advanced Data Ingestion:** Handling `ParserWarning` issues and non-standard field counts.
* **Feature Transformation:** Implementing logarithmic scaling to normalize skewed numerical data.
* **Targeted Data Cleaning:** Removing "ghost" entries (books with high ratings but zero raters).
* **High-Dimensional Modeling:** Utilizing the KNN algorithm for instance-based learning.
* **Exploratory Data Analysis (EDA):** Statistical profiling of book features and language distributions.

---

## Dataset

The dataset used is `books.csv`, consisting of **11,123 entries** and **12 features**. The data required significant cleaning to handle formatting errors and outliers.

### Key Features Used

* **`title`** – The identifier used to retrieve and display recommendations.
* **`authors`** – Encoded to capture creator-specific similarity.
* **`average_rating`** – The primary quantitative measure of book quality.
* **`language_code`** – Consolidated into major groups (e.g., merging `en-US`, `en-GB`, and `en-CA` into `eng`).
* **`num_pages`** – Used to identify similarity in book length.
* **`ratings_count`** – Log-transformed to handle extreme variance in popularity.
* **`text_reviews_count`** – Log-transformed to account for user engagement levels.

### Removed Features

* **`bookID`, `isbn`, `isbn13`, `publication_date`, `publisher**` – Dropped to focus the model on core content attributes rather than administrative metadata.

---

## Project Pipeline

### 1. Data Cleaning & Automated Logic

The raw data contained several structural issues that were resolved programmatically:

* **Handling Bad Lines:** Used `on_bad_lines='warn'` to automatically identify and skip corrupted rows where the field count was incorrect (e.g., lines 3350, 4704).
* **Length Filtering:** Removed books with fewer than 6 pages to eliminate pamphlets and non-book entries.
* **Deduplication:** Sort-ordered by `ratings_count` to keep only the most representative entry for duplicate titles.
* **Quality Control:** Filtered out "Perfect Score" outliers—books with an `average_rating` of 5.0 but 0 actual ratings.

---

### 2. Feature Engineering & Scaling

To ensure the KNN algorithm treats all features fairly, specific mathematical transformations were applied:

* **Logarithmic Scaling:** Applied `np.log1p` to `num_pages`, `ratings_count`, and `text_reviews_count`. This squashes extreme outliers (like "Twilight" with millions of ratings) so they don't overwhelm the model's distance calculations.
* **Language Consolidation:** Standardized various English dialects into a single category to reduce sparsity in the `language_code` feature.

---

### 3. Model Implementation: K-Nearest Neighbors

The system uses an **unsupervised KNN approach** to measure the "distance" between books in a feature space defined by rating, length, and engagement.

* **Metric: Cosine Similarity** – While Euclidean distance measures the straight-line gap between points, I chose Cosine Similarity to measure the angle between vectors. This ensures that a book's "popularity" (magnitude) doesn't overshadow its actual "content similarity."
* **The Logic:** When a book title is queried, the model calculates the distance to every other book. The "Nearest Neighbors" are the titles with the smallest mathematical gap from the input.
* **Handling Sparsity:** By encoding categorical features like authors, the model can recommend books that are contextually and statistically similar even when direct user overlap is low.

---

## Result Interpretation

The model successfully identifies clusters of similar books:

* **Popularity-Matched:** Suggests titles with similar levels of community engagement.
* **Length-Matched:** Groups "epic" novels separately from shorter reads.
* **Quality-Matched:** Ensures highly-rated books are paired with other critically acclaimed titles.

---

## Conclusion

This project demonstrates how to turn raw, messy CSV data into a functional recommendation engine. It highlights the importance of **data normalization** (Log Scaling) and **automated cleaning** in building a robust instance-based model.

---

## The Colab Link

The interactive version of this project is available via the **Colab link found inside the notebook** (`knn_book_recommender.ipynb`).
