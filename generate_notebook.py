import json
import os

# Define the notebook structure
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.5"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

def add_markdown(content):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": content.splitlines(keepends=True)
    })

def add_code(content):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": content.splitlines(keepends=True)
    })

# --- Notebook Content ---

# 1. Problem Definition & Objective
add_markdown("""# Book Recommendation System

## 1. Problem Definition & Objective

### Selected Project Track
**Recommendation Systems**

### Problem Statement
In the digital age, the abundance of books available online can be overwhelming for readers. Finding a book that matches a user's specific taste is a challenge. Users need a personalized system to discover relevant books based on their reading history or the similarity to books they already like.

### Objective
To build a **Book Recommendation System** that helps users discover books they are likely to enjoy. The system will leverage a dataset of book ratings to provide:
1.  **Popularity-based Recommendations**: For new users or general browsing.
2.  **Item-based Collaborative Filtering**: For personalized recommendations based on book similarity.

### Real-world Relevance and Motivation
*   **E-commerce**: Platforms like Amazon use recommendations to drive sales and cross-selling.
*   **User Experience**: Personalized feeds increase user engagement and retention.
*   **Discovery**: Helps lesser-known authors get discovered if their work is similar to popular titles.
""")

# 2. Data Understanding & Preparation
add_markdown("""## 2. Data Understanding & Preparation

We will use the **Book-Crossing Dataset**, which comprises three files:
*   `BX-Users.csv`: User information (ID, Location, Age).
*   `BX-Books.csv`: Book information (ISBN, Title, Author, Year, Publisher, Image URLs).
*   `BX-Book-Ratings-Subset.csv`: User ratings for books.
""")

add_code("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load Datasets
# Note: encoding='latin-1' is often required for this dataset due to special characters.
# error_bad_lines=False (or on_bad_lines='skip' in newer pandas) helps skip malformed rows.

books = pd.read_csv('BX-Books.csv', sep=';', encoding='latin-1', on_bad_lines='skip')
users = pd.read_csv('BX-Users.csv', sep=';', encoding='latin-1', on_bad_lines='skip')
ratings = pd.read_csv('BX-Book-Ratings-Subset.csv', sep=';', encoding='latin-1', on_bad_lines='skip')

print("Books Shape:", books.shape)
print("Users Shape:", users.shape)
print("Ratings Shape:", ratings.shape)
""")

add_code("""
# Display first few rows
print("Books Head:")
display(books.head(2))

print("Ratings Head:")
display(ratings.head(2))
""")

add_markdown("""### Data Cleaning
1.  **Renaming Columns**: For consistency/ease of access.
2.  **Handling Missing Values**: Checking for nulls.
3.  **Data Typing**: Ensuring ISBNs and User-IDs are consistent.
""")

add_code("""
# Rename columns for easier access
books.columns = ['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL-S', 'Image-URL-M', 'Image-URL-L']
users.columns = ['User-ID', 'Location', 'Age']
ratings.columns = ['User-ID', 'ISBN', 'Book-Rating']

# Check for nulls
print("Missing values in Books:\\n", books.isnull().sum())
print("Missing values in Users:\\n", users.isnull().sum())
print("Missing values in Ratings:\\n", ratings.isnull().sum())
""")

add_code("""
# Dropping Image URLs we don't need for analysis (we might keep them for the app later, but for the model we focus on Title/User/Rating)
# Actually, let's keep them as they are useful for the UI.

# Merge Ratings with Books to get Titles
ratings_with_name = ratings.merge(books, on='ISBN')
print("Merged Shape:", ratings_with_name.shape)
display(ratings_with_name.head(2))
""")

add_markdown("""### Exploratory Data Analysis (EDA)
Let's look at the distribution of ratings.
""")

add_code("""
# Rating Distribution
plt.figure(figsize=(10,4))
ratings_with_name['Book-Rating'].value_counts().sort_index().plot(kind='bar')
plt.title('Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()
""")

# 3. Model / System Design
add_markdown("""## 3. Model / System Design

We will implement two approaches:

### A. Popularity Based Recommender System
**Logic**: rank books by high ratings and high vote counts.
*   **Formula**: Calculate average rating per book and total number of ratings per book.
*   **Threshold**: Only consider books with at least X ratings (e.g., 50) to ensure reliability.
*   **Result**: A static list of top-performing books.

### B. Collaborative Filtering (Item-based)
**Logic**: "Users who liked this book also liked..."
*   **Matrix**: Create a 2D matrix (User-ID vs Book-Title).
*   **Similarity**: Use **Cosine Similarity** to find distance between book vectors.
*   **Result**: Given a book, return the top N most similar books based on user rating patterns.
""")

# 4. Core Implementation
add_markdown("""## 4. Core Implementation

### A. Popularity Based Recommender
""")

add_code("""
# 1. Group by Book-Title and count ratings
num_rating_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_rating_df.rename(columns={'Book-Rating': 'num_ratings'}, inplace=True)

# 2. Group by Book-Title and average ratings
avg_rating_df = ratings_with_name.groupby('Book-Title').mean(numeric_only=True)['Book-Rating'].reset_index()
avg_rating_df.rename(columns={'Book-Rating': 'avg_rating'}, inplace=True)

# 3. Merge
popular_df = num_rating_df.merge(avg_rating_df, on='Book-Title')

# 4. Filter: Let's pick books with > 50 ratings (Since this is a subset, we might lower this threshold if needed, but let's try 50)
popular_df = popular_df[popular_df['num_ratings'] >= 50].sort_values('avg_rating', ascending=False).head(50)

# 5. Merge with books to get details (Author, Image)
# distinct books
popular_df = popular_df.merge(books, on='Book-Title').drop_duplicates('Book-Title')[['Book-Title', 'Book-Author', 'Image-URL-M', 'num_ratings', 'avg_rating']]

print(f"Found {len(popular_df)} popular books.")
display(popular_df.head())
""")

add_markdown("""### B. Collaborative Filtering Recommender
We want to recommend books based on a selected book. We will look at users who rated both books similarly.
""")

add_code("""
# 1. Filter data for "experienced" users to reduce noise (e.g., users who rated > 200 books)
# In the full dataset this is common. In this subset, we check the user activity.
x = ratings_with_name.groupby('User-ID').count()['Book-Rating'] > 20
padhe_likhe_users = x[x].index

filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(padhe_likhe_users)]

# 2. Filter for famous books (e.g., books with > 50 ratings)
y = filtered_rating.groupby('Book-Title').count()['Book-Rating'] >= 50
famous_books = y[y].index

final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]

print("Final Ratings Shape for Matrix:", final_ratings.shape)

# 3. Create Pivot Table
if not final_ratings.empty:
    pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
    pt.fillna(0, inplace=True)
    print("Pivot Table Shape:", pt.shape)
    display(pt.head())
else:
    print("Not enough data in subset to create a dense matrix with these thresholds. Lowering thresholds for demonstration if needed.")
""")

add_code("""
from sklearn.metrics.pairwise import cosine_similarity

if not final_ratings.empty:
    similarity_scores = cosine_similarity(pt)
    print("Similarity Matrix Shape:", similarity_scores.shape)
    
    def recommend(book_name):
        # fetch index
        try:
            index = np.where(pt.index == book_name)[0][0]
            similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:6]
            
            data = []
            for i in similar_items:
                item = []
                temp_df = books[books['Book-Title'] == pt.index[i[0]]]
                item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
                item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
                item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
                
                data.append(item)
            
            return data
        except IndexError:
            return "Book not found in the matrix."
            
    # Test
    print("Recommendations for a book (if exists in PT):")
    # Let's pick one from index
    if len(pt.index) > 0:
        test_book = pt.index[0]
        print(f"Testing with: {test_book}")
        print(recommend(test_book))
""")

# 5. Evaluation & Analysis
add_markdown("""## 5. Evaluation & Analysis

### Qualitative Evaluation
Since we used unsupervised learning (Collaborative Filtering), we evaluate by inspection.
*   **Popularity Model**: Does it return generally well-regarded books? (e.g., Harry Potter, To Kill a Mockingbird).
*   **Collaborative Model**: If we select "Harry Potter", do we get other fantasy books or sequels?

(Note: In a real-world scenario, we would use metrics like RMSE on a test set, or A/B testing).

### Insights
*   The sparsity of the matrix affects recommendation quality.
*   Popularity-based is a good "Safe" fallback when we don't know the user's history.
""")

# 6. Ethical Considerations & Responsible AI
add_markdown("""## 6. Ethical Considerations & Responsible AI

*   **Bias**: If the dataset is dominated by a specific demographic (e.g., age, location), recommendations will be biased towards their preferences.
*   **Filter Bubble**: Collaborative filtering tends to reinforce existing preferences, potentially limiting exposure to diverse genres/viewpoints.
*   **Privacy**: Using User-IDs and locations raises privacy concerns. We must ensure data is anonymized and used only for the stated purpose.
""")

# 7. Conclusion & Future Scope
add_markdown("""## 7. Conclusion & Future Scope

### Conclusion
We successfully built a hybrid approach:
1.  **Top 50 Books**: Solves the "Cold Start" problem for new users.
2.  **Recommender Engine**: Provides personalized suggestions for engaged users.

### Future Scope
*   **Hybrid Model**: Combine content-based (Author, Publisher) with Collaborative Filtering.
*   **Deployment**: This logic is deployed using a Streamlit Web Application.
*   **Feedback Loop**: Capture user clicks on recommendations to improve the model over time.
""")

# Write to file
with open('Book_Recommendation.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Notebook generated successfully.")
