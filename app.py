import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- Page Config ---
st.set_page_config(
    page_title="Book Recommender System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        color: white;
        background-color: #ff4b4b;
        border-radius: 10px;
    }
    .book-card {
        background-color: white;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        text-align: center;
        height: 100%;
    }
    .book-title {
        font-weight: bold;
        font-size: 14px;
        margin-top: 5px;
        color: #333;
    }
    .book-author {
        font-size: 12px;
        color: #666;
    }
    </style>
""", unsafe_allow_html=True)

# --- Data Loading (Cached) ---
@st.cache_data
def load_data():
    try:
        # Load datasets
        books_df = pd.read_csv('BX-Books.csv', sep=';', encoding='latin-1', on_bad_lines='skip')
        ratings_df = pd.read_csv('BX-Book-Ratings-Subset.csv', sep=';', encoding='latin-1', on_bad_lines='skip')
        users_df = pd.read_csv('BX-Users.csv', sep=';', encoding='latin-1', on_bad_lines='skip')
        
        # Rename columns
        books_df.columns = ['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL-S', 'Image-URL-M', 'Image-URL-L']
        users_df.columns = ['User-ID', 'Location', 'Age']
        ratings_df.columns = ['User-ID', 'ISBN', 'Book-Rating']
        
        # Merge
        ratings_with_name = ratings_df.merge(books_df, on='ISBN')
        
        return books_df, ratings_with_name
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

@st.cache_data
def get_popular_books(ratings_with_name):
    # Popularity Based
    num_rating_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
    num_rating_df.rename(columns={'Book-Rating': 'num_ratings'}, inplace=True)
    
    avg_rating_df = ratings_with_name.groupby('Book-Title').mean(numeric_only=True)['Book-Rating'].reset_index()
    avg_rating_df.rename(columns={'Book-Rating': 'avg_rating'}, inplace=True)
    
    popular_df = num_rating_df.merge(avg_rating_df, on='Book-Title')
    
    # Filter: Top 50 books with highest average rating (min 50 ratings) based on original logic
    # Adjusting threshold for subset if needed - let's stick to 50 for now, or 10 if list is empty
    filtered_popular = popular_df[popular_df['num_ratings'] >= 50]
    
    if len(filtered_popular) == 0:
        # Fallback for small subset
        filtered_popular = popular_df[popular_df['num_ratings'] >= 5]
        
    popular_df = filtered_popular.sort_values('avg_rating', ascending=False).head(50)
    
    # Merge with book details
    # We need to act carefully to not duplicate entries too much
    # Get one image per book
    books_images = ratings_with_name[['Book-Title', 'Book-Author', 'Image-URL-M']].drop_duplicates('Book-Title')
    popular_df = popular_df.merge(books_images, on='Book-Title')
    
    return popular_df

@st.cache_data
def get_similarity_matrix(ratings_with_name):
    # Collaborative Filtering Logic
    # 1. Filter experienced users
    x = ratings_with_name.groupby('User-ID').count()['Book-Rating'] > 20
    padhe_likhe_users = x[x].index
    filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(padhe_likhe_users)]
    
    # 2. Filter famous books
    y = filtered_rating.groupby('Book-Title').count()['Book-Rating'] >= 50
    famous_books = y[y].index
    final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]
    
    # If not enough data, relax constraints for the sake of the assignment demo
    if final_ratings.empty:
        # Fallback: Relaxed constraints
        x = ratings_with_name.groupby('User-ID').count()['Book-Rating'] > 5
        padhe_likhe_users = x[x].index
        filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(padhe_likhe_users)]
        
        y = filtered_rating.groupby('Book-Title').count()['Book-Rating'] >= 5
        famous_books = y[y].index
        final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]

    pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
    pt.fillna(0, inplace=True)
    
    similarity_scores = cosine_similarity(pt)
    return pt, similarity_scores, final_ratings

# --- Load Data & Models ---
books, ratings_with_name = load_data()

if books is not None:
    popular_df = get_popular_books(ratings_with_name)
    pt, similarity_scores, final_ratings = get_similarity_matrix(ratings_with_name)

    # --- Sidebar ---
    st.sidebar.title("Book Recommender 📚")
    st.sidebar.markdown("Discover your next favorite read!")
    page = st.sidebar.radio("Navigation", ["Top 50 Books", "Recommend Books", "About"])

    # --- Page: Top 50 Books ---
    if page == "Top 50 Books":
        st.title("🏆 Top 50 Popular Books")
        st.markdown("Here are the highest-rated books in our collection.")
        
        # Display in a grid
        cols = st.columns(4) # 4 columns grid
        
        for i, row in enumerate(popular_df.iterrows()):
            data = row[1]
            col_idx = i % 4
            with cols[col_idx]:
                with st.container():
                    st.image(data['Image-URL-M'], width=130)
                    st.markdown(f"<p class='book-title'>{data['Book-Title']}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p class='book-author'>{data['Book-Author']}</p>", unsafe_allow_html=True)
                    st.caption(f"Votes: {data['num_ratings']} | Rating: {data['avg_rating']:.1f}")
                    st.markdown("---")

    # --- Page: Recommend Books ---
    elif page == "Recommend Books":
        st.title("🔍 Recommend Books")
        st.markdown("Select a book you like, and we'll suggest similar ones.")
        
        book_list = list(pt.index)
        selected_book = st.selectbox("Type or select a book from the dropdown", book_list)
        
        if st.button("Show Recommendations"):
            if selected_book:
                try:
                    index = np.where(pt.index == selected_book)[0][0]
                    similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:6] # Top 5
                    
                    st.subheader(f"Because you liked '{selected_book}':")
                    
                    cols = st.columns(5)
                    for i, item in enumerate(similar_items):
                        book_title = pt.index[item[0]]
                        
                        # Fetch details
                        temp_df = books[books['Book-Title'] == book_title].drop_duplicates('Book-Title')
                        image_url = temp_df['Image-URL-M'].values[0]
                        author = temp_df['Book-Author'].values[0]
                        
                        with cols[i]:
                            st.image(image_url, width=130)
                            st.markdown(f"<p class='book-title'>{book_title}</p>", unsafe_allow_html=True)
                            st.markdown(f"<p class='book-author'>{author}</p>", unsafe_allow_html=True)

                except Exception as e:
                    st.error("Could not find recommendations for this book. It might be an outlier.")
                    st.error(e)

    # --- Page: About ---
    elif page == "About":
        st.title("ℹ️ About")
        st.markdown("""
        This is a **Book Recommendation System** built with Streamlit and Python.
        
        **Techniques Used:**
        *   **Popularity Based Filtering**: Ranks books based on average ratings and vote counts.
        *   **Collaborative Filtering**: Suggests books based on user reading patterns (Cosine Similarity).
        
        **Dataset:** Book-Crossing Dataset.
        """)
else:
    st.error("Failed to load data. Please check the CSV files.")
