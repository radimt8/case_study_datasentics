import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict


class BookDataProcessor:
    """Data processing utilities for book recommendation dataset"""
    
    def __init__(self, data_path: str = "./data"):
        self.data_path = data_path
        self.user_id_to_idx = {}
        self.book_title_to_idx = {}
        self.idx_to_user_id = {}
        self.idx_to_book_title = {}
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load the BookCrossing dataset"""
        users = pd.read_csv(f"{self.data_path}/Users.csv")
        ratings = pd.read_csv(f"{self.data_path}/Ratings.csv")
        books = pd.read_csv(f"{self.data_path}/Books.csv", low_memory=False)
        
        return users, ratings, books
    
    def preprocess_data(self, min_book_ratings: int = 50, min_user_ratings: int = 20) -> pd.DataFrame:
        """Preprocess and filter the data"""
        users, ratings, books = self.load_data()
        
        # Clean and merge data
        users_clean = users[['User-ID']]
        ratings_clean = ratings[['User-ID', 'ISBN', 'Book-Rating']]
        books_clean = books[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication']]
        
        # Merge ratings with books
        rating_books = ratings_clean.merge(books_clean, on='ISBN', how='inner')
        
        # Filter to explicit ratings only
        explicit_only = rating_books[rating_books['Book-Rating'] > 0]
        
        # Count ratings per book and user
        book_counts = explicit_only.groupby('Book-Title').size()
        user_counts = explicit_only.groupby('User-ID').size()
        
        # Apply filtering thresholds
        popular_books = book_counts[book_counts >= min_book_ratings].index
        active_users = user_counts[user_counts >= min_user_ratings].index
        
        # Filter dataset
        filtered_data = explicit_only[
            (explicit_only['Book-Title'].isin(popular_books)) &
            (explicit_only['User-ID'].isin(active_users))
        ]
        
        print(f"Filtered dataset: {filtered_data.shape[0]} ratings")
        print(f"Users: {filtered_data['User-ID'].nunique()}")
        print(f"Books: {filtered_data['Book-Title'].nunique()}")
        
        return filtered_data
    
    def create_rating_matrix(self, filtered_data: pd.DataFrame, 
                           test_size: float = 0.2, random_state: int = 42) -> Dict:
        """Create rating matrices and mappings"""
        
        # Create train/test split
        train_data, test_data = train_test_split(
            filtered_data, test_size=test_size, random_state=random_state
        )
        
        # Create user and book mappings
        unique_users = sorted(filtered_data['User-ID'].unique())
        unique_books = sorted(filtered_data['Book-Title'].unique())
        
        self.user_id_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
        self.book_title_to_idx = {title: idx for idx, title in enumerate(unique_books)}
        self.idx_to_user_id = {idx: uid for uid, idx in self.user_id_to_idx.items()}
        self.idx_to_book_title = {idx: title for title, idx in self.book_title_to_idx.items()}
        
        # Create training rating matrix
        R_train = train_data.pivot_table(
            index='User-ID', columns='Book-Title', values='Book-Rating'
        )
        
        # Ensure all users and books are represented
        R_train = R_train.reindex(index=unique_users, columns=unique_books)
        
        # Create masks and normalized matrices
        observed_mask_train = ~R_train.isna()
        R_filled_train = R_train.fillna(0)
        
        # Create test matrix aligned with training matrix
        test_pivot = test_data.pivot_table(
            index='User-ID', columns='Book-Title', values='Book-Rating'
        )
        test_aligned = test_pivot.reindex(index=R_train.index, columns=R_train.columns)
        observed_mask_test = ~test_aligned.isna()
        R_filled_test = test_aligned.fillna(0)
        
        # Convert to tensors and normalize
        R_tensor = torch.tensor(R_filled_train.values, dtype=torch.float32)
        mask_tensor = torch.tensor(observed_mask_train.values, dtype=torch.bool)
        R_normalized = (R_tensor - 1) / 9.0  # Normalize [1,10] to [0,1]
        
        # Test data
        R_test_tensor = torch.tensor(R_filled_test.values, dtype=torch.float32)
        mask_test_tensor = torch.tensor(observed_mask_test.values, dtype=torch.bool)
        R_test_normalized = (R_test_tensor - 1) / 9.0
        
        return {
            'R_normalized': R_normalized,
            'mask_tensor': mask_tensor,
            'R_test_normalized': R_test_normalized,
            'mask_test_tensor': mask_test_tensor,
            'train_data': train_data,
            'test_data': test_data,
            'book_titles': unique_books,
            'user_ids': unique_users,
            'n_users': len(unique_users),
            'n_items': len(unique_books)
        }
    
    def get_user_internal_idx(self, user_id: int) -> int:
        """Convert external user ID to internal matrix index"""
        return self.user_id_to_idx.get(user_id, -1)
    
    def get_external_user_id(self, internal_idx: int) -> int:
        """Convert internal matrix index to external user ID"""
        return self.idx_to_user_id.get(internal_idx, -1)
    
    def get_book_title_by_idx(self, internal_idx: int) -> str:
        """Get book title by internal index"""
        return self.idx_to_book_title.get(internal_idx, "Unknown")