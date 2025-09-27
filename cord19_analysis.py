import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
from datetime import datetime 
import os

import warnings
warnings.filterwarnings('ignore')

# loading data
def load_data(file_path = 'data/CORD19_datasets.csv'):
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path} with shape {df.shape[0]} rows and {df.shape[1]} columns.")
        return df

    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None
    
# basic data overview
def data_overview(df):
    print("=" * 50)
    print("BASIC DATASET EXPLORATION")
    print("=" * 50)

# display information about the dataset
    print("\n1. Dataset Shape:")
    print(f"   Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    
    print("\n2. Column Names and Data Types:")
    print(df.dtypes)
    
    print("\n3. First 5 rows:")
    print(df.head())
    
    print("\n4. Missing Values Summary:")
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing_data,
        'Missing Percentage': missing_percent
    })
    print(missing_df.sort_values('Missing Count', ascending=False).head(10))
    
    print("\n5. Basic Statistics for Numerical Columns:")
    print(df.describe())
    
    return missing_df

def clean_data(df):
    """
    Clean and prepare the dataset for analysis
    """
    print("=" * 50)
    print("DATA CLEANING AND PREPARATION")
    print("=" * 50)
    
    # Create a copy of the dataframe
    df_clean = df.copy()
    
    # Handle publication date (using 'last_updated' instead of 'publish_time')
    print("\n1. Handling last updated dates...")
    df_clean['last_updated'] = pd.to_datetime(df_clean['last_updated'], errors='coerce')
    df_clean['publication_year'] = df_clean['last_updated'].dt.year
    
    # Fill missing years with mode or specific value
    mode_year = df_clean['publication_year'].mode()[0] if not df_clean['publication_year'].mode().empty else 2020
    df_clean['publication_year'].fillna(mode_year, inplace=True)
    df_clean['publication_year'] = df_clean['publication_year'].astype(int)
    
    # Handle missing descriptions (instead of abstracts)
    print("\n2. Handling missing descriptions...")
    df_clean['description'] = df_clean['description'].fillna('No description available')
    
    # Handle missing titles (use 'paper_title')
    print("\n3. Handling missing titles...")
    df_clean = df_clean.dropna(subset=['paper_title'])
    
    # Create description word count
    print("\n4. Creating new features...")
    df_clean['description_word_count'] = df_clean['description'].apply(lambda x: len(str(x).split()))
    df_clean['has_description'] = df_clean['description'] != 'No description available'
    
    # Clean dataset_name
    df_clean['dataset_name'] = df_clean['dataset_name'].fillna('Unknown Dataset')
    
    print(f"Cleaned dataset shape: {df_clean.shape}")
    print(f"Years range: {df_clean['publication_year'].min()} - {df_clean['publication_year'].max()}")
    
    return df_clean

def prepare_analysis_data(df_clean):
    """
    Prepare data for specific analyses
    """
    # Papers by year
    papers_by_year = df_clean['publication_year'].value_counts().sort_index()
    
    # Top organizations (replace 'journal' with 'source_organization')
    top_organizations = df_clean['source_organization'].fillna('Unknown Organization').value_counts().head(15)
    
    # Word frequency in titles (replace 'title' with 'paper_title')
    all_titles = ' '.join(df_clean['paper_title'].dropna().astype(str))
    words = re.findall(r'\w+', all_titles.lower())
    word_freq = Counter(words)
    
    # Remove common stopwords
    stopwords = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from'])
    meaningful_words = {word: count for word, count in word_freq.items() 
                       if word not in stopwords and len(word) > 2}
    
    return {
        'papers_by_year': papers_by_year,
        'top_organizations': top_organizations,
        'word_freq': meaningful_words,
        'df_clean': df_clean
    }

def create_visualizations(analysis_data, save_path='images/'):
    """
    Create various visualizations for the dataset
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    print("=" * 50)
    print("CREATING VISUALIZATIONS")
    print("=" * 50)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('CORD-19 Dataset Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Publications over time
    papers_by_year = analysis_data['papers_by_year']
    axes[0, 0].bar(papers_by_year.index, papers_by_year.values, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Number of Publications by Year', fontweight='bold')
    axes[0, 0].set_xlabel('Year')
    axes[0, 0].set_ylabel('Number of Papers')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Top organizations
    top_organizations = analysis_data['top_organizations']
    axes[0, 1].barh(range(len(top_organizations)), top_organizations.values, color='lightgreen', alpha=0.7)
    axes[0, 1].set_yticks(range(len(top_organizations)))
    axes[0, 1].set_yticklabels(top_organizations.index, fontsize=8)
    axes[0, 1].set_title('Top 15 Organizations by Dataset Count', fontweight='bold')
    axes[0, 1].set_xlabel('Number of Datasets')
    
    # Plot 3: Word cloud
    word_freq = analysis_data['word_freq']
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
    axes[1, 0].imshow(wordcloud, interpolation='bilinear')
    axes[1, 0].set_title('Most Frequent Words in Paper Titles', fontweight='bold')
    axes[1, 0].axis('off')
    
    # Plot 4: Description word count distribution
    df_clean = analysis_data['df_clean']
    description_word_counts = df_clean[df_clean['has_description']]['description_word_count']
    axes[1, 1].hist(description_word_counts, bins=50, color='lightcoral', alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Distribution of Description Word Counts', fontweight='bold')
    axes[1, 1].set_xlabel('Word Count')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    
    # --- Begin: Unique filename logic ---
    base_filename = "cord19_analysis"
    ext = ".png"
    i = 1
    filename = f"{base_filename}{ext}"
    full_path = os.path.join(save_path, filename)
    while os.path.exists(full_path):
        filename = f"{base_filename}_{i}{ext}"
        full_path = os.path.join(save_path, filename)
        i += 1
    # --- End: Unique filename logic ---

    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Image saved as: {full_path}")
    
    # Additional detailed analysis
    print("\nDETAILED ANALYSIS RESULTS:")
    print("=" * 30)
    
    # Publication trends
    print(f"\n1. Publication Trends:")
    print(f"   Total papers: {len(df_clean)}")
    print(f"   Year with most publications: {papers_by_year.idxmax()} ({papers_by_year.max()} papers)")
    print(f"   Average publications per year: {papers_by_year.mean():.1f}")
    
    # Organization analysis
    print(f"\n2. Organization Analysis:")
    print(f"   Total unique organizations: {df_clean['source_organization'].nunique()}")
    print(f"   Top 5 organizations account for {top_organizations.head(5).sum() / len(df_clean) * 100:.1f}% of all datasets")
    
    # Description analysis
    print(f"\n3. Description Analysis:")
    print(f"   Datasets with descriptions: {df_clean['has_description'].sum()} ({df_clean['has_description'].mean() * 100:.1f}%)")
    print(f"   Average description length: {description_word_counts.mean():.1f} words")
    
    return fig

def main_analysis():
    """
    Main function to run the complete analysis
    """
    # Load data
    df = load_data()
    if df is None:
        return None, None
    
    # Basic exploration
    missing_df = data_overview(df)
    
    # Data cleaning
    df_clean = clean_data(df)
    
    # Prepare analysis data
    analysis_data = prepare_analysis_data(df_clean)
    
    # Create visualizations
    create_visualizations(analysis_data)
    
    return df_clean, analysis_data

if __name__ == "__main__":
    df_clean, analysis_data = main_analysis()