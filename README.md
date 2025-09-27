# CORD-19 Data Analysis and Streamlit Application

CORD-19 dataset analysis and Streamlit application.

## Project Structure

```
Frameworks_Assignment/
├── data/
│   └── metadata.csv (or a sample if full dataset is too large)
├── cord19_analysis.py
├── cord19_streamlit.py
├── requirements.txt
├── README.md
└── images/ (for visualizations)
```

## Part 1: Data Loading and Basic Exploration

```python
# cord19_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load the data
def load_data(file_path='data/metadata.csv'):
    """
    Load the CORD-19 metadata dataset
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print("File not found. Please ensure metadata.csv is in the data/ directory")
        return None

def basic_exploration(df):
    """
    Perform basic exploration of the dataset
    """
    print("=" * 50)
    print("BASIC DATASET EXPLORATION")
    print("=" * 50)
    
    # Display basic information
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

if __name__ == "__main__":
    # Load data
    df = load_data()
    
    if df is not None:
        # Basic exploration
        missing_df = basic_exploration(df)
```

## Part 2: Data Cleaning and Preparation

```python
# cord19_analysis.py (continued)

def clean_data(df):
    """
    Clean and prepare the dataset for analysis
    """
    print("=" * 50)
    print("DATA CLEANING AND PREPARATION")
    print("=" * 50)
    
    # Create a copy of the dataframe
    df_clean = df.copy()
    
    # Handle publication date
    print("\n1. Handling publication dates...")
    df_clean['publish_time'] = pd.to_datetime(df_clean['publish_time'], errors='coerce')
    df_clean['publication_year'] = df_clean['publish_time'].dt.year
    
    # Fill missing years with mode or specific value
    mode_year = df_clean['publication_year'].mode()[0] if not df_clean['publication_year'].mode().empty else 2020
    df_clean['publication_year'].fillna(mode_year, inplace=True)
    df_clean['publication_year'] = df_clean['publication_year'].astype(int)
    
    # Handle missing abstracts
    print("\n2. Handling missing abstracts...")
    df_clean['abstract'] = df_clean['abstract'].fillna('No abstract available')
    
    # Handle missing titles
    print("\n3. Handling missing titles...")
    df_clean = df_clean.dropna(subset=['title'])
    
    # Create abstract word count
    print("\n4. Creating new features...")
    df_clean['abstract_word_count'] = df_clean['abstract'].apply(lambda x: len(str(x).split()))
    df_clean['has_abstract'] = df_clean['abstract'] != 'No abstract available'
    
    # Clean journal names
    df_clean['journal'] = df_clean['journal'].fillna('Unknown Journal')
    
    print(f"Cleaned dataset shape: {df_clean.shape}")
    print(f"Years range: {df_clean['publication_year'].min()} - {df_clean['publication_year'].max()}")
    
    return df_clean

def prepare_analysis_data(df_clean):
    """
    Prepare data for specific analyses
    """
    # Papers by year
    papers_by_year = df_clean['publication_year'].value_counts().sort_index()
    
    # Top journals
    top_journals = df_clean['journal'].value_counts().head(15)
    
    # Word frequency in titles
    all_titles = ' '.join(df_clean['title'].dropna().astype(str))
    words = re.findall(r'\w+', all_titles.lower())
    word_freq = Counter(words)
    
    # Remove common stopwords
    stopwords = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from'])
    meaningful_words = {word: count for word, count in word_freq.items() 
                       if word not in stopwords and len(word) > 2}
    
    return {
        'papers_by_year': papers_by_year,
        'top_journals': top_journals,
        'word_freq': meaningful_words,
        'df_clean': df_clean
    }
```

## Part 3: Data Analysis and Visualization

```python
# cord19_analysis.py (continued)

def create_visualizations(analysis_data, save_path='images/'):
    """
    Create various visualizations for the dataset
    """
    import os
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
    
    # Plot 2: Top journals
    top_journals = analysis_data['top_journals']
    axes[0, 1].barh(range(len(top_journals)), top_journals.values, color='lightgreen', alpha=0.7)
    axes[0, 1].set_yticks(range(len(top_journals)))
    axes[0, 1].set_yticklabels(top_journals.index, fontsize=8)
    axes[0, 1].set_title('Top 15 Journals by Publication Count', fontweight='bold')
    axes[0, 1].set_xlabel('Number of Papers')
    
    # Plot 3: Word cloud
    word_freq = analysis_data['word_freq']
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
    axes[1, 0].imshow(wordcloud, interpolation='bilinear')
    axes[1, 0].set_title('Most Frequent Words in Paper Titles', fontweight='bold')
    axes[1, 0].axis('off')
    
    # Plot 4: Abstract word count distribution
    df_clean = analysis_data['df_clean']
    abstract_word_counts = df_clean[df_clean['has_abstract']]['abstract_word_count']
    axes[1, 1].hist(abstract_word_counts, bins=50, color='lightcoral', alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Distribution of Abstract Word Counts', fontweight='bold')
    axes[1, 1].set_xlabel('Word Count')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/cord19_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Additional detailed analysis
    print("\nDETAILED ANALYSIS RESULTS:")
    print("=" * 30)
    
    # Publication trends
    print(f"\n1. Publication Trends:")
    print(f"   Total papers: {len(df_clean)}")
    print(f"   Year with most publications: {papers_by_year.idxmax()} ({papers_by_year.max()} papers)")
    print(f"   Average publications per year: {papers_by_year.mean():.1f}")
    
    # Journal analysis
    print(f"\n2. Journal Analysis:")
    print(f"   Total unique journals: {df_clean['journal'].nunique()}")
    print(f"   Top 5 journals account for {top_journals.head(5).sum() / len(df_clean) * 100:.1f}% of all papers")
    
    # Abstract analysis
    print(f"\n3. Abstract Analysis:")
    print(f"   Papers with abstracts: {df_clean['has_abstract'].sum()} ({df_clean['has_abstract'].mean() * 100:.1f}%)")
    print(f"   Average abstract length: {abstract_word_counts.mean():.1f} words")
    
    return fig

def main_analysis():
    """
    Main function to run the complete analysis
    """
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Basic exploration
    missing_df = basic_exploration(df)
    
    # Data cleaning
    df_clean = clean_data(df)
    
    # Prepare analysis data
    analysis_data = prepare_analysis_data(df_clean)
    
    # Create visualizations
    create_visualizations(analysis_data)
    
    return df_clean, analysis_data

if __name__ == "__main__":
    df_clean, analysis_data = main_analysis()
```

## Part 4: Streamlit Application

```python
# cord19_streamlit.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from collections import Counter
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="CORD-19 Data Explorer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    """Load and cache the dataset"""
    try:
        df = pd.read_csv('data/metadata.csv')
        return df
    except FileNotFoundError:
        st.error("File not found. Please ensure metadata.csv is in the data/ directory")
        return None

@st.cache_data
def clean_data(df):
    """Clean the dataset"""
    df_clean = df.copy()
    
    # Handle dates
    df_clean['publish_time'] = pd.to_datetime(df_clean['publish_time'], errors='coerce')
    df_clean['publication_year'] = df_clean['publish_time'].dt.year
    mode_year = df_clean['publication_year'].mode()[0] if not df_clean['publication_year'].mode().empty else 2020
    df_clean['publication_year'].fillna(mode_year, inplace=True)
    df_clean['publication_year'] = df_clean['publication_year'].astype(int)
    
    # Handle missing values
    df_clean['abstract'] = df_clean['abstract'].fillna('No abstract available')
    df_clean = df_clean.dropna(subset=['title'])
    df_clean['journal'] = df_clean['journal'].fillna('Unknown Journal')
    df_clean['abstract_word_count'] = df_clean['abstract'].apply(lambda x: len(str(x).split()))
    df_clean['has_abstract'] = df_clean['abstract'] != 'No abstract available'
    
    return df_clean

def main():
    st.title("🔬 CORD-19 COVID-19 Research Papers Explorer")
    st.markdown("""
    This interactive dashboard explores the CORD-19 dataset containing metadata about COVID-19 research papers.
    Use the filters in the sidebar to explore specific subsets of the data.
    """)
    
    # Load data
    with st.spinner('Loading data...'):
        df = load_data()
    
    if df is None:
        st.stop()
    
    # Clean data
    with st.spinner('Processing data...'):
        df_clean = clean_data(df)
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Year range selector
    min_year = int(df_clean['publication_year'].min())
    max_year = int(df_clean['publication_year'].max())
    year_range = st.sidebar.slider(
        "Select publication year range:",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )
    
    # Journal selector
    journals = ['All'] + sorted(df_clean['journal'].value_counts().head(20).index.tolist())
    selected_journal = st.sidebar.selectbox("Select journal:", journals)
    
    # Abstract availability filter
    abstract_filter = st.sidebar.radio("Abstract availability:", ['All', 'With Abstract', 'Without Abstract'])
    
    # Apply filters
    filtered_df = df_clean[
        (df_clean['publication_year'] >= year_range[0]) & 
        (df_clean['publication_year'] <= year_range[1])
    ]
    
    if selected_journal != 'All':
        filtered_df = filtered_df[filtered_df['journal'] == selected_journal]
    
    if abstract_filter == 'With Abstract':
        filtered_df = filtered_df[filtered_df['has_abstract'] == True]
    elif abstract_filter == 'Without Abstract':
        filtered_df = filtered_df[filtered_df['has_abstract'] == False]
    
    # Main content
    st.header("Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Papers", len(filtered_df))
    
    with col2:
        st.metric("Years Covered", f"{year_range[0]} - {year_range[1]}")
    
    with col3:
        st.metric("Unique Journals", filtered_df['journal'].nunique())
    
    with col4:
        st.metric("Papers with Abstracts", 
                 f"{filtered_df['has_abstract'].sum()} ({filtered_df['has_abstract'].mean() * 100:.1f}%)")
    
    # Visualizations
    st.header("Data Visualizations")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "Publication Trends", 
        "Journal Analysis", 
        "Word Analysis", 
        "Data Sample"
    ])
    
    with tab1:
        st.subheader("Publication Trends Over Time")
        
        # Publications by year
        yearly_counts = filtered_df['publication_year'].value_counts().sort_index()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(yearly_counts.index, yearly_counts.values, color='skyblue', alpha=0.7)
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of Publications')
        ax.set_title('Publications by Year')
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # Monthly trends (if data is available)
        try:
            monthly_data = filtered_df.set_index('publish_time').resample('M').size()
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(monthly_data.index, monthly_data.values, color='darkblue', linewidth=2)
            ax2.set_xlabel('Month')
            ax2.set_ylabel('Number of Publications')
            ax2.set_title('Monthly Publication Trends')
            ax2.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            st.pyplot(fig2)
        except:
            st.info("Insufficient date data for monthly analysis")
    
    with tab2:
        st.subheader("Journal Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top journals
            top_journals = filtered_df['journal'].value_counts().head(10)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(range(len(top_journals)), top_journals.values, color='lightgreen')
            ax.set_yticks(range(len(top_journals)))
            ax.set_yticklabels(top_journals.index)
            ax.set_xlabel('Number of Publications')
            ax.set_title('Top 10 Journals by Publication Count')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            # Journal statistics
            journal_stats = pd.DataFrame({
                'Journal': top_journals.index,
                'Publications': top_journals.values,
                'Percentage': (top_journals.values / len(filtered_df)) * 100
            })
            st.dataframe(journal_stats, use_container_width=True)
    
    with tab3:
        st.subheader("Word Frequency Analysis")
        
        # Word cloud of titles
        if len(filtered_df) > 0:
            all_titles = ' '.join(filtered_df['title'].dropna().astype(str))
            words = re.findall(r'\w+', all_titles.lower())
            word_freq = Counter(words)
            
            # Remove common stopwords
            stopwords = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
                           'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were'])
            meaningful_words = {word: count for word, count in word_freq.items() 
                              if word not in stopwords and len(word) > 2}
            
            if meaningful_words:
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(meaningful_words)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                ax.set_title('Most Frequent Words in Paper Titles')
                st.pyplot(fig)
                
                # Top words table
                top_words = pd.DataFrame(
                    sorted(meaningful_words.items(), key=lambda x: x[1], reverse=True)[:20],
                    columns=['Word', 'Frequency']
                )
                st.dataframe(top_words, use_container_width=True)
            else:
                st.info("No meaningful words found in the filtered dataset")
        else:
            st.warning("No data available for word analysis with current filters")
    
    with tab4:
        st.subheader("Sample Data")
        
        # Data sample
        st.write(f"Showing 10 random papers from the filtered dataset ({len(filtered_df)} total):")
        sample_data = filtered_df[['title', 'journal', 'publication_year', 'abstract_word_count']].sample(
            min(10, len(filtered_df))
        )
        st.dataframe(sample_data, use_container_width=True)
        
        # Data statistics
        st.subheader("Dataset Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Column Information:**")
            column_info = pd.DataFrame({
                'Column': filtered_df.columns,
                'Non-Null Count': filtered_df.notnull().sum(),
                'Data Type': filtered_df.dtypes
            })
            st.dataframe(column_info, use_container_width=True)
        
        with col2:
            st.write("**Missing Values (Top 10):**")
            missing_data = filtered_df.isnull().sum()
            missing_percent = (missing_data / len(filtered_df)) * 100
            missing_df = pd.DataFrame({
                'Missing Count': missing_data,
                'Missing Percentage': missing_percent
            }).sort_values('Missing Count', ascending=False).head(10)
            st.dataframe(missing_df, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Dataset Source:** [CORD-19 Research Challenge](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge) | "
        "**Created with:** Python, Pandas, Matplotlib, Streamlit"
    )

if __name__ == "__main__":
    main()
```

## Supporting Files

### requirements.txt
```
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
streamlit>=1.12.0
wordcloud>=1.8.0
jupyter>=1.0.0
numpy>=1.21.0
```

### README.md
```markdown
# CORD-19 COVID-19 Research Papers Analysis

## Project Overview
This project analyzes the CORD-19 dataset containing metadata about COVID-19 research papers. The analysis includes data exploration, cleaning, visualization, and an interactive web application.

## Files Structure
```
Frameworks_Assignment/
├── data/
│   └── metadata.csv
├── cord19_analysis.py
├── cord19_streamlit.py
├── requirements.txt
├── README.md
└── images/
```

## Installation
1. Clone this repository
2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Data Analysis
Run the analysis script to generate insights and visualizations:
```bash
python cord19_analysis.py
```

### Streamlit Application
Launch the interactive web application:
```bash
streamlit run cord19_streamlit.py
```

## Key Findings
- Analysis of publication trends over time
- Identification of top journals publishing COVID-19 research
- Word frequency analysis in paper titles
- Distribution of abstract lengths

## Dataset Information
The CORD-19 dataset contains metadata about COVID-19 research papers, including titles, abstracts, authors, publication dates, and source information.

## Technologies Used
- Python
- Pandas (data manipulation)
- Matplotlib/Seaborn (visualization)
- Streamlit (web application)
- WordCloud (text analysis)
```

## How to Run the Project

1. **Download the dataset**: Get the `metadata.csv` file from the CORD-19 dataset on Kaggle and place it in the `data/` folder.

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the analysis**:
```bash
python cord19_analysis.py
```

4. **Launch the Streamlit app**:
```bash
streamlit run cord19_streamlit.py
```

## Key Features

### Analysis Script (`cord19_analysis.py`)
- Loads and explores the dataset
- Cleans and prepares data for analysis
- Creates comprehensive visualizations
- Generates insights about publication trends

### Streamlit Application (`cord19_streamlit.py`)
- Interactive filters for year range, journal selection, and abstract availability
- Multiple visualization tabs
- Real-time data filtering and updating
- Responsive layout for different screen sizes