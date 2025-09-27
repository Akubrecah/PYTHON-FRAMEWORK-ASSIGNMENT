import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import wordcloud 
import re
from collections import Counter
import numpy as np

# page configuration
st.set_page_config(
    page_title = "CORD19 Data Explorer",
    page_icon = "🔬",
    layout = "wide",
    inisital_sidebar_state = "expanded"
)
@set.cache_data
def load_data(file_path = 'data/CORD19_datasets.csv'):
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path} with shape {df.shape[0]} rows and {df.shape[1]} columns.")
        return df

    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

@st.cache_data
def clean_data(df):
    # clean dataset
    df_clean = df.copy()

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