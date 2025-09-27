# CORD-19 Data Analysis and Streamlit Application

This project provides a full workflow for exploring, cleaning, analyzing, and interactively visualizing the CORD-19 dataset of COVID-19 research papers and datasets. It includes both a Python analysis script and a Streamlit web application.

---

## Project Structure

```
PYTHON FRAMEWORK ASSIGNMENT/
├── data/
│   └── CORD19_datasets.csv         # Main dataset (CSV)
├── images/
│   ├── cord19_analysis.png         # Saved analysis visualizations
│   ├── cord19_analysis_1.png       # Incremental visualizations
│   └── ...                         # More images/screenshots
├── cord19_analysis.py              # Data analysis and visualization script
├── cord19_streamlit.py             # Streamlit interactive dashboard
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

## Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/Akubrecah PYTHON-FRAMEWORK-ASSIGNMENT.git
   cd PYTHON FRAMEWORK ASSIGNMENT
   ```

2. **Set up a virtual environment (recommended)**  
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**  
   - Place your `CORD19_datasets.csv` file in the `data/` folder.

---

## Usage

### 1. Data Analysis Script

Run the analysis and visualization script:

```bash
python cord19_analysis.py
```

- **What it does:**  
  - Loads and explores the dataset.
  - Cleans and prepares the data.
  - Generates visualizations (saved in the `images/` folder with incrementing filenames).
  - Prints summary statistics and insights to the terminal.

### 2. Streamlit Interactive Dashboard

Launch the web app:

```bash
streamlit run cord19_streamlit.py
```

- **What it does:**  
  - Loads and cleans the dataset.
  - Provides sidebar filters for year, organization, and description availability.
  - Interactive tabs for:
    - Publication trends over time
    - Top organizations
    - Word frequency analysis (word cloud and table)
    - Data sample and statistics
  - Responsive layout for desktop and mobile.

---

## Visualizations

- All static visualizations from `cord19_analysis.py` are saved in the `images/` folder.
- Each run saves a new image with an incrementing filename (e.g., `cord19_analysis.png`, `cord19_analysis_1.png`, ...).

---

## Dataset Information

- **File:** `data/CORD19_datasets.csv`
- **Columns used:**  
  - `paper_title`, `description`, `last_updated`, `source_organization`, `dataset_name`, etc.
- **Description:**  
  Metadata about COVID-19 research papers and datasets, including titles, descriptions, organizations, and update dates.

---

## Key Features

- **Data Exploration:**  
  - Shape, columns, missing values, and statistics.
- **Data Cleaning:**  
  - Handles missing values, parses dates, creates new features.
- **Analysis:**  
  - Trends by year, top organizations, word frequency in titles, description length distribution.
- **Visualization:**  
  - Bar charts, histograms, word clouds, and summary tables.
- **Streamlit App:**  
  - Interactive filtering and visualization.
  - Data sample and statistics display.

---

## Technologies Used

- Python 3
- pandas, numpy
- matplotlib, seaborn
- wordcloud
- streamlit
- jupyter (optional, for notebooks)

---

## ⚡ Example Commands

**Run analysis and save visualizations:**
```bash
python cord19_analysis.py
```

**Launch the Streamlit dashboard:**
```bash
streamlit run cord19_streamlit.py
```

---

## Requirements

See `requirements.txt` for all dependencies:

```
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
streamlit>=1.12.0
wordcloud>=1.8.0
jupyter>=1.0.0
numpy>=1.21.0
```

---

## License

This project is for educational and research purposes.  
Dataset source: [CORD-19 Research Challenge](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge)

---

## Contact

For questions or contributions, please open an issue or pull request.