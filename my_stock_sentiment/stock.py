# streamlit_stock_sentiment.py
# Freelance-ready Streamlit dashboard for Sentiment Analysis on Stock News Headlines
# Features:
# - Load dataset from a .zip (uploaded file path provided) or via file uploader
# - Detect headline column automatically (common names: headline, title, news)
# - Compute sentiment using NLTK VADER (fast) or TextBlob (optional)
# - Display sample table, sentiment distribution, time-series (if date column), wordcloud
# - Single-headline quick prediction box, and export predictions
# - Includes helpful sidebar controls and a professional layout

import streamlit as st
import pandas as pd
import numpy as np
import io
import zipfile
import os
from pathlib import Path
from datetime import datetime

# Visualization
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# NLP
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import re

# Ensure NLTK resources
nltk.download('vader_lexicon')

# ------------------------- Helper functions -------------------------

def detect_headline_column(df: pd.DataFrame):
    """Try to guess which column contains the news headline."""
    candidates = [c for c in df.columns if any(k in c.lower() for k in ['headline', 'title', 'news', 'text'])]
    if candidates:
        return candidates[0]
    # fallback: choose the longest text-like column
    text_cols = [c for c in df.columns if df[c].dtype == object]
    if not text_cols:
        return None
    # pick column with highest average string length
    avg_len = {c: df[c].dropna().astype(str).map(len).mean() for c in text_cols}
    return max(avg_len, key=avg_len.get)


def read_first_csv_from_zip(zip_path: str):
    """Open a zip file and return the first .csv or .xlsx as a dataframe."""
    z = zipfile.ZipFile(zip_path)
    # preference: .csv, then .xlsx, then .txt
    names = z.namelist()
    for ext in ['.csv', '.xlsx', '.xls', '.txt']:
        for name in names:
            if name.lower().endswith(ext):
                with z.open(name) as f:
                    if ext == '.csv' or ext == '.txt':
                        return pd.read_csv(f, encoding='utf-8')
                    else:
                        return pd.read_excel(f, encoding='latin1')
    raise FileNotFoundError('No CSV/XLSX/TXT found in zip')


def clean_text(s: str) -> str:
    if pd.isna(s):
        return ''
    s = str(s)
    s = re.sub(r'https?://\S+', '', s)  # remove urls
    s = re.sub(r'[^\w\s\$%.,-]', ' ', s)  # remove weird punctuation but keep $ % . , -
    s = re.sub(r'\s+', ' ', s).strip()
    return s


class SentimentModels:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()

    def vader_sentiment(self, text: str):
        if not text:
            return {'compound': 0.0, 'label': 'neutral'}
        scores = self.vader.polarity_scores(text)
        compound = scores['compound']
        if compound >= 0.05:
            label = 'positive'
        elif compound <= -0.05:
            label = 'negative'
        else:
            label = 'neutral'
        return {'compound': compound, 'label': label}

    def textblob_sentiment(self, text: str):
        if not text:
            return {'polarity': 0.0, 'label': 'neutral'}
        tb = TextBlob(text)
        p = tb.sentiment.polarity
        if p > 0.05:
            label = 'positive'
        elif p < -0.05:
            label = 'negative'
        else:
            label = 'neutral'
        return {'polarity': p, 'label': label}


# ------------------------- Streamlit layout -------------------------

st.set_page_config(page_title='Stock News Sentiment — Freelance Dashboard', layout='wide')

st.title('📈 Stock News Headlines — Sentiment Analysis (Freelance-ready)')
st.markdown(
    'A professional Streamlit dashboard to analyze sentiment in stock-related news headlines.\n'
    '**Upload** your dataset (CSV/Excel inside a ZIP is supported), preview it, run sentiment, visualize results, and export.'
)

# Sidebar: data input
st.sidebar.header('Upload & Options')

uploaded = st.sidebar.file_uploader('Upload a CSV/XLSX or a ZIP containing CSV/XLSX', type=['csv', 'xlsx', 'xls', 'zip', 'txt'])
# also provide the provided file path as quick-load button if it exists
provided_zip = '/mnt/data/ab35d406-40c6-456c-acbb-58ae7414969e.zip'
use_provided = False
if os.path.exists(provided_zip):
    if st.sidebar.button('Load provided dataset (freelancer upload)'):
        use_provided = True

# Model selection
model_choice = st.sidebar.selectbox('Sentiment model', ['VADER (recommended for short headlines)', 'TextBlob (alternative)'])
run_button = st.sidebar.button('Run sentiment on dataset')

# Columns selection will be shown after load

# Load data
@st.cache_data
def load_data_from_uploaded(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith('.zip'):
        # read zip bytes
        z = zipfile.ZipFile(uploaded_file)
        # find first csv/xlsx
        for ext in ['.csv', '.xlsx', '.xls', '.txt']:
            for n in z.namelist():
                if n.lower().endswith(ext):
                    with z.open(n) as f:
                        if ext == '.csv' or ext == '.txt':
                            return pd.read_csv(f)
                        else:
                            return pd.read_excel(f)
        raise FileNotFoundError('No CSV/XLSX/TXT found in uploaded zip')
    elif name.endswith('.csv') or name.endswith('.txt'):
        return pd.read_csv(uploaded_file, encoding='utf-8')
    else:
        return pd.read_excel(uploaded_file,  encoding='latin1')


def try_load():
    if use_provided:
        try:
            df = read_first_csv_from_zip(provided_zip)
            st.sidebar.success('Loaded provided zip file successfully.')
            return df
        except Exception as e:
            st.sidebar.error(f'Failed to load provided zip: {e}')
            return None
    elif uploaded is not None:
        try:
            df = load_data_from_uploaded(uploaded)
            st.sidebar.success('Uploaded file loaded successfully.')
            return df
        except Exception as e:
            st.sidebar.error(f'Upload failed: {e}')
            return None
    else:
        st.sidebar.info('No file uploaded. You can upload a file or click the provided dataset button (if available).')
        return None


df = try_load()

# If data loaded, show preview and let user pick columns
if df is not None:
    st.subheader('Data preview')
    st.write('Rows: ', len(df))
    st.dataframe(df.head(10))

    # detect headline column
    default_headline = detect_headline_column(df)
    headline_col = st.selectbox('Select headline column', options=df.columns.tolist(), index=(df.columns.get_loc(default_headline) if default_headline in df.columns else 0))
    # optional date column
    date_col = None
    date_cols = [c for c in df.columns if any(k in c.lower() for k in ['date', 'time'])]
    if date_cols:
        date_col = st.selectbox('Select date column (optional, used for time-series)', options=[None] + date_cols, index=0)

    # Clean text and preview
    if st.checkbox('Show cleaned headline sample'):
        sample = df[headline_col].astype(str).map(clean_text).head(10)
        st.write(sample)

    models = SentimentModels()

    if run_button:
        st.info('Running sentiment analysis — this may take a moment for large datasets.')
        df = df.copy()
        df['_clean_headline'] = df[headline_col].astype(str).map(clean_text)

        if model_choice.startswith('VADER'):
            results = df['_clean_headline'].map(models.vader_sentiment)
            df['sentiment_compound'] = [r['compound'] for r in results]
            df['sentiment_label'] = [r['label'] for r in results]
        else:
            results = df['_clean_headline'].map(models.textblob_sentiment)
            df['sentiment_polarity'] = [r['polarity'] for r in results]
            df['sentiment_label'] = [r['label'] for r in results]

        st.success('Sentiment analysis completed.')

        # Summary metrics
        st.subheader('Summary')
        col1, col2, col3 = st.columns(3)
        with col1:
            total = len(df)
            st.metric('Total headlines', f'{total}')
        with col2:
            pos = (df['sentiment_label'] == 'positive').sum()
            st.metric('Positive', f'{pos} ({pos/total:.1%})')
        with col3:
            neg = (df['sentiment_label'] == 'negative').sum()
            st.metric('Negative', f'{neg} ({neg/total:.1%})')

        # Distribution chart
        st.subheader('Sentiment distribution')
        dist = df['sentiment_label'].value_counts().reset_index()
        dist.columns = ['sentiment', 'count']
        fig = px.pie(dist, names='sentiment', values='count', title='Sentiment share')
        st.plotly_chart(fig, use_container_width=True)

        # Time series if available
        if date_col:
            try:
                df['_parsed_date'] = pd.to_datetime(df[date_col], errors='coerce')
                ts = df.dropna(subset=['_parsed_date']).set_index('_parsed_date')
                if not ts.empty:
                    st.subheader('Sentiment over time')
                    # rolling mean of compound or polarity
                    if 'sentiment_compound' in df.columns:
                        ts_res = ts['sentiment_compound'].resample('D').mean().dropna()
                        fig2 = px.line(ts_res, x=ts_res.index, y=ts_res.values, labels={'x': 'date', 'y':'avg compound'}, title='Average sentiment (compound) — daily')
                        st.plotly_chart(fig2, use_container_width=True)
                    elif 'sentiment_polarity' in df.columns:
                        ts_res = ts['sentiment_polarity'].resample('D').mean().dropna()
                        fig2 = px.line(ts_res, x=ts_res.index, y=ts_res.values, labels={'x': 'date', 'y':'avg polarity'}, title='Average polarity — daily')
                        st.plotly_chart(fig2, use_container_width=True)
            except Exception as e:
                st.warning(f'Could not create time-series plot: {e}')

        # Show top positive and negative
        st.subheader('Top headlines')
        try:
            pos_top = df[df['sentiment_label'] == 'positive'].sort_values(by=['sentiment_compound' if 'sentiment_compound' in df.columns else 'sentiment_polarity'], ascending=False).head(10)
            neg_top = df[df['sentiment_label'] == 'negative'].sort_values(by=['sentiment_compound' if 'sentiment_compound' in df.columns else 'sentiment_polarity']).head(10)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('**Top Positive**')
                st.table(pos_top[[headline_col, '_clean_headline']].reset_index(drop=True).head(10))
            with c2:
                st.markdown('**Top Negative**')
                st.table(neg_top[[headline_col, '_clean_headline']].reset_index(drop=True).head(10))
        except Exception as e:
            st.warning(f'Could not compute top headlines: {e}')

        # Wordcloud for positive and negative
        st.subheader('Wordclouds')
        wc_col1, wc_col2 = st.columns(2)
        try:
            for widget_col, lbl in [(wc_col1, 'positive'), (wc_col2, 'negative')]:
                txt = ' '.join(df[df['sentiment_label'] == lbl]['_clean_headline'].dropna().astype(str).tolist())
                if not txt:
                    widget_col.info(f'No {lbl} headlines for wordcloud')
                    continue
                wc = WordCloud(width=600, height=400, background_color='white').generate(txt)
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                widget_col.pyplot(fig)
        except Exception as e:
            st.warning(f'Wordcloud generation failed: {e}')

        # Full results table with download
        st.subheader('Full results')
        st.dataframe(df.head(200))
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button('Download results as CSV', data=csv, file_name='sentiment_results.csv', mime='text/csv')

        st.success('Dashboard run finished. You can tweak options and re-run.')

    # Quick single-prediction box
    st.sidebar.markdown('---')
    st.sidebar.header('Quick headline check')
    single_headline = st.sidebar.text_area('Enter a headline for quick prediction', height=80)
    if st.sidebar.button('Predict headline'):
        sh = clean_text(single_headline)
        if not sh:
            st.sidebar.error('Please enter a headline.')
        else:
            models = SentimentModels()
            if model_choice.startswith('VADER'):
                out = models.vader_sentiment(sh)
                st.sidebar.success(f"Label: {out['label']} — compound={out['compound']:.3f}")
            else:
                out = models.textblob_sentiment(sh)
                st.sidebar.success(f"Label: {out['label']} — polarity={out['polarity']:.3f}")

else:
    st.info('No dataset loaded yet. Upload a file from the sidebar or use the provided dataset button if available.')

# Footer / freelancing tips
st.markdown('---')

# Freelance-ready notes:\n'
# This app is packaged as a single-file Streamlit app, ready to send to a client.\n'
# Suggested `requirements.txt` (put next to this file):\n'
    
# streamlit
# pandas
# numpy
# nltk
# textblob
# plotly
# wordcloud
# matplotlib
    
# '- To run locally: `pip install -r requirements.txt` then `streamlit run streamlit_stock_sentiment.py`\n'


st.write('Made for freelance delivery — neat layout, clear controls, and exportable results. Good luck!')
