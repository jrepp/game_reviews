import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="Simple Steam Review Analyzer",
    page_icon="🎮",
    layout="wide"
)

# Main title
st.title("Simple Steam Review Analyzer")
st.markdown("A basic tool to verify Steam review sentiment analysis.")

# Create simple app that just focuses on loading data and sentiment analysis
def main():
    # File upload
    uploaded_file = st.file_uploader("Upload Steam Reviews CSV", type="csv")
    
    if uploaded_file:
        # Step 1: Load the data and display basic info
        st.header("1. Data Loading Test")
        
        # Load CSV and show sample
        df = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(df)} reviews")
        st.write("First 5 rows:")
        st.dataframe(df.head())
        
        # Show columns
        st.write("Columns in dataset:")
        st.write(df.columns.tolist())
        
        # Step 2: Identify review text column
        st.header("2. Review Text Identification")
        
        # Auto-detect review column
        review_col = None
        for col in df.columns:
            if 'review' in col.lower() or 'text' in col.lower() or 'comment' in col.lower():
                review_col = col
                break
        
        if not review_col:
            review_col = st.selectbox("Select the column containing review text:", df.columns)
        else:
            st.success(f"Detected review column: {review_col}")
            
        # Show text samples
        st.write("Sample review texts:")
        for i in range(min(3, len(df))):
            text = df.iloc[i][review_col]
            st.text_area(f"Review {i+1}", value=text, height=100)
            
        # Step 3: Basic sentiment analysis
        st.header("3. Basic Sentiment Analysis")
        
        # Define a very simple sentiment function
        def simple_sentiment(text):
            if not isinstance(text, str):
                return "neutral"
                
            text = text.lower()
            
            # Simple positive and negative word lists
            positive_words = ['good', 'great', 'love', 'awesome', 'excellent', 'recommend', 'fun', 'best']
            negative_words = ['bad', 'hate', 'terrible', 'awful', 'waste', 'boring', 'refund', 'not recommend']
            
            # Count occurrences
            pos_count = sum(text.count(word) for word in positive_words)
            neg_count = sum(text.count(word) for word in negative_words)
            
            # Determine sentiment
            if pos_count > neg_count:
                return "positive"
            elif neg_count > pos_count:
                return "negative"
            else:
                # For testing, let's bias toward positive if equal
                if len(text) < 100 or "recommend" in text:
                    return "positive"
                return "neutral"
        
        # Apply sentiment analysis
        with st.spinner("Analyzing sentiment..."):
            df['sentiment'] = df[review_col].apply(simple_sentiment)
            
            # Count sentiments
            sentiment_counts = df['sentiment'].value_counts()
            st.write("Sentiment distribution:")
            st.write(sentiment_counts)
            
            # Visualize
            fig = px.pie(
                names=sentiment_counts.index,
                values=sentiment_counts.values,
                title="Sentiment Distribution",
                color=sentiment_counts.index,
                color_discrete_map={'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
            )
            st.plotly_chart(fig)
            
            # Show examples of each sentiment
            st.write("Examples of each sentiment:")
            for sentiment in df['sentiment'].unique():
                st.subheader(f"{sentiment.title()} Review Examples")
                examples = df[df['sentiment'] == sentiment].head(3)
                for i, (_, row) in enumerate(examples.iterrows()):
                    st.text_area(f"{sentiment.title()} Example {i+1}", value=row[review_col], height=100)
        
        # Step 4: Identify playtime column
        if uploaded_file:
            st.header("4. Playtime Column Identification")
            
            # Auto-detect playtime column
            playtime_col = None
            for col in df.columns:
                if 'playtime' in col.lower() or 'hours' in col.lower() or 'minutes' in col.lower():
                    playtime_col = col
                    break
            
            if not playtime_col:
                playtime_col = st.selectbox("Select the column containing playtime data:", df.columns)
            else:
                st.success(f"Detected playtime column: {playtime_col}")
                
            # Convert playtime to numeric
            df[playtime_col] = pd.to_numeric(df[playtime_col], errors='coerce').fillna(0)
            
            # Show playtime distribution
            st.write(f"Playtime range: {df[playtime_col].min()} to {df[playtime_col].max()}")
            
            # Create playtime segments
            st.write("Creating player segments...")
            df['player_segment'] = pd.cut(
                df[playtime_col],
                bins=[0, 120, 600, float('inf')],
                labels=['Casual', 'Regular', 'Dedicated']
            )
            
            # Count segments
            segment_counts = df['player_segment'].value_counts()
            st.write("Player segments:")
            st.write(segment_counts)
            
            # Visualize
            fig = px.pie(
                names=segment_counts.index,
                values=segment_counts.values,
                title="Player Distribution by Engagement Level",
                color=segment_counts.index,
                color_discrete_map={'Casual':'#FF9999', 'Regular':'#66B2FF', 'Dedicated':'#99FF99'}
            )
            st.plotly_chart(fig)
            
            # Show sentiment by segment
            st.write("Sentiment by segment:")
            cross_tab = pd.crosstab(df['player_segment'], df['sentiment'])
            st.write(cross_tab)
            
            # Visualize sentiment by segment
            cross_tab_pct = cross_tab.div(cross_tab.sum(axis=1), axis=0) * 100
            
            fig = px.bar(
                cross_tab_pct.reset_index().melt(id_vars='player_segment'),
                x='player_segment',
                y='value',
                color='sentiment',
                title="Sentiment Distribution by Player Segment (%)",
                labels={'value': 'Percentage', 'player_segment': 'Player Segment'},
                color_discrete_map={'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
            )
            st.plotly_chart(fig)

if __name__ == "__main__":
    main()