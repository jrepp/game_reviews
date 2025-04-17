import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
import datetime

# Set page config
st.set_page_config(
    page_title="Steam Review Classification",
    page_icon="🎮",
    layout="wide"
)

# Main title
st.title("Steam Review Classification Tool")
st.markdown("Analyze Steam reviews for sentiment, humor, recommendations, and recency.")

# Sidebar settings
st.sidebar.header("Analysis Settings")

# Review classification function
def classify_review(text):
    """
    Classify a review's sentiment, humor, and extract recommendations
    Returns a dictionary with classification results
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {
            "sentiment": "neutral",
            "is_funny": False,
            "recommendation": None
        }
        
    text = text.lower()
    result = {}
    
    # Classify sentiment (positive/negative/neutral)
    positive_patterns = [
        'love', 'great', 'good', 'excellent', 'amazing', 'awesome', 'fun', 'enjoy', 
        'best', 'fantastic', 'worth', 'recommend', 'addictive', 'immersive',
        'beautiful', 'masterpiece', 'brilliant', 'perfect', 'wonderful'
    ]
    
    negative_patterns = [
        'hate', 'bad', 'terrible', 'awful', 'poor', 'broken', 'disappointing', 'worst', 
        'boring', 'unplayable', 'waste', 'refund', 'buggy', 'crash', 'garbage',
        'avoid', 'overpriced', 'frustrating', 'annoying', 'trash', 'horrible'
    ]
    
    # Look for explicit Steam recommendation patterns
    if "recommend" in text and not any(phrase in text for phrase in ["not recommend", "don't recommend", "wouldn't recommend", "cannot recommend"]):
        result["sentiment"] = "positive"
    elif any(phrase in text for phrase in ["not recommend", "don't recommend", "wouldn't recommend", "cannot recommend"]):
        result["sentiment"] = "negative"
    else:
        # Count positive and negative words with word boundaries
        pos_count = sum(1 for pattern in positive_patterns if re.search(r'\b' + pattern + r'\b', text))
        neg_count = sum(1 for pattern in negative_patterns if re.search(r'\b' + pattern + r'\b', text))
        
        if pos_count > neg_count:
            result["sentiment"] = "positive"
        elif neg_count > pos_count:
            result["sentiment"] = "negative"
        else:
            # If tied, look for other clues
            if any(phrase in text for phrase in ["worth the money", "worth it", "enjoyed", "like this game"]):
                result["sentiment"] = "positive"
            elif any(phrase in text for phrase in ["waste of money", "waste of time", "don't buy"]):
                result["sentiment"] = "negative"
            else:
                # Default to neutral if still undetermined
                result["sentiment"] = "neutral"
    
    # Detect humor/funny reviews
    humor_patterns = [
        'lol', 'funny', 'laugh', 'hilarious', 'haha', 'lmao', 'rofl', 'joke',
        '😂', '🤣', 'xd', 'hehe', 'humor', 'comedy', 'amusing', 'ridiculous'
    ]
    
    # Look for humor indicators
    humor_count = sum(1 for pattern in humor_patterns if pattern in text)
    result["is_funny"] = humor_count > 0
    
    # Extract recommendations
    recommendation_phrases = [
        'should add', 'need to add', 'would be better if', 'could improve',
        'please add', 'wish they would', 'recommend adding', 'suggest', 
        'needs more', 'should implement', 'hope they add'
    ]
    
    # Look for recommendation sentences
    recommendations = []
    sentences = re.split(r'[.!?]+', text)
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 10:  # Skip very short sentences
            if any(phrase in sentence for phrase in recommendation_phrases):
                recommendations.append(sentence)
    
    result["recommendation"] = recommendations if recommendations else None
    
    return result

# Function to classify recency based on timestamp
def classify_recency(timestamp):
    """
    Classifies when the user last played based on the timestamp
    """
    if timestamp is None or pd.isna(timestamp):
        return "unknown"
        
    try:
        # Convert timestamp to datetime
        timestamp = int(timestamp)
        last_played = datetime.datetime.fromtimestamp(timestamp)
        
        # Get current time
        current_time = datetime.datetime.now()
        
        # Calculate time difference
        time_diff = current_time - last_played
        
        # Classify
        if time_diff.days < 7:
            return "fresh"  # Less than a week
        elif time_diff.days < 30:
            return "monthly"  # Last month
        elif time_diff.days < 365:
            return "last year"  # Last year
        else:
            return "older than a year"
    except (ValueError, TypeError):
        return "unknown"

# Function to analyze a dataframe
def analyze_reviews(df, review_col):
    """
    Process all reviews in the dataframe
    """
    # Create copy to avoid warnings
    df = df.copy()
    
    # Apply classification to each review
    classifications = []
    for idx, row in df.iterrows():
        text = row[review_col]
        result = classify_review(text)
        classifications.append(result)
    
    # Add sentiment and humor columns
    df['sentiment'] = [c["sentiment"] for c in classifications]
    df['is_funny'] = [c["is_funny"] for c in classifications]
    df['recommendations'] = [c["recommendation"] for c in classifications]
    
    # Process recency
    timestamp_col = None
    for col in df.columns:
        if 'timestamp' in col.lower() or 'last_played' in col.lower():
            timestamp_col = col
            break
    
    if timestamp_col:
        df['recency'] = df[timestamp_col].apply(classify_recency)
    else:
        df['recency'] = "unknown"
    
    return df

# Main function
def main():
    # File upload
    uploaded_file = st.file_uploader("Upload Steam Reviews CSV", type="csv")
    
    # Test with sample text
    st.header("Test Classification on Sample Text")
    sample_text = st.text_area(
        "Enter a sample review text to test:", 
        value="This game is really good, I highly recommend it! Developers should add more content though.", 
        height=150
    )
    
    # Analyze sample text
    if sample_text:
        st.write("Classification results for sample text:")
        sample_result = classify_review(sample_text)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Sentiment", sample_result["sentiment"].title())
        col2.metric("Is Funny", "Yes" if sample_result["is_funny"] else "No")
        
        with col3:
            st.write("Recommendations:")
            if sample_result["recommendation"]:
                for rec in sample_result["recommendation"]:
                    st.write(f"- {rec}")
            else:
                st.write("No specific recommendations found.")
    
    if uploaded_file:
        # Load data
        df = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(df)} reviews")
        
        # Auto-detect review column
        review_col = None
        # First try exact match for 'review'
        if 'review' in df.columns:
            review_col = 'review'
        else:
            # Fall back to broader matching if 'review' column doesn't exist
            for col in df.columns:
                col_lower = col.lower()
                # Avoid columns that are likely counts or metadata
                if 'review' in col_lower and not any(x in col_lower for x in ['number', 'count', 'score', 'id', 'url']):
                    review_col = col
                    break
        
        if not review_col:
            review_col = st.selectbox("Select the column containing review text:", df.columns)
        else:
            st.success(f"Detected review column: {review_col}")
        
        # Show sample of review text for verification
        with st.expander("View sample reviews"):
            for i in range(min(5, len(df))):
                text = df.iloc[i][review_col]
                preview = text[:200] + "..." if isinstance(text, str) and len(text) > 200 else text
                st.text_area(f"Review {i+1}", value=preview, height=100)
        
        # Analyze reviews
        with st.spinner("Analyzing reviews..."):
            result_df = analyze_reviews(df, review_col)
            
            # Display results
            st.header("Review Classification Results")
            
            # Sentiment distribution
            sentiment_counts = result_df['sentiment'].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Sentiment Distribution")
                
                fig = px.pie(
                    names=sentiment_counts.index,
                    values=sentiment_counts.values,
                    title="Sentiment Distribution",
                    color=sentiment_counts.index,
                    color_discrete_map={'positive': 'green', 'neutral': 'gray', 'negative': 'red'},
                    hole=0.4
                )
                st.plotly_chart(fig)
            
            with col2:
                st.subheader("Humor Distribution")
                
                humor_counts = result_df['is_funny'].value_counts()
                humor_labels = ["Funny" if idx else "Not Funny" for idx in humor_counts.index]
                
                fig = px.pie(
                    names=humor_labels,
                    values=humor_counts.values,
                    title="Humor Distribution",
                    color=humor_labels,
                    color_discrete_map={'Funny': 'blue', 'Not Funny': 'lightgray'},
                    hole=0.4
                )
                st.plotly_chart(fig)
            
            # Recency distribution
            st.subheader("Play Recency Distribution")
            
            recency_counts = result_df['recency'].value_counts()
            
            fig = px.bar(
                x=recency_counts.index,
                y=recency_counts.values,
                title="Play Recency Distribution",
                labels={'x': 'Recency', 'y': 'Count'},
                color=recency_counts.index,
                color_discrete_map={
                    'fresh': '#00CC66',
                    'monthly': '#66CCFF',
                    'last year': '#FFCC66',
                    'older than a year': '#FF6666',
                    'unknown': '#CCCCCC'
                }
            )
            st.plotly_chart(fig)
            
            # Sentiment by recency
            st.subheader("Sentiment by Play Recency")
            
            sentiment_recency = pd.crosstab(result_df['recency'], result_df['sentiment'])
            
            fig = px.bar(
                sentiment_recency.reset_index().melt(id_vars='recency'),
                x='recency',
                y='value',
                color='sentiment',
                title="Sentiment Distribution by Play Recency",
                labels={'value': 'Count', 'recency': 'Play Recency'},
                color_discrete_map={'positive': 'green', 'neutral': 'gray', 'negative': 'red'},
                barmode='group'
            )
            st.plotly_chart(fig)
            
            # Recommendations 
            st.header("Extracted Recommendations")
            
            # Count reviews with recommendations
            has_recommendations = result_df['recommendations'].apply(lambda x: x is not None and len(x) > 0)
            rec_count = has_recommendations.sum()
            
            st.write(f"Found {rec_count} reviews with specific recommendations")
            
            if rec_count > 0:
                # Show sample recommendations
                with st.expander("View sample recommendations"):
                    rec_samples = result_df[has_recommendations].head(10)
                    
                    for i, (_, row) in enumerate(rec_samples.iterrows()):
                        st.subheader(f"Review {i+1} ({row['sentiment'].title()} sentiment)")
                        
                        # Show recommendations
                        if row['recommendations']:
                            for j, rec in enumerate(row['recommendations']):
                                st.write(f"{j+1}. {rec}")
                        
                        # Show review preview
                        review_text = row[review_col]
                        preview = review_text[:150] + "..." if isinstance(review_text, str) and len(review_text) > 150 else review_text
                        st.text_area("Review text", value=preview, height=100)
            
            # Detailed view by category
            st.header("Browse Reviews by Category")
            
            # Create tabs for different views
            browse_tabs = st.tabs(["By Sentiment", "By Recency", "By Humor", "With Recommendations"])
            
            # By Sentiment tab
            with browse_tabs[0]:
                sentiment_filter = st.selectbox(
                    "Select sentiment to view:",
                    options=["All"] + list(result_df['sentiment'].unique()),
                    key="sentiment_filter"
                )
                
                filtered_df = result_df if sentiment_filter == "All" else result_df[result_df['sentiment'] == sentiment_filter]
                
                st.write(f"Showing {len(filtered_df)} {sentiment_filter.lower() if sentiment_filter != 'All' else ''} reviews")
                
                for i, (_, row) in enumerate(filtered_df.head(5).iterrows()):
                    with st.expander(f"Review {i+1} - {row['sentiment'].title()} - {row['recency']}"):
                        review_text = row[review_col]
                        st.write(review_text)
                        
                        if row['recommendations']:
                            st.write("**Recommendations:**")
                            for rec in row['recommendations']:
                                st.write(f"- {rec}")
            
            # By Recency tab
            with browse_tabs[1]:
                recency_filter = st.selectbox(
                    "Select recency to view:",
                    options=["All"] + list(result_df['recency'].unique()),
                    key="recency_filter"
                )
                
                filtered_df = result_df if recency_filter == "All" else result_df[result_df['recency'] == recency_filter]
                
                st.write(f"Showing {len(filtered_df)} {recency_filter} reviews")
                
                for i, (_, row) in enumerate(filtered_df.head(5).iterrows()):
                    with st.expander(f"Review {i+1} - {row['sentiment'].title()} - {row['recency']}"):
                        review_text = row[review_col]
                        st.write(review_text)
                        
                        if row['recommendations']:
                            st.write("**Recommendations:**")
                            for rec in row['recommendations']:
                                st.write(f"- {rec}")
            
            # By Humor tab
            with browse_tabs[2]:
                humor_filter = st.radio(
                    "Show funny reviews:",
                    options=["All", "Only Funny", "Not Funny"],
                    key="humor_filter",
                    horizontal=True
                )
                
                if humor_filter == "All":
                    filtered_df = result_df
                elif humor_filter == "Only Funny":
                    filtered_df = result_df[result_df['is_funny']]
                else:
                    filtered_df = result_df[~result_df['is_funny']]
                
                st.write(f"Showing {len(filtered_df)} reviews")
                
                for i, (_, row) in enumerate(filtered_df.head(5).iterrows()):
                    humor_label = "Funny" if row['is_funny'] else "Not Funny"
                    with st.expander(f"Review {i+1} - {row['sentiment'].title()} - {humor_label}"):
                        review_text = row[review_col]
                        st.write(review_text)
            
            # With Recommendations tab
            with browse_tabs[3]:
                recommend_filter = st.radio(
                    "Show reviews with recommendations:",
                    options=["All", "Only with Recommendations", "Without Recommendations"],
                    key="recommend_filter",
                    horizontal=True
                )
                
                if recommend_filter == "All":
                    filtered_df = result_df
                elif recommend_filter == "Only with Recommendations":
                    filtered_df = result_df[has_recommendations]
                else:
                    filtered_df = result_df[~has_recommendations]
                
                st.write(f"Showing {len(filtered_df)} reviews")
                
                for i, (_, row) in enumerate(filtered_df.head(5).iterrows()):
                    rec_label = "Has Recommendations" if row['recommendations'] else "No Recommendations"
                    with st.expander(f"Review {i+1} - {row['sentiment'].title()} - {rec_label}"):
                        review_text = row[review_col]
                        st.write(review_text)
                        
                        if row['recommendations']:
                            st.write("**Recommendations:**")
                            for rec in row['recommendations']:
                                st.write(f"- {rec}")
            
            # Download results
            st.header("Download Results")
            
            # Prepare download data
            download_df = result_df[[review_col, 'sentiment', 'is_funny', 'recency']]
            # Convert recommendation lists to strings for CSV compatibility
            download_df['recommendations'] = result_df['recommendations'].apply(
                lambda x: "; ".join(x) if isinstance(x, list) and len(x) > 0 else ""
            )
            
            # Convert to CSV
            csv = download_df.to_csv(index=False)
            
            st.download_button(
                label="Download analyzed data as CSV",
                data=csv,
                file_name="steam_review_analysis.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()