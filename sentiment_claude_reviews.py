import streamlit as st
import pandas as pd
import plotly.express as px
import datetime
import requests
import json
import time
import os
import re
import numpy as np
from dotenv import load_dotenv

# Load environment variables for API keys
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Steam Review Analyzer with Claude",
    page_icon="🎮",
    layout="wide"
)

# Main title
st.title("Steam Review Analyzer with Claude API")
st.markdown("Analyze Steam reviews using Claude's advanced language capabilities.")


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


# Function to analyze a review using Claude API
def analyze_with_claude(review_text, api_key):
    """
    Sends a review to Claude API and gets nuanced sentiment analysis and extracted desires
    """
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01"
    }

    prompt = f"""
You are analyzing a Steam game review to understand the underlying user sentiment and desires.

Please analyze the following review text: "{review_text}"

Your task is to:

1. Identify the underlying sentiment - not just positive/negative/neutral, but the emotional tone and attitude of the user. Categorize this into a concise sentiment token like "excited", "disappointed", "frustrated", "satisfied", "impressed", "hopeful", "mixed", "cautiously_optimistic", etc.

2. Extract what the user wants from the game. This could be explicit requests or implied desires. Categorize each desire with a concise token like "more_content", "bug_fixes", "better_performance", "improved_mechanics", "balance_changes", "easier_learning_curve", "multiplayer_features", "ui_improvements", "price_adjustment", etc.

3. Determine the key context that frames the user's experience (e.g., "new_player", "veteran", "returning_player", "competitive", "casual", "achievement_hunter", etc.)

Please respond in JSON format with these keys:
- sentiment_token (string): One concise token representing the main emotional tone
- sentiment_description (string): Brief explanation of the sentiment (1-2 sentences)
- desire_tokens (array of strings): List of 1-5 concise tokens representing what the user wants
- desire_description (string): Brief explanation of these desires (1-2 sentences)
- context_token (string): One token representing the user's context or relationship with the game
- context_description (string): Brief explanation of this context (1 sentence)

JSON Response:
"""

    data = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 1000,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=data
        )

        if response.status_code == 200:
            # Extract and parse the response
            response_json = response.json()
            content = response_json.get('content', [])
            if content and len(content) > 0:
                text_content = content[0].get('text', '')

                # Find JSON block (could be within ```json``` or plain)
                if "```json" in text_content:
                    json_str = text_content.split("```json")[1].split("```")[0].strip()
                else:
                    # Try to extract JSON from the raw text
                    json_str = text_content.strip()

                # Parse the JSON response
                result = json.loads(json_str)
                return result

        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return {
                "sentiment_token": "error",
                "sentiment_description": f"API Error: {response.status_code}",
                "desire_tokens": [],
                "desire_description": "Could not extract desires due to API error",
                "context_token": "error",
                "context_description": "Could not determine context due to API error"
            }
    except Exception as e:
        st.error(f"Error calling Claude API: {str(e)}")
        return {
            "sentiment_token": "error",
            "sentiment_description": f"Error: {str(e)}",
            "desire_tokens": [],
            "desire_description": "Could not extract desires due to error",
            "context_token": "error",
            "context_description": "Could not determine context due to error"
        }


# Main function
def main():
    # Sidebar for API key
    st.sidebar.header("Claude API Configuration")

    # API key input
    api_key = st.sidebar.text_input(
        "Enter Claude API Key",
        type="password",
        value=os.getenv("ANTHROPIC_API_KEY", "")
    )

    # Sample size
    sample_size = st.sidebar.number_input(
        "Number of reviews to analyze (sample size)",
        min_value=1,
        max_value=1000,
        value=20
    )

    # Analysis settings
    st.sidebar.header("Analysis Settings")
    recency_filter = st.sidebar.multiselect(
        "Filter reviews by recency",
        options=["fresh", "monthly", "last year", "older than a year", "unknown"],
        default=["fresh"]
    )

    # Quantile filter for weighted review score
    st.sidebar.header("Quantile Filter")
    quantile_range = st.sidebar.slider(
        "Select quantile range for weighted review score:",
        min_value=0.0,
        max_value=1.0,
        value=(0.0, 1.0),
        step=0.01)

    # File upload
    uploaded_file = st.file_uploader("Upload Steam Reviews CSV", type="csv")

    if not uploaded_file:
        return

    # Load data
    df = pd.read_csv(uploaded_file)
    st.write(f"Loaded {len(df)} reviews")

    # Auto-detect columns
    # Review column detection
    review_col = None
    if 'review' in df.columns:
        review_col = 'review'
    else:
        # Fall back to broader matching
        for col in df.columns:
            col_lower = col.lower()
            if 'review' in col_lower and not any(x in col_lower for x in ['number', 'count', 'score', 'id', 'url']):
                review_col = col
                break

    if not review_col:
        review_col = st.selectbox("Select the column containing review text:", df.columns)
    else:
        st.success(f"Detected review column: {review_col}")

    # Timestamp column detection
    timestamp_col = None
    for col in df.columns:
        if 'timestamp' in col.lower() or 'last_played' in col.lower():
            timestamp_col = col
            break

    if not timestamp_col:
        timestamp_col = st.selectbox("Select the column containing last played timestamp:", df.columns)
    else:
        st.success(f"Detected timestamp column: {timestamp_col}")

    # Calculate quantiles for weighted_review_score column
    if "weighted_review_score" in df.columns:
        lower_quantile, upper_quantile = quantile_range
        lower_bound = np.quantile(df['weighted_review_score'].dropna(), lower_quantile)
        upper_bound = np.quantile(df['weighted_review_score'].dropna(), upper_quantile)

        filtered_df = df[
            (df['weighted_review_score'] >= lower_bound) &
            (df['weighted_review_score'] <= upper_bound)
            ]
        st.write(
            f"Filtered {len(filtered_df)} of {len(df)} by quantile range [{lower_quantile:.2f}, {upper_quantile:.2f}] with scores between {lower_bound:.2f} and {upper_bound:.2f}")
    else:
        st.warning("Column 'weighted_review_score' not found in the uploaded dataset.")
        filtered_df = df.copy()

    # Process recency
    df['recency'] = filtered_df[timestamp_col].apply(classify_recency)

    # Filter by recency
    filtered_df = df[df['recency'].isin(recency_filter)]
    st.write(f"Found {len(filtered_df)} reviews matching the recency filter: {', '.join(recency_filter)}")

    # Show recency distribution
    recency_counts = filtered_df['recency'].value_counts()

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

    # Take a uniform sample
    if len(filtered_df) > sample_size:
        sampled_df = filtered_df.sample(n=sample_size, random_state=42)
        st.write(f"Analyzing a random sample of {sample_size} reviews")
    else:
        sampled_df = filtered_df
        st.write(f"Analyzing all {len(sampled_df)} filtered reviews")

    # Show sample reviews
    with st.expander("View sample reviews to be analyzed"):
        for i, (_, row) in enumerate(sampled_df.head(5).iterrows()):
            text = row[review_col]
            preview = text[:200] + "..." if isinstance(text, str) and len(text) > 200 else text
            st.text_area(f"Review {i + 1}", value=preview, height=100)

    # Process reviews with Claude
    if st.button("Analyze with Claude API"):
        # Initialize session state for storing results if it doesn't exist
        if 'claude_results' not in st.session_state:
            st.session_state.claude_results = None

        progress_bar = st.progress(0)
        status_text = st.empty()

        # Create a new dataframe for results
        results = []

        # Call Claude API
        if not api_key:
            return

        # Process each review
        for i, (idx, row) in enumerate(sampled_df.iterrows()):
            # Update progress
            progress = (i + 1) / len(sampled_df)
            progress_bar.progress(progress)
            status_text.text(f"Processing review {i + 1} of {len(sampled_df)}...")

            # Get review text
            review_text = row[review_col]

            # Skip empty reviews
            if not isinstance(review_text, str) or review_text.strip() == "":
                continue

            # Call Claude API
            analysis = analyze_with_claude(review_text, api_key)

            # Add to results
            result = {
                'original_index': idx,
                'review': review_text,
                'recency': row['recency'],
                **analysis
            }

            # Add any other relevant columns from original data
            for col in df.columns:
                if col not in [review_col, 'recency']:
                    result[col] = row[col]

            results.append(result)

            # Add a small delay to avoid hitting rate limits
            time.sleep(0.5)

        # Create results dataframe
        results_df = pd.DataFrame(results)

        # Save to session state
        st.session_state.claude_results = results_df

        # Complete progress
        progress_bar.progress(1.0)
        status_text.text("Analysis complete!")

    # Display results if available
    if st.session_state.get('claude_results') is None:
        return

    results_df = st.session_state.claude_results

    st.header("Analysis Results")

    # Sentiment distribution
    sentiment_counts = results_df['sentiment'].value_counts()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sentiment Distribution")

        fig = px.pie(
            names=sentiment_counts.index,
            values=sentiment_counts.values,
            title="Sentiment Distribution (Claude Analysis)",
            color=sentiment_counts.index,
            color_discrete_map={'positive': 'green', 'neutral': 'gray', 'negative': 'red',
                                'unknown': '#CCCCCC'},
            hole=0.4
        )
        st.plotly_chart(fig)

    with col2:
        st.subheader("Recommendations Found")

        # Count reviews with recommendations
        has_recommendations = results_df['recommendations'].apply(lambda x: isinstance(x, list) and len(x) > 0)
        rec_count = has_recommendations.sum()
        no_rec_count = len(results_df) - rec_count

        rec_data = pd.DataFrame({
            'Category': ['With Recommendations', 'Without Recommendations'],
            'Count': [rec_count, no_rec_count]
        })

        fig = px.pie(
            rec_data,
            names='Category',
            values='Count',
            title="Reviews With Recommendations",
            color='Category',
            color_discrete_map={'With Recommendations': 'blue', 'Without Recommendations': 'lightgray'},
            hole=0.4
        )
        st.plotly_chart(fig)

    # Sentiment by recency
    st.subheader("Sentiment by Play Recency")

    sentiment_recency = pd.crosstab(results_df['recency'], results_df['sentiment'])

    fig = px.bar(
        sentiment_recency.reset_index().melt(id_vars='recency'),
        x='recency',
        y='value',
        color='sentiment',
        title="Sentiment Distribution by Play Recency",
        labels={'value': 'Count', 'recency': 'Play Recency'},
        color_discrete_map={'positive': 'green', 'neutral': 'gray', 'negative': 'red', 'unknown': '#CCCCCC'},
        barmode='group'
    )
    st.plotly_chart(fig)

    # Show all recommendations
    st.header("Extracted Recommendations")

    all_recommendations = []
    for idx, row in results_df.iterrows():
        if isinstance(row['recommendations'], list):
            for rec in row['recommendations']:
                all_recommendations.append({
                    'recommendation': rec,
                    'sentiment': row['sentiment'],
                    'recency': row['recency'],
                    'review_index': idx
                })

    rec_df = pd.DataFrame(all_recommendations)

    if len(rec_df) > 0:
        st.write(f"Found {len(rec_df)} specific recommendations across {rec_count} reviews")

        # Group similar recommendations (simple grouping by exact matches)
        rec_counts = rec_df['recommendation'].value_counts().reset_index()
        rec_counts.columns = ['Recommendation', 'Count']

        # Show top recommendations
        st.subheader("Top Recommendations")

        # We'll only show the most frequent ones in a table
        top_recs = rec_counts.head(15)
        st.table(top_recs)

        # Show all recommendations by category
        st.subheader("Browse Recommendations")

        # Let user filter by sentiment or recency
        col1, col2 = st.columns(2)

        with col1:
            sentiment_filter = st.selectbox(
                "Filter by sentiment:",
                options=["All"] + list(rec_df['sentiment'].unique())
            )

        with col2:
            recency_filter_display = st.selectbox(
                "Filter by recency:",
                options=["All"] + list(rec_df['recency'].unique())
            )

        # Apply filters
        filtered_recs = rec_df.copy()
        if sentiment_filter != "All":
            filtered_recs = filtered_recs[filtered_recs['sentiment'] == sentiment_filter]

        if recency_filter_display != "All":
            filtered_recs = filtered_recs[filtered_recs['recency'] == recency_filter_display]

        st.write(f"Showing {len(filtered_recs)} recommendations")

        # Display recommendations with context
        for i, (idx, row) in enumerate(filtered_recs.head(20).iterrows()):
            with st.expander(f"Recommendation {i + 1} ({row['sentiment']} review, {row['recency']})"):
                st.write(f"**Recommendation:** {row['recommendation']}")

                # Get the original review
                original_review_idx = row['review_index']
                original_review = results_df.iloc[original_review_idx]['review']

                # Display preview of the review
                preview = original_review[:300] + "..." if len(original_review) > 300 else original_review
                st.text_area("From review:", value=preview, height=100)
        else:
            st.write("No specific recommendations found in the analyzed reviews.")

        # Browse reviews by sentiment
        st.header("Browse Reviews by Sentiment")

        sentiment_tabs = st.tabs(["Positive", "Neutral", "Negative", "Unknown"])

        for i, sentiment in enumerate(["positive", "neutral", "negative", "unknown"]):
            with sentiment_tabs[i]:
                filtered_reviews = results_df[results_df['sentiment'] == sentiment]

                if len(filtered_reviews) > 0:
                    st.write(f"Found {len(filtered_reviews)} {sentiment} reviews")

                    for j, (idx, row) in enumerate(filtered_reviews.head(5).iterrows()):
                        with st.expander(f"{sentiment.title()} Review {j + 1} ({row['recency']})"):
                            st.text_area("Review text:", value=row['review'], height=150)

                            if isinstance(row['recommendations'], list) and len(row['recommendations']) > 0:
                                st.write("**Recommendations:**")
                                for rec in row['recommendations']:
                                    st.write(f"- {rec}")
                            else:
                                st.write("No specific recommendations found.")
                else:
                    st.write(f"No {sentiment} reviews found.")

        # Download results
        st.header("Download Results")

        # Prepare download data
        download_df = results_df.copy()

        # Convert recommendation lists to strings for CSV compatibility
        download_df['recommendations_text'] = download_df['recommendations'].apply(
            lambda x: "; ".join(x) if isinstance(x, list) and len(x) > 0 else ""
        )

        # Drop the original list column
        download_df = download_df.drop(columns=['recommendations'])

        # Convert to CSV
        csv = download_df.to_csv(index=False)

        st.download_button(
            label="Download analyzed data as CSV",
            data=csv,
            file_name="claude_review_analysis.csv",
            mime="text/csv"
        )

        # Also offer to download just the recommendations
        if len(rec_df) > 0:
            rec_csv = rec_df.to_csv(index=False)

            st.download_button(
                label="Download recommendations as CSV",
                data=rec_csv,
                file_name="recommendations.csv",
                mime="text/csv",
                key="rec_download"
            )

    elif not api_key and uploaded_file:
        st.warning("Please enter your Claude API key in the sidebar to analyze reviews.")
    elif not uploaded_file:
        st.info("Please upload a Steam reviews CSV file to begin analysis.")


if __name__ == "__main__":
    main()