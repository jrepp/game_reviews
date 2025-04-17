import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud

# Set page configuration
st.set_page_config(
    page_title="Predecessor Review Analysis",
    page_title_icon="🎮",
    layout="wide"
)

# Main title
st.title("Predecessor Game Review Analysis")
st.markdown("This tool analyzes player feedback from Predecessor reviews to extract actionable insights for game improvement.")

# Download NLTK resources silently if needed
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    with st.spinner('Downloading required resources...'):
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)

# Function to load and preprocess data
@st.cache_data
def load_data(file):
    # Load CSV file
    df = pd.read_csv(file)
    
    # Convert playtime to numeric
    df['author_minutes_playtime_last_two_weeks'] = pd.to_numeric(df['author_minutes_playtime_last_two_weeks'], errors='coerce').fillna(0)
    
    # Add segment column
    df['player_segment'] = pd.cut(
        df['author_minutes_playtime_last_two_weeks'],
        bins=[0, 120, 600, float('inf')],
        labels=['Casual', 'Regular', 'Dedicated']
    )
    
    return df

# Function to analyze sentiments
@st.cache_data
def analyze_sentiment(text):
    # Simple sentiment analysis using keyword lists
    if not isinstance(text, str):
        return 'neutral'
        
    text = text.lower()
    
    positive_words = ['love', 'great', 'good', 'excellent', 'amazing', 'awesome', 'fun', 'enjoy', 
                     'best', 'fantastic', 'perfect', 'recommend', 'beautiful', 'smooth', 'polished']
    
    negative_words = ['hate', 'bad', 'terrible', 'awful', 'poor', 'broken', 'disappointing', 'worst', 
                     'boring', 'unplayable', 'waste', 'refund', 'buggy', 'crash', 'expensive', 'lag']
    
    positive_count = sum(1 for word in positive_words if word in text)
    negative_count = sum(1 for word in negative_words if word in text)
    
    if positive_count > negative_count:
        return 'positive'
    elif negative_count > positive_count:
        return 'negative'
    else:
        return 'neutral'

# Function to categorize feedback
@st.cache_data
def categorize_feedback(text):
    if not isinstance(text, str):
        return {}
        
    text = text.lower()
    
    categories = {
        'Core gameplay mechanics': ['gameplay', 'mechanics', 'controls', 'abilities', 'heroes', 'characters', 
                                  'balance', 'combat', 'skills', 'movement', 'jungle', 'lanes', 'moba'],
        
        'Progression and rewards': ['progression', 'rewards', 'levels', 'unlocks', 'achievements', 
                                  'experience', 'xp', 'coins', 'currency', 'grind', 'battlepass'],
        
        'Technical performance': ['performance', 'fps', 'lag', 'crash', 'bug', 'glitch', 'optimization', 
                               'servers', 'ping', 'stuttering', 'loading', 'freeze', 'disconnect'],
        
        'Matchmaking/multiplayer': ['matchmaking', 'teams', 'teammates', 'queue', 'waiting', 'matching', 
                                  'ranked', 'casual', 'competitive', 'players', 'mmr', 'skill'],
        
        'Monetization/value perception': ['price', 'cost', 'expensive', 'value', 'microtransactions', 
                                        'purchases', 'skins', 'cosmetics', 'free', 'pay', 'money', 'shop'],
        
        'Content depth and variety': ['content', 'variety', 'maps', 'modes', 'depth', 'replayability', 
                                    'boredom', 'diversity', 'options', 'choices', 'limited', 'more']
    }
    
    results = {}
    for category, keywords in categories.items():
        results[category] = any(keyword in text for keyword in keywords)
    
    return results

# Function to extract key phrases
@st.cache_data
def extract_key_phrases(text, max_phrases=3):
    if not isinstance(text, str):
        return []
        
    # Split text into sentences
    sentences = re.split(r'[.!?]+', text)
    
    # Keywords for identifying important sentences
    important_keywords = [
        'need', 'should', 'could', 'would', 'improve', 'better', 'worst', 'best',
        'change', 'fix', 'issue', 'problem', 'feature', 'add', 'remove', 'update',
        'balance', 'gameplay', 'mechanic', 'performance', 'bug', 'matchmaking'
    ]
    
    # Score sentences based on keywords
    scored_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 10:  # Skip very short sentences
            continue
            
        score = sum(1 for keyword in important_keywords if keyword in sentence.lower())
        scored_sentences.append((sentence, score))
    
    # Return top scoring sentences
    top_sentences = sorted(scored_sentences, key=lambda x: x[1], reverse=True)
    return [s[0] for s in top_sentences[:max_phrases] if s[1] > 0]

# Function for text preprocessing to find frequent words
@st.cache_data
def preprocess_text(text):
    if not isinstance(text, str):
        return []
        
    # Lowercase
    text = text.lower()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words and len(word) > 2]
    
    return tokens

# Function to generate wordcloud
def generate_wordcloud(text_series):
    all_text = ' '.join(text_series.dropna().astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(all_text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

# Function to find frequent improvement requests
@st.cache_data
def find_improvement_requests(reviews):
    improvement_phrases = ['should add', 'need to add', 'would be better if', 'could improve', 
                         'please add', 'fix', 'needs improvement', 'missing', 'lacks', 'issue with']
    
    requests = []
    for review in reviews:
        if not isinstance(review, str):
            continue
            
        review_lower = review.lower()
        for phrase in improvement_phrases:
            if phrase in review_lower:
                # Find the sentence containing the phrase
                sentences = re.split(r'[.!?]+', review)
                for sentence in sentences:
                    if phrase in sentence.lower():
                        requests.append(sentence.strip())
    
    return requests

# Main application
def main():
    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Upload Predecessor Reviews CSV", type="csv")
    
    if uploaded_file:
        # Load data
        with st.spinner('Loading and processing data...'):
            df = load_data(uploaded_file)
        
        # Display basic info
        st.sidebar.success(f"Loaded {len(df)} reviews")
        
        # Add sentiment analysis
        df['sentiment'] = df['review'].apply(analyze_sentiment)
        
        # Overview metrics
        st.header("Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Reviews", len(df))
        col2.metric("Positive Reviews", len(df[df['sentiment'] == 'positive']))
        col3.metric("Neutral Reviews", len(df[df['sentiment'] == 'neutral']))
        col4.metric("Negative Reviews", len(df[df['sentiment'] == 'negative']))
        
        # Player segments
        st.header("Player Segments")
        segment_counts = df['player_segment'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                names=segment_counts.index, 
                values=segment_counts.values,
                title="Player Distribution by Engagement Level",
                color=segment_counts.index,
                color_discrete_map={'Casual':'#FF9999', 'Regular':'#66B2FF', 'Dedicated':'#99FF99'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("Segment Descriptions")
            st.markdown("""
            - **Casual Players** (<2 hours playtime): New players or those trying the game
            - **Regular Players** (2-10 hours): Engaged players who play occasionally
            - **Dedicated Players** (>10 hours): Highly engaged, core playerbase
            """)
            
            sentiment_by_segment = pd.crosstab(df['player_segment'], df['sentiment'])
            st.dataframe(sentiment_by_segment)
        
        # Categorize reviews
        categories_list = []
        for idx, row in df.iterrows():
            categories = categorize_feedback(row['review'])
            for category, is_mentioned in categories.items():
                if is_mentioned:
                    categories_list.append({
                        'review_id': idx,
                        'player_segment': row['player_segment'],
                        'sentiment': row['sentiment'],
                        'category': category
                    })
        
        category_df = pd.DataFrame(categories_list)
        
        # Feedback Categories Analysis
        st.header("Feedback Categories Analysis")
        
        cat_tab1, cat_tab2 = st.tabs(["Overall Analysis", "Segment Analysis"])
        
        with cat_tab1:
            if len(category_df) > 0:
                category_counts = category_df['category'].value_counts().reset_index()
                category_counts.columns = ['Category', 'Count']
                
                fig = px.bar(
                    category_counts,
                    x='Count',
                    y='Category',
                    title="Most Mentioned Game Systems",
                    orientation='h'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Categories by sentiment
                cat_sentiment = pd.crosstab(category_df['category'], category_df['sentiment'])
                
                # Calculate percentages
                cat_sentiment_pct = cat_sentiment.div(cat_sentiment.sum(axis=1), axis=0) * 100
                
                fig = px.bar(
                    cat_sentiment_pct.reset_index().melt(id_vars='category'),
                    x='category',
                    y='value',
                    color='sentiment',
                    title="Sentiment Distribution by Category",
                    labels={'category': 'Category', 'value': 'Percentage', 'sentiment': 'Sentiment'},
                    color_discrete_map={'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No category data available. Make sure reviews contain relevant keywords.")
        
        with cat_tab2:
            if len(category_df) > 0:
                # Categories by segment
                segment_cat_counts = pd.crosstab(category_df['player_segment'], category_df['category'])
                
                # Normalize by segment size for fair comparison
                segment_sizes = df['player_segment'].value_counts()
                normalized_counts = segment_cat_counts.copy()
                
                for segment in segment_cat_counts.index:
                    normalized_counts.loc[segment] = segment_cat_counts.loc[segment] / segment_sizes[segment] * 100
                
                # Plot heatmap
                fig = px.imshow(
                    normalized_counts,
                    title="Category Importance by Player Segment (Normalized %)",
                    color_continuous_scale='Viridis',
                    labels=dict(x="Category", y="Player Segment", color="Percentage")
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No segment-category data available.")
        
        # Detailed Segment Analysis
        st.header("Detailed Segment Analysis")
        
        segment_tabs = st.tabs(["Casual Players", "Regular Players", "Dedicated Players"])
        
        for i, segment in enumerate(['Casual', 'Regular', 'Dedicated']):
            with segment_tabs[i]:
                segment_df = df[df['player_segment'] == segment]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader(f"{segment} Player Sentiment")
                    sentiment_counts = segment_df['sentiment'].value_counts()
                    fig = px.pie(
                        names=sentiment_counts.index,
                        values=sentiment_counts.values,
                        title=f"Sentiment Distribution for {segment} Players",
                        color=sentiment_counts.index,
                        color_discrete_map={'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader(f"{segment} Players Common Words")
                    
                    if len(segment_df) > 0:
                        # Process all reviews in this segment
                        all_tokens = []
                        for review in segment_df['review']:
                            all_tokens.extend(preprocess_text(review))
                        
                        # Get word frequencies
                        word_freq = Counter(all_tokens).most_common(20)
                        
                        if word_freq:
                            words_df = pd.DataFrame(word_freq, columns=['Word', 'Frequency'])
                            fig = px.bar(
                                words_df,
                                x='Frequency',
                                y='Word',
                                orientation='h',
                                title=f"Top 20 Words in {segment} Player Reviews"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Not enough text data for word frequency analysis.")
                    else:
                        st.info(f"No {segment} player reviews available.")
                
                # Extract pain points and appreciated features
                positive_reviews = segment_df[segment_df['sentiment'] == 'positive']['review']
                negative_reviews = segment_df[segment_df['sentiment'] == 'negative']['review']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Pain Points")
                    if len(negative_reviews) > 0:
                        # Find common issues
                        improvement_requests = find_improvement_requests(negative_reviews)
                        
                        if improvement_requests:
                            for i, request in enumerate(improvement_requests[:10]):
                                st.markdown(f"**{i+1}.** {request}")
                        else:
                            pain_points = []
                            for review in negative_reviews:
                                phrases = extract_key_phrases(review, max_phrases=1)
                                pain_points.extend(phrases)
                            
                            for i, point in enumerate(pain_points[:10]):
                                st.markdown(f"**{i+1}.** {point}")
                    else:
                        st.info(f"No negative reviews from {segment} players.")
                
                with col2:
                    st.subheader("Appreciated Features")
                    if len(positive_reviews) > 0:
                        appreciated_features = []
                        for review in positive_reviews:
                            phrases = extract_key_phrases(review, max_phrases=1)
                            appreciated_features.extend(phrases)
                        
                        for i, feature in enumerate(appreciated_features[:10]):
                            st.markdown(f"**{i+1}.** {feature}")
                    else:
                        st.info(f"No positive reviews from {segment} players.")
        
        # Recommendations
        st.header("Actionable Recommendations")
        
        # Prepare data for recommendations
        if len(category_df) > 0:
            # Get problematic categories (more negative than positive sentiment)
            category_sentiment = pd.crosstab(category_df['category'], category_df['sentiment'])
            
            # Categories with the most mentions
            top_categories = category_df['category'].value_counts().head(3).index.tolist()
            
            # Categories with highest negative ratio
            if 'negative' in category_sentiment.columns:
                category_sentiment['neg_ratio'] = category_sentiment['negative'] / category_sentiment.sum(axis=1)
                problem_categories = category_sentiment.sort_values('neg_ratio', ascending=False).head(3).index.tolist()
            else:
                problem_categories = []
            
            # Combine and deduplicate
            recommendation_categories = list(dict.fromkeys(top_categories + problem_categories))
            
            # Generate recommendations
            for category in recommendation_categories:
                st.subheader(f"Improving {category}")
                
                # Current state
                category_reviews = df[df.index.isin(category_df[category_df['category'] == category]['review_id'])]
                
                # Get positive and negative reviews for this category
                pos_reviews = category_reviews[category_reviews['sentiment'] == 'positive']['review']
                neg_reviews = category_reviews[category_reviews['sentiment'] == 'negative']['review']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Current State:**")
                    
                    # Calculate sentiment ratio
                    total = len(category_reviews)
                    if total > 0:
                        pos_percent = len(pos_reviews) / total * 100
                        neg_percent = len(neg_reviews) / total * 100
                        
                        st.markdown(f"- Mentioned in {total} reviews")
                        st.markdown(f"- Sentiment: {pos_percent:.1f}% Positive, {neg_percent:.1f}% Negative")
                    
                    # Extract key issues
                    if len(neg_reviews) > 0:
                        issues = []
                        for review in neg_reviews:
                            phrases = extract_key_phrases(review, max_phrases=1)
                            issues.extend(phrases)
                        
                        if issues:
                            st.markdown("**Key Issues:**")
                            for issue in issues[:3]:
                                st.markdown(f"- {issue}")
                
                with col2:
                    st.markdown("**Recommendations:**")
                    
                    # Generate generic recommendations based on category
                    if category == 'Core gameplay mechanics':
                        st.markdown("""
                        1. **Combat Balance**: Review and adjust hero abilities for better balance
                        2. **Controls Improvement**: Refine movement and responsiveness
                        3. **Gameplay Clarity**: Improve visual feedback for abilities and effects
                        """)
                    elif category == 'Progression and rewards':
                        st.markdown("""
                        1. **Reward Frequency**: Increase rewards for regular play
                        2. **Progression System**: Add more meaningful progression milestones
                        3. **Achievement System**: Expand the achievement system with varied goals
                        """)
                    elif category == 'Technical performance':
                        st.markdown("""
                        1. **Optimization**: Focus on improving FPS stability on mid-range PCs
                        2. **Server Stability**: Improve connection reliability and reduce lag
                        3. **Loading Times**: Reduce loading screen durations
                        """)
                    elif category == 'Matchmaking/multiplayer':
                        st.markdown("""
                        1. **Matchmaking Algorithm**: Refine skill-based matchmaking
                        2. **Queue Times**: Implement systems to reduce waiting times
                        3. **Team Balance**: Ensure teams have balanced skill levels
                        """)
                    elif category == 'Monetization/value perception':
                        st.markdown("""
                        1. **Value Bundles**: Create better-value starter packages
                        2. **Cosmetic Pricing**: Review pricing strategy for cosmetic items
                        3. **Free Rewards**: Increase free rewards to improve perception
                        """)
                    elif category == 'Content depth and variety':
                        st.markdown("""
                        1. **Map Variety**: Add more maps or map variations
                        2. **Game Modes**: Introduce casual/alternative game modes
                        3. **Character Roster**: Accelerate new character releases
                        """)
        else:
            st.info("Not enough data to generate recommendations.")
        
        # Priority matrix
        st.header("Recommendation Priority Matrix")
        
        # Create sample priority data
        if len(category_df) > 0:
            priority_data = []
            
            for category in category_df['category'].unique():
                # Calculate metrics
                mentions = len(category_df[category_df['category'] == category])
                segments_count = len(category_df[category_df['category'] == category]['player_segment'].unique())
                
                # Estimate retention impact (higher for core gameplay and technical issues)
                retention_impact = 3  # Medium by default
                if category in ['Core gameplay mechanics', 'Technical performance']:
                    retention_impact = 5  # High
                elif category in ['Progression and rewards', 'Matchmaking/multiplayer']:
                    retention_impact = 4  # Medium-high
                
                # Estimate implementation complexity (higher for core systems)
                complexity = 3  # Medium by default
                if category in ['Core gameplay mechanics', 'Technical performance']:
                    complexity = 4  # More complex
                elif category in ['Monetization/value perception', 'Content depth and variety']:
                    complexity = 2  # Less complex
                
                # Calculate priority score
                priority_score = (mentions / 10) + (segments_count * 2) + retention_impact - (complexity / 2)
                
                priority_data.append({
                    'Category': category,
                    'Mentions': mentions,
                    'Cross-segment': segments_count,
                    'Retention Impact': retention_impact,
                    'Implementation Complexity': complexity,
                    'Priority Score': priority_score
                })
            
            priority_df = pd.DataFrame(priority_data).sort_values('Priority Score', ascending=False)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Create bubble chart
                fig = px.scatter(
                    priority_df,
                    x='Implementation Complexity',
                    y='Retention Impact',
                    size='Mentions',
                    color='Priority Score',
                    hover_name='Category',
                    text='Category',
                    title="Recommendation Priority Matrix",
                    color_continuous_scale='viridis',
                    size_max=60
                )
                
                fig.update_layout(
                    xaxis=dict(
                        title='Implementation Complexity',
                        tickvals=[1, 2, 3, 4, 5],
                        ticktext=['Very Easy', 'Easy', 'Medium', 'Hard', 'Very Hard'],
                        range=[0.5, 5.5]
                    ),
                    yaxis=dict(
                        title='Retention Impact',
                        tickvals=[1, 2, 3, 4, 5],
                        ticktext=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
                        range=[0.5, 5.5]
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Priority Ranking")
                
                # Display priority table
                st.dataframe(
                    priority_df[['Category', 'Priority Score']].reset_index(drop=True),
                    use_container_width=True
                )
                
                st.markdown("""
                **Priority Score Formula:**
                - Mentions ÷ 10
                - + Cross-segment presence × 2
                - + Retention Impact
                - - Implementation Complexity ÷ 2
                """)
    
    else:
        # Show instructions if no file uploaded
        st.info("Please upload the Predecessor reviews CSV file to start the analysis.")
        
        st.markdown("""
        ### How to Use This Tool
        
        1. Upload the Predecessor reviews CSV file using the sidebar
        2. The tool will automatically:
           - Segment players by engagement level
           - Analyze sentiment in reviews
           - Categorize feedback by game system
           - Extract key pain points and appreciated features
           - Generate actionable recommendations
           - Prioritize recommendations based on impact and feasibility
        
        ### Analysis Methodology
        
        - **Text Analysis**: Natural language processing to categorize feedback
        - **Sentiment Analysis**: Keyword-based positive/negative sentiment detection
        - **Feedback Categorization**: Classification of reviews into game systems
        - **Recommendation Generation**: Data-driven insights for each category
        - **Priority Matrix**: Visualization of recommendation priorities
        """)

if __name__ == "__main__":
    main()
