import streamlit as st
import pandas as pd
import praw
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from autocorrect import Speller
import google.generativeai as genai
import textwrap
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Reddit API client
reddit = praw.Reddit(client_id='fpqm-JgqdYpiAmZodSh8Pw',
                     client_secret='LCrn_D_tPMfnl_uj3pxVSXz_gWjjZw',
                     redirect_uri="http://localhost:8080",
                     user_agent='gojoinfinity1')

# Load VADER Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Configure genai
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Helper function to format text as Markdown
def to_markdown(text):
    text = text.replace('•', '  *')
    return textwrap.indent(text, '> ', predicate=lambda _: True)

# Function to fetch response from Gemini API
def get_gemini_response(question):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(question)
    return response.text

# Function to perform Auto Text Correction
def auto_text_correction():
    st.title("Auto Text Correction")
    spell = Speller(lang='en')
    user_input = st.text_input("Enter your text:")
    if st.button("Correct Text"):
        corrected_text = spell(user_input)
        st.write("Corrected Text:", corrected_text)

# Function to perform Reddit Sentiment Analysis
def reddit_sentiment_analysis():
    st.title('Reddit Sentiment Analysis')
    emojis = {'positive': '😊', 'negative': '😔', 'neutral': '😐'}
    user_input = st.text_area("Enter your text here:")
    if st.button('Predict Sentiment'):
        sentiment = predict_sentiment_vader(user_input)
        emoji = emojis[sentiment]
        st.write("Predicted sentiment:", sentiment, emoji)

# Function to predict sentiment using VADER
def predict_sentiment_vader(text):
    # Preprocess the text
    processed_text = text
    # Analyze sentiment using VADER
    scores = analyzer.polarity_scores(processed_text)
    # Determine sentiment label based on compound score
    if scores['compound'] >= 0.05:
        return 'positive'
    elif scores['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Streamlit interface for Auto Text Generation
def auto_text_generation():
    st.title("Auto Text Generation")
    input_text = st.text_input("Input: ", key="input")
    submit = st.button("Ask the question")
    if submit:
        response = get_gemini_response(input_text)
        st.subheader("The Response is")
        st.write(response)

# Streamlit interface for Reddit Data Fetcher
def reddit_data_fetcher():
    st.title('Reddit Data Fetcher')
    subreddit_input = st.text_input("Enter the subreddit(s) you want to fetch data from (comma-separated): ")
    if st.button('Fetch Data'):
        subreddits = [sub.strip() for sub in subreddit_input.split(',')]
        if subreddits:
            posts = []
            for subreddit_name in subreddits:
                subreddit = reddit.subreddit(subreddit_name)
                for post in subreddit.top(time_filter="all", limit=100):
                    posts.append({
                        'post_id': post.id,
                        'subreddit': post.subreddit.display_name,
                        'created_utc': post.created_utc,
                        'selftext': post.selftext,
                        'post_url': post.url,
                        'post_title': post.title,
                        'link_flair_text': post.link_flair_text,
                        'score': post.score,
                        'num_comments': post.num_comments,
                        'upvote_ratio': post.upvote_ratio
                    })
            df = pd.DataFrame(posts)
            st.write("Data Fetched Successfully!")
            st.dataframe(df)  # Display the dataframe in the app
        else:
            st.write("Please enter at least one subreddit.")

# Main function to run the Streamlit app
def main():
    st.set_page_config(page_title="Q&A Demo")
    
    reddit_data_fetcher()  # Reddit Data Fetcher at the top
    
    auto_text_correction()
    reddit_sentiment_analysis()
    auto_text_generation()

# Run the app
if __name__ == "__main__":
    main()

