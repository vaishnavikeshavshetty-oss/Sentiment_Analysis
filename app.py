import streamlit as st
import pickle
import re
import os
import nltk
import gdown
from nltk.corpus import stopwords

# ---------------- NLTK SETUP ----------------
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
download_nltk_data()

# Try importing snscrape safely
try:
    import snscrape.modules.twitter as sntwitter
    SCRAPER_AVAILABLE = True
except:
    SCRAPER_AVAILABLE = False

# ---------------- DATASET ID ----------------
# REMOVE THIS SECTION if you don't actually use the dataset in the app
# Downloading 500MB will crash Streamlit Cloud free tier
# DATASET_ID = "1XWnoBq0Rtgl-_eBC-Ra_L_633jVmg0Nw"
# DATASET_FILE = "training.1600000.processed.noemoticon.csv"
# if not os.path.exists(DATASET_FILE):
# url = f"https://drive.google.com/uc?id={DATASET_ID}"
# gdown.download(url, DATASET_FILE, quiet=False)

# ---------------- STOPWORDS ----------------
@st.cache_resource
def load_stopwords():
    return set(stopwords.words('english'))

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model_and_vectorizer():
    model_path = "model.pkl"
    vectorizer_path = "vectorizer.pkl"

    if not os.path.exists(model_path):
        st.error("❌ model.pkl not found. Upload it to your GitHub repo.")
        st.stop()

    if not os.path.exists(vectorizer_path):
        st.error("❌ vectorizer.pkl not found. Upload it to your GitHub repo.")
        st.stop()

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer

# ---------------- CLEAN TEXT ----------------
def clean_text(text, stop_words):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))
    words = text.lower().split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# ---------------- PREDICT ----------------
def predict_sentiment(text, model, vectorizer, stop_words):
    cleaned = clean_text(text, stop_words)
    if cleaned.strip() == "":
        return "Neutral"
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)
    return "Negative" if prediction[0] == 0 else "Positive"

# ---------------- FETCH TWEETS ----------------
def fetch_tweets(username, num_tweets):
    tweets = []
    if not SCRAPER_AVAILABLE:
        st.error("❌ snscrape not installed or failed to import. Twitter feature disabled.")
        return tweets
    try:
        scraper = sntwitter.TwitterUserScraper(username)
        for i, tweet in enumerate(scraper.get_items()):
            if i >= num_tweets:
                break
            if tweet.content and tweet.lang == "en":
                tweets.append(tweet.content)
    except Exception as e:
        st.error(f"⚠️ Twitter scraping failed: {e}")
        return []
    return tweets

# ---------------- MAIN APP ----------------
def main():
    st.set_page_config(page_title="Twitter Sentiment Analysis", layout="centered")
    st.title("🐦 Twitter Sentiment Analysis")

    st.sidebar.title("📁 Project Info")
    st.sidebar.info("Sentiment Analysis using ML + Streamlit")

    # Load resources
    try:
        stop_words = load_stopwords()
        model, vectorizer = load_model_and_vectorizer()
    except Exception as e:
        st.error(f"Startup Error: {e}")
        return

    option = st.selectbox("Choose an option", ["Input text", "Get Tweets from User"])

    # ---------------- TEXT INPUT ----------------
    if option == "Input text":
        text_input = st.text_area("Enter text")
        if st.button("Analyze"):
            if text_input.strip() == "":
                st.warning("⚠️ Please enter text")
            else:
                sentiment = predict_sentiment(text_input, model, vectorizer, stop_words)
                st.success(f"Sentiment: {sentiment}")

    # ---------------- TWITTER INPUT ----------------
    elif option == "Get Tweets from User":
        if not SCRAPER_AVAILABLE:
            st.warning("Twitter scraping is disabled because snscrape failed to load on Streamlit Cloud.")
        else:
            username = st.text_input("Enter Twitter Username")
            num_tweets = st.slider("Number of tweets", 1, 20, 5) # Reduced max to avoid timeouts
            
            if st.button("Fetch & Analyze"):
                if username.strip() == "":
                    st.warning("⚠️ Enter username")
                else:
                    with st.spinner("Fetching tweets..."):
                        tweets = fetch_tweets(username, num_tweets)

                    if len(tweets) == 0:
                        st.warning("No tweets found or scraping failed")
                    else:
                        positive = negative = neutral = 0
                        st.subheader("📊 Results")
                        for tweet in tweets:
                            sentiment = predict_sentiment(tweet, model, vectorizer, stop_words)
                            if sentiment == "Positive": positive += 1
                            elif sentiment == "Negative": negative += 1
                            else: neutral += 1
                            st.write(f"📝 {tweet}")
                            st.write(f"👉 Sentiment: {sentiment}")
                            st.write("---")

                        st.subheader("📈 Summary")
                        st.write(f"✅ Positive: {positive}")
                        st.write(f"❌ Negative: {negative}")
                        st.write(f"➖ Neutral: {neutral}")

if __name__ == "__main__":
    main()
