import streamlit as st
import pickle
import re
import os
import nltk
from nltk.corpus import stopwords

# Try importing snscrape safely
try:
    import snscrape.modules.twitter as sntwitter
    SCRAPER_AVAILABLE = True
except:
    SCRAPER_AVAILABLE = False

# ---------------- STOPWORDS ----------------
@st.cache_resource
def load_stopwords():
    try:
        return set(stopwords.words('english'))
    except:
        nltk.download('stopwords')
        return set(stopwords.words('english'))

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model_and_vectorizer():
    base_path = os.path.dirname(__file__)

    model_path = os.path.join(base_path, "model.pkl")
    vectorizer_path = os.path.join(base_path, "vectorizer.pkl")

    if not os.path.exists(model_path):
        st.error("❌ model.pkl not found in project folder")
        st.stop()

    if not os.path.exists(vectorizer_path):
        st.error("❌ vectorizer.pkl not found in project folder")
        st.stop()

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)

    return model, vectorizer

# ---------------- CLEAN TEXT ----------------
def clean_text(text, stop_words):
    text = re.sub('[^a-zA-Z]', ' ', str(text))
    words = text.lower().split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# ---------------- PREDICT ----------------
def predict_sentiment(text, model, vectorizer, stop_words):
    cleaned = clean_text(text, stop_words)

    if cleaned.strip() == "":
        return "Neutral"

    vector = vectorizer.transform([cleaned])
    sentiment = model.predict(vector)

    return "Negative" if sentiment[0] == 0 else "Positive"

# ---------------- FETCH TWEETS ----------------
def fetch_tweets(username, num_tweets):
    tweets = []

    if not SCRAPER_AVAILABLE:
        st.error("snscrape not installed. Run: pip install snscrape")
        return tweets

    try:
        scraper = sntwitter.TwitterUserScraper(username)

        for i, tweet in enumerate(scraper.get_items()):
            if i >= num_tweets:
                break

            if tweet.content and tweet.lang == "en":
                tweets.append(tweet.content)

    except Exception:
        st.error("⚠️ Twitter scraping failed. Try another username.")
        return []

    return tweets

# ---------------- MAIN ----------------
def main():
    st.set_page_config(page_title="Sentiment Analyzer", layout="centered")

    st.title("🐦 Twitter Sentiment Analysis")

    # Load resources safely
    try:
        stop_words = load_stopwords()
        model, vectorizer = load_model_and_vectorizer()
    except Exception as e:
        st.error(f"Startup Error: {e}")
        return

    option = st.selectbox(
        "Choose an option",
        ["Input text", "Get Tweets from User"]
    )

    # -------- OPTION 1 --------
    if option == "Input text":
        text_input = st.text_area("Enter text")

        if st.button("Analyze", key="analyze_btn"):
            if text_input.strip() == "":
                st.warning("⚠️ Please enter some text")
            else:
                sentiment = predict_sentiment(
                    text_input, model, vectorizer, stop_words
                )
                st.success(f"Sentiment: {sentiment}")

    # -------- OPTION 2 --------
    elif option == "Get Tweets from User":
        username = st.text_input("Enter Twitter Username (without @)")
        num_tweets = st.slider("Number of tweets", 1, 50, 10)

        if st.button("Fetch & Analyze", key="fetch_btn"):
            if username.strip() == "":
                st.warning("⚠️ Please enter a username")
            else:
                with st.spinner("Fetching tweets..."):
                    tweets = fetch_tweets(username, num_tweets)

                if len(tweets) == 0:
                    st.warning("No tweets found or scraping blocked.")
                else:
                    positive, negative, neutral = 0, 0, 0

                    st.subheader("📊 Results")

                    for t in tweets:
                        sentiment = predict_sentiment(
                            t, model, vectorizer, stop_words
                        )

                        if sentiment == "Positive":
                            positive += 1
                        elif sentiment == "Negative":
                            negative += 1
                        else:
                            neutral += 1

                        st.write(f"📝 {t}")
                        st.write(f"👉 Sentiment: {sentiment}")
                        st.write("---")

                    # -------- SUMMARY --------
                    st.subheader("📈 Summary")
                    st.write(f"✅ Positive: {positive}")
                    st.write(f"❌ Negative: {negative}")
                    st.write(f"➖ Neutral: {neutral}")

# ---------------- RUN ----------------
if __name__ == "__main__":
    main()
