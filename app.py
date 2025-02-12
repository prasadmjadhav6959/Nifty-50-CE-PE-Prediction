import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob  # For sentiment analysis

def preprocess_nifty50_data(df):
    df.columns = df.columns.str.strip()
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%b-%Y")
    df[["Open", "High", "Low", "Close"]] = df[["Open", "High", "Low", "Close"]].astype(float)
    
    df["Returns"] = df["Close"].pct_change()
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["Volatility"] = df["Returns"].rolling(window=10).std()
    
    df.dropna(inplace=True)
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    return df

def analyze_news_sentiment(news_articles):
    sentiment_scores = [TextBlob(article).sentiment.polarity for article in news_articles]
    avg_sentiment = np.mean(sentiment_scores)
    return avg_sentiment

def train_model(df):
    features = ["Open", "High", "Low", "Close", "SMA_10", "EMA_10", "Volatility"]
    X = df[features]
    y = df["Target"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler

def predict_ce_pe(model, scaler, latest_data, news_articles):
    latest_data_scaled = scaler.transform([latest_data])
    prediction = model.predict(latest_data_scaled)[0]
    sentiment_score = analyze_news_sentiment(news_articles)
    
    results = []
    if prediction == 1:
        for i in range(2):
            strike_price = round(latest_data["Close"] / 100) * 100 + (100 * i)
            results.append(f"{strike_price} CE Buy it")
        for i in range(3):
            strike_price = round(latest_data["Close"] / 100) * 100 - (100 * i)
            results.append(f"{strike_price} PE Buy it")
    else:
        for i in range(3):
            strike_price = round(latest_data["Close"] / 100) * 100 - (100 * i)
            results.append(f"{strike_price} PE Buy it")
        for i in range(2):
            strike_price = round(latest_data["Close"] / 100) * 100 + (100 * i)
            results.append(f"{strike_price} CE Buy it")
    
    return results

# Streamlit UI
st.title("Nifty 50 CE/PE Prediction")

uploaded_file = st.file_uploader("Upload NIFTY 50 CSV file", type=["csv"])
news_input = st.text_area("Paste top 5-10 news articles here")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = preprocess_nifty50_data(df)
    model, scaler = train_model(df)
    latest_values = df.iloc[-1][["Open", "High", "Low", "Close", "SMA_10", "EMA_10", "Volatility"]].values
    news_articles = news_input.split("\n")
    
    if st.button("Predict CE/PE"):
        predictions = predict_ce_pe(model, scaler, latest_values, news_articles)
        st.subheader("Predictions:")
        for p in predictions:
            st.write(p)
