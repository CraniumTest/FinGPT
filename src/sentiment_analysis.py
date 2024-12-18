from transformers import pipeline

# Initialize sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    """Analyzes sentiment of input text."""
    sentiment = sentiment_pipeline(text)[0]
    return sentiment['label'], sentiment['score']

# Example Usage
text = "The stock market is seeing unprecedented growth this year."
label, score = analyze_sentiment(text)
print(f"Sentiment: {label}, Confidence Score: {score:.2f}")
