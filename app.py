import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
from flask import Flask, render_template ,request
# Download VADER lexicon
import nltk
nltk.download('vader_lexicon')



app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')


@app.route("/analyse",methods = ['POST','GET'])
def analyse():
    if request.method == "POST" :
        text = request.form.get("tweetInput")
    predicted_sentiment = classify_sentiment(text)
    data={
        "result":predicted_sentiment
    }
    # print(f"Predicted sentiment: {predicted_sentiment}")
    return render_template("textbox.html",data=data)


@app.route("/text_box_html")
def returntext():
    return render_template('textbox.html')
    



# Load your labeled sentiment dataset
# Replace 'your_dataset.csv' with the path to your dataset file
df = pd.read_csv('tweet_data1.csv')

def preprocess_text(text):
    # Remove URLs, special characters, and numbers
    text = re.sub(r'http\S+|www\S+|[^a-zA-Z\s]', '', text)
    # Tokenization and lowercasing
    words = text.lower().split()
    # Remove stopwords
    stop_words = set(["the", "a", "an", "i", "you", "he", "she", "it"])  # Add more stopwords as needed
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)


def classify_sentiment(text):
    # Preprocess the input text
    preprocessed_text = preprocess_text(text)

    # Use VADER for sentiment analysis
    analyzer = SentimentIntensityAnalyzer()
    compound_score = analyzer.polarity_scores(preprocessed_text)['compound']

    if compound_score >= 0.05:
        return "positive"
    elif compound_score <= -0.05:
        return "negative"
    else:
        return "neutral"


# Example usage:
# while True:
#     user_input = input("Enter a statement to check its sentiment: ")

#     if user_input.strip() == "":
#         print("Please enter a valid sentiment.")
#         continue

    


if __name__ == "__main__":
    app.run(debug = True)