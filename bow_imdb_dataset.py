# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords

# Download stopwords and initialize the English stop words set
nltk.download("stopwords") 
stop_words_eng = set(stopwords.words("english"))

# Load the dataset
# Note: Ensure the "IMDB dataset.csv" file is in the same directory as the script
df = pd.read_csv("IMDB dataset.csv")

# Extract text data and labels
documents = df["review"]
labels = df["sentiment"] # Positive and negative

# Text cleaning function
def clean_text(text):
    text = text.lower() # Convert to lowercase
    text = re.sub(r"\d+", "", text) # Remove digits
    text = re.sub(r"[^\w\s]", "", text) # Remove punctuation marks
    
    # Filter out short words (length <= 2) and stop words
    text = " ".join([word for word in text.split() if len(word) > 2 and word not in stop_words_eng])
    
    return text

# Apply the cleaning function to the documents
cleaned_doc = [clean_text(row) for row in documents]

# %% Bag of Words (BoW) Implementation

# Initialize the vectorizer
vectorizer = CountVectorizer()

# Convert text into a numerical matrix (fitting on the first 75 reviews for demonstration)
X = vectorizer.fit_transform(cleaned_doc[:75])

# Extract the vocabulary (feature names)
feature_names = vectorizer.get_feature_names_out()

# Get the dense vector representation
vector_representation = X.toarray()

# Create a DataFrame for a structured view of the BoW matrix
db_bow = pd.DataFrame(vector_representation, columns=feature_names)

# Calculate word frequencies across the selected documents
word_counts = X.sum(axis=0).A1
word_freq = dict(zip(feature_names, word_counts))

# Print the top 5 most common words
most_common_5_words = Counter(word_freq).most_common(5)
print(f"Top 5 most common words: {most_common_5_words}")