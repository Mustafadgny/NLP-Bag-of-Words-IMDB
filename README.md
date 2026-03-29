# NLP Bag of Words (BoW) on IMDB Dataset 🎬

This repository contains a complete Natural Language Processing (NLP) pipeline that processes a real-world dataset. It demonstrates how to clean movie reviews from the IMDB dataset and convert the text data into numerical format using the **Bag of Words (BoW)** model.

## 🚀 Pipeline Features
1. **Data Loading:** Reading the IMDB dataset using Pandas.
2. **Text Preprocessing:** - Lowercasing text
   - Removing digits and punctuation using Regex (`re`)
   - Filtering out words shorter than 3 characters
   - Removing English stop words using NLTK
3. **Feature Extraction (BoW):**
   - Transforming the cleaned text into a sparse numerical matrix using `CountVectorizer`.
   - Creating a structured Pandas DataFrame to visualize the vocabulary and word occurrences.
4. **Word Frequency Analysis:** Extracting and displaying the top 5 most frequently used words in the processed documents.

## 🛠️ Libraries Used
- `pandas` (Data manipulation)
- `scikit-learn` (CountVectorizer for BoW)
- `nltk` (Stop words removal)
- `re` (Regular expressions for text cleaning)
- `collections` (Counter for frequency analysis)

## 💻 How to Run
1. Ensure you have the required libraries installed (`pip install pandas scikit-learn nltk`).
2. Place the `IMDB dataset.csv` file in the same directory as the script.
3. Run the Python script. The script limits the BoW transformation to the first 75 rows for fast demonstration purposes, but it can be scaled to the entire dataset.
