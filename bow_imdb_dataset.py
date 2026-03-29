# import libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords") 
stop_words_eng = set(stopwords.words("english"))

# veri setinin iceriye aktarilmasi
# Not: CSV dosyanın kod ile aynı klasörde olduğundan emin ol
df = pd.read_csv("IMDB dataset.csv")

# metin verilerini alalim
documents = df["review"]
labels = df["sentiment"] # positive and negative

# metin temizleme fonksiyonu
def clean_text(text):
    text = text.lower() # Küçük harfe çevirme
    text = re.sub(r"\d+", "", text) # Sayıları kaldırma
    text = re.sub(r"[^\w\s]", "", text) # Noktalama işaretlerini kaldırma
    text = " ".join([word for word in text.split() if len(word) > 2 and word not in stop_words_eng])
    
    return text


cleaned_doc = [clean_text(row) for row in documents]


# %% bow
# vectorizer tanimla
vectorizer = CountVectorizer()

# metin'i sayılsal hale getir
X = vectorizer.fit_transform(cleaned_doc[:75])

# kelime kumesi goster
feature_names = vectorizer.get_feature_names_out()

#vektor temsili goster
vektor_temsili2 = X.toarray()

db_bow = pd.DataFrame(vektor_temsili2, columns= feature_names)

# kelime frekanslarını göster

word_counts = X.sum(axis=0).A1
word_freq = dict(zip(feature_names, word_counts))

# ilk 5 kelimeyi print ettir

most_common_5_words = Counter(word_freq).most_common(5)
print(f"most_common_5_words: {most_common_5_words}")





