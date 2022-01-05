import codecademylib3_seaborn
import pandas as pd
import numpy as np
from articles import articles
from preprocessing import preprocess_text

# import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

# view article
for i in range(len(articles)): #<-- Iterate through each article in articles
  print('{}\n'.format(articles[i])) #<-- View each article

# preprocess articles
processed_articles = [] #<-- Create an empty list
for i in range(len(articles)): #<-- Iterate through each article in articles
  processed_articles.append(preprocess_text(articles[i])) #<-- append each processed article to processed_articles

# View a processed article
print(processed_articles[1])

# initialize and fit CountVectorizer
vectorizer = CountVectorizer() #<-- Initialize a CountVectorizer object as vectorizer

counts = vectorizer.fit_transform(processed_articles) #<-- Get word counts for each article



# convert counts to tf-idf
transformer = TfidfTransformer(norm=None) #<-- Initialize TfidfTransformer object as transformer

tfidf_scores_transformed = transformer.fit_transform(counts) #<-- Convert the word counts into tf-idf scores for each article

# initialize and fit TfidfVectorizer
vectorizer = TfidfVectorizer(norm=None) #<-- Initialize TfidfVectorizer object as vectorizer

tfidf_scores = vectorizer.fit_transform(processed_articles) #<-- Calculate the tf-idf scores for each article

# check if tf-idf scores are equal
if np.allclose(tfidf_scores_transformed.todense(), tfidf_scores.todense()):
  print(pd.DataFrame({'Are the tf-idf scores the same?':['YES']}))
else:
  print(pd.DataFrame({'Are the tf-idf scores the same?':['No, something is wrong :(']}))




# get vocabulary of terms
try:
  feature_names = vectorizer.get_feature_names()
except:
  pass

# get article index
try:
  article_index = [f"Article {i+1}" for i in range(len(articles))]
except:
  pass

# create pandas DataFrame with word counts
try:
  df_word_counts = pd.DataFrame(counts.T.todense(), index=feature_names, columns=article_index)
  print(df_word_counts)
except:
  pass

# create pandas DataFrame(s) with tf-idf scores
try:
  df_tf_idf = pd.DataFrame(tfidf_scores_transformed.T.todense(), index=feature_names, columns=article_index)
  print(df_tf_idf)
except:
  pass

try:
  df_tf_idf = pd.DataFrame(tfidf_scores.T.todense(), index=feature_names, columns=article_index)
  print(df_tf_idf)
except:
  pass

# get highest scoring tf-idf term for each article
for i in range(1, 11):
  print(df_tf_idf[[f'Article {i}']].idxmax())
