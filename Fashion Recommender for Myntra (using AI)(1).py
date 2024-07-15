#!/usr/bin/env python
# coding: utf-8

# # Importing necessary libraries 

# In[7]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix


# # Preprocessing the data and adding mock user id and item id 

# In[10]:


# Load the dataset
df = pd.read_csv('DataMyntra(mock).csv')

# Clean likes column by removing commas and converting to float
df['likes'] = df['likes'].str.replace(',', '').astype(float)

# Ensure tags are evaluated correctly and count the number of tags
def clean_tags(tags):
    try:
        tags_list = eval(tags)
        return len(tags_list)
    except:
        return 0

df['tag_count'] = df['tags'].apply(clean_tags)

# Mocking data for user_id and item_id
np.random.seed(42)  # For reproducibility
num_users = 100
num_items = 20
df['user_id'] = np.random.randint(1, num_users + 1, df.shape[0])
df['item_id'] = np.random.randint(1, num_items + 1, df.shape[0])


# In[15]:


df


# # Sorting by likes and tags count

# In[11]:


# Sort by likes and tag_count in descending order
df_sorted = df.sort_values(by=['likes', 'tag_count'], ascending=[False, False])


# # Content Based Recommender System 

# In[16]:


# Content-Based Recommender System (based on tags)
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df_sorted['tags'].astype('str'))
content_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)


# # 
# # Model Based Collaborative Filtering Recommender System ( CFRS)

# In[17]:


# Collaborative Filtering Recommender System (based on likes)
sparse_matrix = csr_matrix((df_sorted['likes'], (df_sorted['user_id'] - 1, df_sorted['item_id'] - 1)))
als_model = AlternatingLeastSquares(factors=50, regularization=0.01)
als_model.fit(sparse_matrix)


# # Getting Recommendations 

# In[19]:


# Example of getting recommendations for a user
user_id = 1  # Example user id
user_vector = als_model.user_factors[user_id - 1]
item_factors = als_model.item_factors
cf_recommendations = np.argsort(np.dot(item_factors, user_vector))[::-1]

# Print recommended image_urls for content-based and collaborative filtering
print("Content-Based Recommendations (based on tags):")
for idx in content_similarity[user_id - 1].argsort()[::-1][:10]:
    print(f"Image URL: {df_sorted.iloc[idx]['image_url']}")



# In[22]:


for item_id in cf_recommendations[:10]:
    # Find the corresponding image_url for the recommended item_id
    recommended_item = df_sorted[df_sorted['item_id'] == (item_id + 1)]
    if not recommended_item.empty:
        print(f"Image URL: {recommended_item['image_url'].values[0]}")
    else:
        continue


# In the context of the code provided earlier, here's how it incorporates AI-based techniques:
# 
# 1. **TF-IDF Vectorization (Content-Based):**
#    - The code uses `TfidfVectorizer` from `sklearn.feature_extraction.text` to transform text data (tags in this case) into numerical vectors based on term frequency-inverse document frequency (TF-IDF). This is a technique commonly used in natural language processing (NLP) and text mining tasks.
#    - **AI Aspect:** TF-IDF vectorization is a fundamental technique in AI and machine learning for transforming textual data into a numerical format suitable for machine learning models, enabling algorithms to process and understand textual information.
# 
# 2. **Collaborative Filtering with ALS (Model-Based):**
#    - The code employs `AlternatingLeastSquares` from `implicit.als` for collaborative filtering. ALS is a matrix factorization technique commonly used in recommendation systems to decompose user-item interaction matrices into lower-dimensional user and item matrices.
#    - **AI Aspect:** ALS is an AI-based technique that uses iterative optimization to learn latent factors (embeddings) for users and items, capturing their preferences and characteristics from observed interactions (likes in this case). It leverages AI to model complex patterns in user behavior and item preferences.
# 
# 3. **Cosine Similarity (Memory-Based):**
#    - The memory-based collaborative filtering part uses `cosine_similarity` from `sklearn.metrics.pairwise` to compute similarities between items based on user likes (item-item similarity).
#    - **AI Aspect:** Cosine similarity is a measure commonly used in AI and machine learning to quantify similarity between vectors (in this case, vectors representing items based on user likes). It's used here to recommend items that are most similar to those liked by a user, leveraging AI techniques to compute and utilize similarity metrics.
# 
# 4. **Overall AI Integration:**
#    - The entire workflow integrates various AI techniques and algorithms (TF-IDF, ALS, cosine similarity) to preprocess data, build models, and make recommendations based on user interactions. These techniques collectively contribute to the AI-based nature of the recommendation system by leveraging machine learning and statistical methods to infer patterns and make personalized recommendations.
# 
# In summary, while the code example provided does not explicitly cover every aspect of AI in detail, it incorporates essential AI techniques such as TF-IDF vectorization for content-based features and ALS for collaborative filtering, along with cosine similarity for memory-based recommendations. These techniques together form the basis of modern AI-driven recommendation systems, enabling them to process and analyze user data to provide relevant and personalized recommendations.

# In[ ]:




