#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# # Data Collection & Pre-Processing

# In[2]:


# loading the data from the csv file to a pandas dataframe
movies_data = pd.read_csv('movies.csv')


# In[3]:


# printing the first 5 rows of the dataframe
movies_data.head()


# In[4]:


movies_data.shape


# In[5]:


# selecting the relevant features for content based recommendation
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
print(selected_features)


# In[6]:


# replacing the null values with null string
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')


# In[7]:


# combining all the 5 selected features
combined_features = movies_data['genres'] + movies_data['keywords'] + movies_data['tagline'] + movies_data['cast'] + movies_data['director']


# In[8]:


print(combined_features)


# In[9]:


# converting the text data to feature vectors
vectorizer = TfidfVectorizer()


# In[10]:


feature_vectors = vectorizer.fit_transform(combined_features)


# In[11]:


print(feature_vectors)


# # Cosine Similarity

# In[12]:


# getting the similarity scores using cosine similarity
similarity = cosine_similarity(feature_vectors)


# In[13]:


print(similarity)


# In[14]:


similarity.shape


# In[15]:


def recommend_using_title(title):
    movie_name = title
    
    list_of_all_titles = movies_data['title'].tolist()
    
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
    
    close_match = find_close_match[0]
    
    index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
    
    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    
    sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True)
    
    titles = []
    
    i = 1
    
    for movie in sorted_similar_movies:
        
        index = movie[0]
        
        title_from_index = movies_data[movies_data.index == index]['title'].values[0]
        
        if(i<=10):
            titles.append(title_from_index)
            i += 1
            
        else:
            break
            
    return titles


# In[16]:


recommend_using_title('iron man')


# In[17]:


def recommend_using_genre(genres):
    movie_genre = genres
    list_of_all_genres = movies_data['genres'].tolist()
    find_close_match = []
    close_match = []
    titles = []
    indices = []
    
    for genre in list_of_all_genres:
        current_genre = genre.split()
        
        find_close_match = difflib.get_close_matches(movie_genre, current_genre)
        
        if find_close_match:
            close_match.append(genre)
            
            
    index_of_the_movie = movies_data[movies_data.genres == close_match[0]]['index'].values[0]
    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True)     
    
    i = 1
    
    for movie in sorted_similar_movies:
        index = movie[0]
#         title_from_index = movies_data[movies_data.index == index]['title'].values[0]
        
        if(i<=30):
#             titles.append(title_from_index)
            indices.append(index)
            i += 1
        else:            
            break
            
            
    result = pd.DataFrame()

    for i in indices:
        title = movies_data.loc[i].to_frame().T
#         print(title)
        result = pd.concat([result, title], ignore_index=True)
    
    return result


# In[18]:


# test_df = movies_data
# test_df.set_index('index', inplace=True)
result = recommend_using_genre('Action')
result


# In[19]:


def browse_movies(shown, num):
    data = movies_data.iloc[shown : num, :]
    df = pd.DataFrame(data)
    return data


# In[20]:


data = browse_movies(10, 21)
data


# In[21]:


# get a list of all genres
def get_genres():
    list_of_all_genres = movies_data['genres'].tolist()
    unique_genres = []
    
    for genre in list_of_all_genres:
        current_genre = genre.split()
        
        for gen in current_genre:
            if gen not in unique_genres:
                unique_genres.append(gen)
                
    return unique_genres


# In[22]:


print(get_genres())


# In[23]:


for col in movies_data.columns:
    print(col)


# In[24]:


def get_languages():
    all_languages = movies_data['original_language'].tolist()
    unique_lang = []

    for lang in all_languages:
        if lang not in unique_lang:
            unique_lang.append(lang)

    print(unique_lang)


# In[25]:


def get_popular_movies(show_new, shown):
    sorted_movies = movies_data.sort_values(by=['popularity'], ascending=False)
    return sorted_movies.iloc[shown:show_new+1, :]


# In[26]:


popular_movies = get_popular_movies(10, 0)


# In[27]:


popular_movies


# In[28]:


movies_data.isnull().sum()


# In[29]:


movies_data.shape


# In[30]:


movies_data.drop(['homepage'], axis=1, inplace=True)


# In[31]:


movies_data.shape


# In[32]:


movies_data.isnull().sum()


# In[33]:


movies_data.dropna(inplace=True)


# In[34]:


movies_data.isnull().sum()


# In[ ]:




