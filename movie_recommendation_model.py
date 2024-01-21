#!/usr/bin/env python
# coding: utf-8

# In[1]:


import Movie_Recommendation_System as mrs


# In[2]:


import pickle


# In[3]:


model_dict = {
    'recommend_using_title': mrs.recommend_using_title,
    'recommend_using_genre': mrs.recommend_using_genre,
    'browse_movies': mrs.browse_movies,
    'get_genres': mrs.get_genres,
    'get_languages': mrs.get_languages
}


# In[4]:


filename = 'movie_recommendation.sav'
pickle.dump(model_dict, open(filename, 'wb'))


# In[5]:


#loading the saved model
loaded_model = pickle.load(open('movie_recommendation.sav', 'rb'))


# In[ ]:





# In[ ]:




