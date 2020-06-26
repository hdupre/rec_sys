import pandas as pd 
import numpy as np
import os
import math


ratings_df = pd.read_csv('https://raw.githubusercontent.com/hdupre/rec_sys/master/Project3/ratings.csv')
movies_df = pd.read_csv('https://raw.githubusercontent.com/hdupre/rec_sys/master/Project3/movies.csv')

ratings_df.head()

ratings_pivot = ratings_df.pivot_table(index='userId', columns='movieId', values='rating')
ratings_mat.head()
ratings_mat= ratings_pivot.replace(np.nan, 0)
u, s, v_t = np.linalg.svd(ratings_mat, full_matrices=False)
print(u.shape,sig.shape,v_t.shape)
prediction_array = np.dot(u, np.dot(np.diag(s),v_t))
prediction_df = pd.DataFrame(prediction_array, columns= ratings_mat.columns)
prediction_df.index += 1


selected_user = prediction_df.loc[5, : ]

selected_user = selected_user.sort_values(ascending=False)

j=0

while i < 10:
    if ratings_pivot[selected_user.index[

def recommender(user_id, prediction_matrix, ratings_pivot, movies_df,n_recommendations):

