import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from surprise import Reader, Dataset
from surprise.model_selection import train_test_split
from surprise import SVD, accuracy
from surprise import KNNWithMeans


ratings_df = pd.read_csv('https://raw.githubusercontent.com/hdupre/rec_sys/master/Project4/ratings.csv')

ratings_df.head()

ratings_df['user_id'].nunique()
ratings_df['book_id'].nunique()

ratings_df.rating.value_counts().plot(kind='bar',)

plt.show()

ratings_df.info()
ratings_df.isnull().sum()

reader = Reader(rating_scale=(1,5))

data = Dataset.load_from_df(ratings_df[['user_id', 'book_id','rating']], reader)

trainset, testset = train_test_split(data, test_size=0.2)

sim_options_ubcf = {'name':'cosine','user_based':False}
sim_option_ibcf = {'name':'cosine','user_based':True}

algo_svd = SVD()
algo_ubcf = KNNWithMeans(sim_options=sim_options_ubcf)
algo_ibcf = KNNWithMeans(sim_options=sim_option_ibcf)



algo_svd.fit(trainset)
algo_ubcf.fit(trainset)
algo_ibcf.fit(trainset)

predictions_svd = algo_svd.test(testset)
predictions_ubcf = algo_ubcf.test(testset)
predictions_ibcf = algo_ibcf.test(testset)

rmse_svd = accuracy.rmse(predictions_svd)
rmse_ubcf = accuracy.rmse(predictions_ubcf)
rmse_ibcf = accuracy.rmse(predictions_ibcf)

predictions_svd()