import numpy as np

from surprise import Reader, Dataset, accuracy
from surprise.model_selection import train_test_split
from surprise import SVD, SVDpp, KNNBasic, KNNWithMeans, KNNWithZScore, SlopeOne, BaselineOnly, NormalPredictor
from surprise import KNNWithMeans

r = pd.read_csv('ratings.csv')
tr = pd.read_csv('to_read.csv')
b = pd.read_csv('books.csv')
t = pd.read_csv('tags.csv')
bt = pd.read_csv('book_tags.csv')

r.head()
b.head()

# create a reader that takes the rating scale as a parameter
reader = Reader(rating_scale=(1,5))
# use the load_from_df function to load our book ratings dataframe
data = Dataset.load_from_df(r[['user_id', 'book_id','rating']], reader)

# split data into a training set and a test set with an 80/20 ratio
trainset, testset = train_test_split(data, test_size=0.2)
algo_svd = SVD()

algo_svd.fit(trainset)

predictions = algo_svd.test(trainset.build_anti_testset())

predictions_svd = algo_svd.test(testset)
pred_svd = pd.DataFrame(predictions_svd)

r.loc[(r['user_id']==27523) & (r['book_id'] == 2203)]

SVD().fit

SVD().fit(trainset)
SVDpp().fit(trainset)
KNNBasic(sim_options={'name':'cosine', 'user_based':True}).fit(trainset)
KNNWithMeans(sim_options={'name':'cosine', 'user_based':True}).fit(trainset)
KNNWithZScore(sim_options={'name':'cosine', 'user_based':True}).fit(trainset)
KNNBasic(sim_options={'name':'cosine', 'user_based':False}).fit(trainset)
KNNWithMeans(sim_options={'name':'cosine', 'user_based':False}).fit(trainset)
KNNWithZScore(sim_options={'name':'cosine', 'user_based':False}).fit(trainset)
SlopeOne().fit(trainset)
BaselineOnly().fit(trainset)
NormalPredictor().fit(trainset)

SVD().fit(trainset)
SVDpp().fit(trainset)
KNNBasic(sim_options={'name':'cosine', 'user_based':True}).fit(trainset)
KNNWithMeans(sim_options={'name':'cosine', 'user_based':True}).fit(trainset)
KNNWithZScore(sim_options={'name':'cosine', 'user_based':True}).fit(trainset)
KNNBasic(sim_options={'name':'cosine', 'user_based':False}).fit(trainset)
KNNWithMeans(sim_options={'name':'cosine', 'user_based':False}).fit(trainset)
KNNWithZScore(sim_options={'name':'cosine', 'user_based':False}).fit(trainset)
SlopeOne().fit(trainset)
BaselineOnly().fit(trainset)
NormalPredictor().fit(trainset)


pred_svd = SVD().test(testset)
pred_svdpp = SVDpp().test(testset)

r.head(10)

count_series = r['user_id'].value_counts().head(7000).index.tolist()

r = r[r['user_id'].isin(r['user_id'].value_counts().head(7000).index.tolist())]