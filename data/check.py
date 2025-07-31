import scipy.io as sio
# 对于 Amazon
data_amazon = sio.loadmat('/home/zjnu/voice_LLM/ljc/graph/unirhogad_project/data/hetero/amazon/Amazon.mat')
print("Amazon keys:", data_amazon.keys())
# 对于 Yelp
data_yelp = sio.loadmat('/home/zjnu/voice_LLM/ljc/graph/unirhogad_project/data/hetero/yelp/YelpChi.mat')
print("Yelp keys:", data_yelp.keys())