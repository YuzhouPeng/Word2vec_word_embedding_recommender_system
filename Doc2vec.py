from gensim.models import KeyedVectors
import csv

filename = '/Users/pengyuzhou/Downloads/GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(filename, binary=True)
csvfile = csv.reader(open('/Users/pengyuzhou/Google Drive/Linkedin_datafile/data.csv',"r"))

for row in csvfile:
    user_profile_list = []

result = model.most_similar(positive=['woman', 'king','army'], topn=3)
print(result)