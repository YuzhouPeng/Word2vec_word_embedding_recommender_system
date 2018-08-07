from gensim.models import KeyedVectors
import csv
# gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
# numpy
import numpy,linkedindata,datafilter,globalparameter, random,labeled_data_sentence
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
# random
from random import shuffle
# classifier
from sklearn.linear_model import LogisticRegression
#
# filename = '/Users/pengyuzhou/Downloads/GoogleNews-vectors-negative300.bin'
# model = KeyedVectors.load_word2vec_format(filename, binary=True)
# csvfile = csv.reader(open('/Users/pengyuzhou/Google Drive/Linkedin_datafile/data.csv',"r"))

linkedIndata_list = []
with open('/Users/pengyuzhou/Google Drive/Linkedin_datafile/data_v3.csv', 'r') as f:
    next(f)
    reader = csv.reader(f)
    for row in reader:
        parameter_list = [row[index] for index in range(112)]
        linkedIndata_list.append(
            linkedindata.LinkedInData(*parameter_list))

for i in range(6):
    average_precision = 0.0
    average_recall = 0.0
    average_accuracy = 0.0
    print('Began to read job title: '+ globalparameter.jobtitle_list[i])
relevant_user_list = datafilter.filter_data_with_job_title_oo(linkedIndata_list, globalparameter.jobtitle_list[0],
                                                              1)
non_relevant_user_list = datafilter.filter_data_with_job_title_oo(linkedIndata_list,
                                                                  globalparameter.jobtitle_list[0], 2)
pos_list = random.sample(relevant_user_list, 500)
neg_list = random.sample(non_relevant_user_list, 500)


# You can change the features you want to use
pos_profile_list = []
neg_profile_list = []
for i in range(len(pos_list)):
    userprofile = ' '.join(pos_list[i].return_value_all())
    pos_profile_list.append(userprofile)

for i in range(len(neg_list)):
    userprofile = ' '.join(neg_list[i].return_value_all())
    neg_profile_list.append(userprofile)

pos_label_list = ['pos_profile_'+ str(i) for i in range(len(pos_profile_list))]
neg_label_list = ['neg_profile_'+ str(i) for i in range(len(pos_profile_list))]

documents = labeled_data_sentence.LabeledLineSentence(pos_profile_list,pos_label_list,neg_profile_list,neg_label_list)
model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=8)
model.build_vocab(documents.to_array())
for epoch in range(20):
    model.train(documents.sentences_perm(), total_examples=model.corpus_count, epochs=model.iter)
    # model.train(documents.sentences_perm())
model.save('/Users/pengyuzhou/Downloads/word_embedding_result/job_title_'+globalparameter.jobtitle_list[0]+'.d2v')
model = Doc2Vec.load('/Users/pengyuzhou/Downloads/word_embedding_result/job_title_'+globalparameter.jobtitle_list[0]+'.d2v')

test2 = model.most_similar('software')

train_arrays = numpy.zeros((500,100))
train_labels = numpy.zeros(500)

for i in range(250):
    prefix_pos_train = 'pos_profile_'+ str(i)
    prefix_neg_train = 'neg_profile_'+ str(i)
    train_arrays[i] = model[prefix_pos_train]
    train_arrays[250+i] = model[prefix_neg_train]
    train_labels[i] = 1
    train_labels[250+i] = 0

test_arrays = numpy.zeros((500, 100))
test_labels = numpy.zeros(500)
for i in range(250):
    prefix_test_pos = 'pos_profile_' + str(250+i)
    prefix_test_neg = 'neg_profile_' + str(250+i)
    test_arrays[i] = model[prefix_test_pos]
    test_arrays[250 + i] = model[prefix_test_neg]
    test_labels[i] = 1
    test_labels[250 + i] = 0

classifier = LogisticRegression()
classifier.fit(train_arrays, train_labels)
prediction = classifier.predict(test_arrays)
accuracy_score_p = accuracy_score(test_labels,prediction)
accuracy_score = classifier.score(test_arrays, test_labels)
precision_score = precision_score(test_labels,prediction)
recall_score = recall_score(test_labels,prediction)

print('accuracy_score_p is : '+str(accuracy_score_p))
print('accuracy_score is : '+ str(accuracy_score_p))
print('precision acore is :' + str(precision_score))
print('recall score is :' + str(recall_score))


print(1)