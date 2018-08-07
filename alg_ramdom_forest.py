import pandas as pd
import bag_of_words
import numpy as np
import globalparameter, csv, itertools, generate_train_test_set,n_grams,time
from sklearn import preprocessing
import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

def random_forest(folderpath,jobtitle_path_list,ratio,sum_index):
    user_profile = pd.DataFrame(pd.read_csv(folderpath+'/test1.csv'))

    X = user_profile[['normalized_work_year_past1', 'normalized_work_year_past2',
                      'normalized_work_year_past3', 'normalized_work_year_past4', 'normalized_work_year_past5',
                      'normalized_work_year_past6']]  # use highest degree, six past work experience
    Y = user_profile['normalized_now_relevant_job']
    # np.unique(Y)   # out: array([0, 1, 2])

    # split test and train set
    # X_train = pd.concat(
    #     [X.iloc[0:int(globalparameter.extract_number * ratio)], X.iloc[int(globalparameter.extract_number):int(
    #         globalparameter.extract_number + (globalparameter.total_number - globalparameter.extract_number) * ratio)]])
    # X_test = pd.concat([X.iloc[int(globalparameter.extract_number * ratio):globalparameter.extract_number], X.iloc[int(
    #     globalparameter.extract_number + (
    #             globalparameter.total_number - globalparameter.extract_number) * ratio):globalparameter.total_number]])
    Y_train = pd.concat([Y.iloc[0:int(globalparameter.extract_number * ratio)], Y.iloc[int(globalparameter.extract_number):int(
            globalparameter.extract_number + (globalparameter.total_number - globalparameter.extract_number) * ratio)]])
    Y_test = pd.concat([Y.iloc[int(globalparameter.extract_number * ratio):globalparameter.extract_number], Y.iloc[int(
        globalparameter.extract_number + (
                globalparameter.total_number - globalparameter.extract_number) * ratio):globalparameter.total_number]])

    # generate matrix of bag-of-words
    matrix = bag_of_words.extractall_information(folderpath + '/' + 'output_pos_for_dummy.csv',
                                                 folderpath + '/' + 'output_neg_for_dummy.csv',
                                                 globalparameter.extract_skills_list)

    # generate matrix of 2-gram
    # matrix = n_grams.extractall_information_n_gram(folderpath + '/' + 'output_pos_for_dummy.csv',
    #                                                folderpath + '/' + 'output_neg_for_dummy.csv',
    #                                                globalparameter.extract_column_list, 2)

    # generate matrix of 3-gram
    # matrix = n_grams.extractall_information_n_gram(folderpath + '/' + 'output_pos_for_dummy.csv',
    #                                                folderpath + '/' + 'output_neg_for_dummy.csv',
    #                                                globalparameter.extract_skills_list, 3)

    X_train = generate_train_test_set.generate_X_train(matrix, X, ratio, globalparameter.train_pos_start_loc,
                                                         globalparameter.train_pos_end_loc,
                                                         globalparameter.train_neg_start_loc,
                                                         globalparameter.train_neg_end_loc)
    X_test = generate_train_test_set.generate_X_test(matrix, X, ratio, globalparameter.test_pos_start_loc,
                                                       globalparameter.test_pos_end_loc,
                                                       globalparameter.test_neg_start_loc,
                                                       globalparameter.test_neg_end_loc)

    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # train logestic regression
    decision_tree_classifier = RandomForestClassifier()
    start_time = time.time()
    decision_tree_classifier.fit(X_train, Y_train)
    end_time = time.time()
    time_interval = end_time-start_time
    print('time_intervial is: {}'.format(time_interval))
    # Y_score = decision_tree_classifier.decision_function(X_test_std)

    # predict
    prediction = decision_tree_classifier.predict((X_test_std))
    Y_test.index = range(int(globalparameter.total_number * (1 - ratio)))
    prepro = decision_tree_classifier.predict_proba(X_test_std)
    acc = decision_tree_classifier.score(X_test_std, Y_test)
    precision = precision_score(Y_test,prediction,labels=[0,1],pos_label=1)
    recall = recall_score(Y_test, prediction,labels=[0, 1], pos_label=1)

    # print('prediction is : {}'.format(prediction))
    print('-------')
    print('random forest')
    # print('prepro is : {}'.format(prepro))
    print('acc is predict proba is {}'.format(acc))
    print('precision is: {}'.format(precision))
    print('recall is {}'.format(recall))

    globalparameter.alg_accuracy[sum_index+7] = globalparameter.alg_accuracy[sum_index+7] + acc
    globalparameter.alg_precision[sum_index+7] = globalparameter.alg_precision[sum_index+7] + precision
    globalparameter.alg_recall[sum_index+7] = globalparameter.alg_recall[sum_index+7] + recall
    globalparameter.time[sum_index+7] = globalparameter.time[sum_index+7] + time_interval

    # print('average precision and recall is {}'.format(avg_precesion))

    #plot the diagram
    # precision, recall, _ = precision_recall_curve(Y_test,Y_score)
    #
    # # plt.step(recall, precision, color='blue', alpha=0.2,where='post')
    # plt.plot(recall,precision,label = 'decision tree')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.ylim([0.0, 1.05])
    # plt.xlim([0.0, 1.0])
    # plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(avg_precesion))
    # plt.savefig(globalparameter.folderpath[1] + '/diagram_logestic_regression_p-r_' + globalparameter.jobtitle_path_list[1] + '-extract' + '.png')