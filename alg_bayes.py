from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import globalparameter
def calculate_bayes(X_train, Y_train, X_test, Y_test,sum_index,top_index):
    classifier = GaussianNB()
    classifier.fit(X_train, Y_train)
    prediction = classifier.predict(X_test)
    accuracy_score_p = metrics.accuracy_score(Y_test, prediction)
    accuracy_score = classifier.score(X_test, Y_test)
    precision_score = metrics.precision_score(Y_test, prediction)
    recall_score = metrics.recall_score(Y_test, prediction)
    f1_score = metrics.f1_score(Y_test, prediction)
    print('accuracy_score_p is : ' + str(accuracy_score_p))
    print('accuracy_score is : ' + str(accuracy_score_p))
    print('precision acore is :' + str(precision_score))
    print('recall score is :' + str(recall_score))
    print('f1_score is :' + str(f1_score))
    # calculate precision@n and recall@n
    recommend_relevant = 0
    for i in range(top_index):
        if prediction[i] == Y_test[i] and prediction[i] ==1:
            recommend_relevant = recommend_relevant+1
    precision_atn = recommend_relevant/top_index
    recall_atn = recommend_relevant/200
    print('precision@n acore is :' + str(precision_atn))
    print('recall@n score is :' + str(recall_atn))

    globalparameter.alg_accuracy[sum_index + 5] = globalparameter.alg_accuracy[sum_index + 5] + accuracy_score
    # globalparameter.alg_precision[sum_index + 5] = globalparameter.alg_precision[sum_index + 5] + precision_score
    # globalparameter.alg_recall[sum_index + 5] = globalparameter.alg_recall[sum_index + 5] + recall_score
    globalparameter.alg_f1_score[sum_index + 5] = globalparameter.alg_f1_score[sum_index + 5] + f1_score

    globalparameter.alg_precision[sum_index + 5] = globalparameter.alg_precision[sum_index + 5] + precision_atn
    globalparameter.alg_recall[sum_index + 5] = globalparameter.alg_recall[sum_index + 5] + recall_atn