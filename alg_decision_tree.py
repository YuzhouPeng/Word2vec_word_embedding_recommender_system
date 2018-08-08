from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import globalparameter
def calculate_decision_tree(X_train, Y_train, X_test, Y_test,sum_index):
    classifier = DecisionTreeClassifier()
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

    globalparameter.alg_accuracy[sum_index + 6] = globalparameter.alg_accuracy[sum_index + 6] + accuracy_score
    globalparameter.alg_precision[sum_index + 6] = globalparameter.alg_precision[sum_index + 6] + precision_score
    globalparameter.alg_recall[sum_index + 6] = globalparameter.alg_recall[sum_index + 6] + recall_score
    globalparameter.alg_f1_score[sum_index + 6] = globalparameter.alg_f1_score[sum_index + 6] + f1_score