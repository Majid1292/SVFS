import pandas as pd
import numpy as np
import time
import os
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from reduction import svfs
def main():
    # Biological datasets
    #dataset_list = ['GDS1615_full_NoFeature', 'GDS3268_full_NoFeature', 'GDS531_full_NoFeature','GDS1962_full_NoFeature', 'GDS968_full_NoFeature','GDS3929_full_NoFeature', 'GDS2545_full_NoFeature', 'GDS2546_full_NoFeature', 'GDS2547_full_NoFeature']
    # Image datasets
    # dataset_list=['pixraw10P','warpAR10P','warpPIE10P','orlraws10P']
    # Genomic datasets
    dataset_list=['TOX_171','SMK_CAN_187','Prostate_GE','lymphoma','leukemia','lung','GLIOMA','GLI_85','CLL_SUB_111','ALLAML','colon','nci9']
    path = os.path.abspath(os.getcwd()) + "/Datasets/"
    # Parameters
    k = 50
    th_irr = 3
    th_red = 4
    alpha = 50
    beta = 5
    print(os.path.abspath(os.getcwd())
)
    for dataset in dataset_list:
        print("\nDataset: ", dataset)
        print("Loading Dataset")
        data = pd.read_csv(path + dataset + ".csv", header=None)
        dataX = data.copy().iloc[:, :-1]
        dataY = data.copy().iloc[:, data.shape[1] - 1]
        acc_list = []
        features_list = []
        # Split into train and test
        k_fold = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1)
        start_time = time.time()
        for train_idx, test_dix in k_fold.split(dataX, dataY):
            train_x, test_x = dataX.iloc[train_idx, :].copy(), dataX.iloc[test_dix, :].copy()
            train_y, test_y = dataY.iloc[train_idx].copy(), dataY.iloc[test_dix].copy()
            list_features = []
            best_acc = 0
            best_idx = 0
            fs = svfs(train_x, train_y, th_irr, 1.7, th_red, k, alpha, beta)
            reduced_data = fs.reduction()
            high_x = fs.high_rank_x()
            clean_features = high_x[reduced_data]
            dic_cls = fs.selection(high_x,reduced_data,clean_features)
            idx = 0
            for key, value in dic_cls.items():
                list_features.append(clean_features[key])
                X_train = train_x.iloc[:, list_features].copy()
                X_test = test_x.iloc[:, list_features].copy()
                clf = RandomForestClassifier()
                clf.fit(X_train, train_y)
                y_pred = clf.predict(X_test)
                acc = metrics.accuracy_score(test_y, y_pred)
                if acc > best_acc:
                    best_acc = acc
                    best_idx = idx
                idx += 1
            print("Best ACC is: %.2f" % (best_acc * 100), "for ", best_idx + 1, " # of features")
            acc_list.append(best_acc)
            features_list.append(best_idx + 1)
        print("Average ACC is : %.2f" % (np.average(acc_list) * 100))
        print("Average ACC is : %.2f" % (np.average(features_list)))
        print("SD of ACC is : %.2f" % (np.std(acc_list) * 100) + " SD of Feature is : %.2f" % (np.std(features_list)))
        print("Running Time: %s seconds\n\n" % round((time.time() - start_time), 2))
        
main()
