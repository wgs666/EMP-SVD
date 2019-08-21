'''
Code of paper "Prediction of drug-disease associations based on ensemble meta paths and singular value decomposition".
Please kindly cite the paper:
@article{wu2019EMP-SVD,
  title={Prediction of drug-disease associations based on ensemble meta paths and singular value decomposition},
  author={Wu, Guangsheng and Liu, Juan and Yue, Xiang},
  journal={BMC bioinformatics},
  volume={20},
  number={3},
  pages={134},
  year={2019},
  publisher={BioMed Central}
}
If you have any questions, please do not hesitate to contact wgs@whu.edu.cn
'''

# Python 3 or Anaconda 3 is needed.
import time
import numpy as np
import random
from sklearn.model_selection import KFold
from numpy import linalg as la
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
from scipy import interp


def EMP_SVD(drug_disease_matrix, drug_protein_matrix, disease_protein_matrix, latent_feature_percent=0.03):
    # print(drug_disease_matrix.shape)
    # print(drug_protein_matrix.shape)
    # print(disease_protein_matrix.shape)
    none_zero_position = np.where(drug_disease_matrix != 0)
    none_zero_row_index = none_zero_position[0]
    none_zero_col_index = none_zero_position[1]

    ##### code for randomly selected nagative samples
    # zero_position = np.where(drug_disease_matrix == 0)
    # zero_row_index = zero_position[0]
    # zero_col_index = zero_position[1]
    # random.seed(1)    
    # zero_random_index = random.sample( range(len(zero_row_index)), len(none_zero_row_index) )
    # zero_row_index = zero_row_index[zero_random_index]
    # zero_col_index = zero_col_index[zero_random_index]


    ##### code for reliable nagative samples
    drug_protein_dis_matrix=np.dot(drug_protein_matrix, disease_protein_matrix.T)
    zero_deduction_dpd_position=np.where(drug_protein_dis_matrix == 0)
    zero_deduction_dpd_row_index=zero_deduction_dpd_position[0]
    zero_deduction_dpd_col_index = zero_deduction_dpd_position[1]
    random.seed(1)
    zero_random_index = random.sample(range(len(zero_deduction_dpd_row_index)), len(none_zero_row_index))
    zero_row_index = zero_deduction_dpd_row_index[zero_random_index]
    zero_col_index = zero_deduction_dpd_col_index[zero_random_index]




    row_index = np.append(none_zero_row_index, zero_row_index)
    col_index = np.append(none_zero_col_index, zero_col_index)

    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    metric_avg = np.zeros((6, 7), float)
    count = 1

    tprs = []
    precisions = []
    auprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 30)
    mean_recall = np.linspace(0, 1, 30)

    for train, test in kf.split(row_index):
        # print('begin cross validation experiment ' + str(count) + '/' + str(kf.n_splits))
        count += 1
        train_drug_disease_matrix = np.copy(drug_disease_matrix)
        test_row = row_index[test]
        test_col = col_index[test]
        train_row = row_index[train]
        train_col = col_index[train]
        np.savetxt('../crossValidation/test_row' + str(count-1) + '.txt', test_row, fmt='%d')
        np.savetxt('../crossValidation/test_col' + str(count-1) + '.txt', test_col, fmt='%d')

        train_drug_disease_matrix[test_row, test_col] = 0
        np.savetxt('../crossValidation/train_drug_disease_matrix_' + str(count-1) + '.txt', train_drug_disease_matrix, fmt='%d')

        ######################################################################################################
        #### step1: define meta paths

        # meta-path-1: drug->disease
        meta_path_1 = train_drug_disease_matrix

        # meta-path-2: drug->protein->disease
        meta_path_2 = np.dot(drug_protein_matrix, disease_protein_matrix.T)

        # meta-path-3: drug->protein->drug->disease
        meta_path_3 = np.dot(np.dot(drug_protein_matrix, drug_protein_matrix.T), train_drug_disease_matrix)

        # meta-path-4: drug->disease->drug->disease
        meta_path_4 = np.dot(np.dot(train_drug_disease_matrix, train_drug_disease_matrix.T), train_drug_disease_matrix)

        # meta-path-5: drug->disease->protein->disease
        meta_path_5 = np.dot(np.dot(train_drug_disease_matrix, disease_protein_matrix), disease_protein_matrix.T)

        #############################################################################################################
        #### step 2 extract features by SVD
        # latent_feature_percent = 0.03
        (row, col) = train_drug_disease_matrix.shape
        latent_feature_num = int(min(row, col) * latent_feature_percent)

        ## using SVD
        U, Sigma, VT = la.svd(meta_path_1)
        drug_feature_matrix_1 = U[:, :latent_feature_num]
        disease_feature_matrix_1 = VT.T[:, :latent_feature_num]

        U, Sigma, VT = la.svd(meta_path_2)
        drug_feature_matrix_2 = U[:, :latent_feature_num]
        disease_feature_matrix_2 = VT.T[:, :latent_feature_num]

        U, Sigma, VT = la.svd(meta_path_3)
        drug_feature_matrix_3 = U[:, :latent_feature_num]
        disease_feature_matrix_3 = VT.T[:, :latent_feature_num]

        U, Sigma, VT = la.svd(meta_path_4)
        drug_feature_matrix_4 = U[:, :latent_feature_num]
        disease_feature_matrix_4 = VT.T[:, :latent_feature_num]

        U, Sigma, VT = la.svd(meta_path_5)
        drug_feature_matrix_5 = U[:, :latent_feature_num]
        disease_feature_matrix_5 = VT.T[:, :latent_feature_num]




        ##########################################################################################################
        #### step 3: construct training dataset and testing dataset

        train_feature_matrix_1 = []
        train_feature_matrix_2 = []
        train_feature_matrix_3 = []
        train_feature_matrix_4 = []
        train_feature_matrix_5 = []

        train_label_vector = []
        for num in range(len(train_row)):
            feature_vector = np.append(drug_feature_matrix_1[train_row[num], :],
                                       disease_feature_matrix_1[train_col[num], :])
            train_feature_matrix_1.append(feature_vector)

            feature_vector = np.append(drug_feature_matrix_2[train_row[num], :],
                                       disease_feature_matrix_2[train_col[num], :])
            train_feature_matrix_2.append(feature_vector)

            feature_vector = np.append(drug_feature_matrix_3[train_row[num], :],
                                       disease_feature_matrix_3[train_col[num], :])
            train_feature_matrix_3.append(feature_vector)

            feature_vector = np.append(drug_feature_matrix_4[train_row[num], :],
                                       disease_feature_matrix_4[train_col[num], :])
            train_feature_matrix_4.append(feature_vector)

            feature_vector = np.append(drug_feature_matrix_5[train_row[num], :],
                                       disease_feature_matrix_5[train_col[num], :])
            train_feature_matrix_5.append(feature_vector)

            train_label_vector.append(drug_disease_matrix[train_row[num], train_col[num]])

        test_feature_matrix_1 = []
        test_feature_matrix_2 = []
        test_feature_matrix_3 = []
        test_feature_matrix_4 = []
        test_feature_matrix_5 = []

        test_label_vector = []
        for num in range(len(test_row)):
            feature_vector = np.append(drug_feature_matrix_1[test_row[num], :],
                                       disease_feature_matrix_1[test_col[num], :])
            test_feature_matrix_1.append(feature_vector)

            feature_vector = np.append(drug_feature_matrix_2[test_row[num], :],
                                       disease_feature_matrix_2[test_col[num], :])
            test_feature_matrix_2.append(feature_vector)

            feature_vector = np.append(drug_feature_matrix_3[test_row[num], :],
                                       disease_feature_matrix_3[test_col[num], :])
            test_feature_matrix_3.append(feature_vector)

            feature_vector = np.append(drug_feature_matrix_4[test_row[num], :],
                                       disease_feature_matrix_4[test_col[num], :])
            test_feature_matrix_4.append(feature_vector)

            feature_vector = np.append(drug_feature_matrix_5[test_row[num], :],
                                       disease_feature_matrix_5[test_col[num], :])
            test_feature_matrix_5.append(feature_vector)

            test_label_vector.append(drug_disease_matrix[test_row[num], test_col[num]])

        train_feature_matrix_1 = np.array(train_feature_matrix_1)
        train_feature_matrix_2 = np.array(train_feature_matrix_2)
        train_feature_matrix_3 = np.array(train_feature_matrix_3)
        train_feature_matrix_4 = np.array(train_feature_matrix_4)
        train_feature_matrix_5 = np.array(train_feature_matrix_5)

        test_feature_matrix_1 = np.array(test_feature_matrix_1)
        test_feature_matrix_2 = np.array(test_feature_matrix_2)
        test_feature_matrix_3 = np.array(test_feature_matrix_3)
        test_feature_matrix_4 = np.array(test_feature_matrix_4)
        test_feature_matrix_5 = np.array(test_feature_matrix_5)

        train_label_vector = np.array(train_label_vector)
        test_label_vector = np.array(test_label_vector)

        #################################################################################################
        #### step 4: training and testing
        # here, using random forest as an example
        clf1 = RandomForestClassifier(random_state=1, n_estimators=256, oob_score=True, n_jobs=-1)
        clf2 = RandomForestClassifier(random_state=1, n_estimators=256, oob_score=True, n_jobs=-1)
        clf3 = RandomForestClassifier(random_state=1, n_estimators=256, oob_score=True, n_jobs=-1)
        clf4 = RandomForestClassifier(random_state=1, n_estimators=256, oob_score=True, n_jobs=-1)
        clf5 = RandomForestClassifier(random_state=1, n_estimators=256, oob_score=True, n_jobs=-1)

        m = test_label_vector.shape[0]  ## the number of test examples
        ensembleScore = np.zeros(m)  ## the ensembled predict_y_proba: average all the predict_y_proba
        ensembleLable = np.zeros(m, dtype=int)  

        # print("training meta-path 1...")
        clf1.fit(train_feature_matrix_1, train_label_vector)
        # print("testing meta-path 1...")
        predict_y_proba = clf1.predict_proba(test_feature_matrix_1)[:, 1]
        predict_y = clf1.predict(test_feature_matrix_1)
        # print("evaluating meta-path 1...")
        AUPR = average_precision_score(test_label_vector, predict_y_proba)
        AUC = roc_auc_score(test_label_vector, predict_y_proba)
        recall, precision, thresholds_pr = precision_recall_curve(test_label_vector, predict_y_proba)
        fpr, tpr, thresholds_roc = roc_curve(test_label_vector, predict_y_proba)
        f1 = 2 * (precision * recall) / (precision + recall)
        max_index = np.argwhere(f1 == max(f1))
        PRE = precision[max_index]
        REC = recall[max_index]
        F1 = f1[max_index]
        threshold = thresholds_pr[max_index]
        y_pre = np.copy(predict_y_proba)
        y_pre[y_pre > threshold[0][0]] = 1
        y_pre[y_pre < threshold[0][0]] = 0
        y_pre = y_pre.astype(int)
        ACC = accuracy_score(test_label_vector, y_pre)
        MCC = matthews_corrcoef(test_label_vector, y_pre)
        metric = np.array((AUPR, AUC, PRE, REC, ACC, MCC, F1))
        metric_avg[0, :] = metric_avg[0, :] + metric

        for i in range(0, m):
            ensembleScore[i] += predict_y_proba[i]

        # print("training meta-path 2...")
        clf2.fit(train_feature_matrix_2, train_label_vector)
        # print("testing meta-path 2...")
        predict_y_proba = clf2.predict_proba(test_feature_matrix_2)[:, 1]
        predict_y = clf2.predict(test_feature_matrix_2)
        # print("evaluating meta-path 2...")
        AUPR = average_precision_score(test_label_vector, predict_y_proba)
        AUC = roc_auc_score(test_label_vector, predict_y_proba)
        recall, precision, thresholds_pr = precision_recall_curve(test_label_vector, predict_y_proba)
        fpr, tpr, thresholds_roc = roc_curve(test_label_vector, predict_y_proba)
        f1 = 2 * (precision * recall) / (precision + recall)
        max_index = np.argwhere(f1 == max(f1))
        PRE = precision[max_index]
        REC = recall[max_index]
        F1 = f1[max_index]
        threshold = thresholds_pr[max_index]
        y_pre = np.copy(predict_y_proba)
        y_pre[y_pre > threshold[0][0]] = 1
        y_pre[y_pre < threshold[0][0]] = 0
        y_pre = y_pre.astype(int)
        ACC = accuracy_score(test_label_vector, y_pre)
        MCC = matthews_corrcoef(test_label_vector, y_pre)
        metric = np.array((AUPR, AUC, PRE, REC, ACC, MCC, F1))
        metric_avg[1, :] = metric_avg[1, :] + metric

        for i in range(0, m):
            ensembleScore[i] += predict_y_proba[i]

        # print("traing meta-path 3...")
        clf3.fit(train_feature_matrix_3, train_label_vector)
        # print("testing meta-path 3...")
        predict_y_proba = clf3.predict_proba(test_feature_matrix_3)[:, 1]
        predict_y = clf3.predict(test_feature_matrix_3)
        # print("evaluating meta-path 3...")
        AUPR = average_precision_score(test_label_vector, predict_y_proba)
        AUC = roc_auc_score(test_label_vector, predict_y_proba)
        recall, precision, thresholds_pr = precision_recall_curve(test_label_vector, predict_y_proba)
        fpr, tpr, thresholds_roc = roc_curve(test_label_vector, predict_y_proba)
        f1 = 2 * (precision * recall) / (precision + recall)
        max_index = np.argwhere(f1 == max(f1))
        PRE = precision[max_index]
        REC = recall[max_index]
        F1 = f1[max_index]
        threshold = thresholds_pr[max_index]
        y_pre = np.copy(predict_y_proba)
        y_pre[y_pre > threshold[0][0]] = 1
        y_pre[y_pre < threshold[0][0]] = 0
        y_pre = y_pre.astype(int)
        ACC = accuracy_score(test_label_vector, y_pre)
        MCC = matthews_corrcoef(test_label_vector, y_pre)
        metric = np.array((AUPR, AUC, PRE, REC, ACC, MCC, F1))
        metric_avg[2, :] = metric_avg[2, :] + metric

        for i in range(0, m):
            ensembleScore[i] += predict_y_proba[i]

        # print("training meta-path 4...")
        clf4.fit(train_feature_matrix_4, train_label_vector)
        # print("testing meta-path 4...")
        predict_y_proba = clf4.predict_proba(test_feature_matrix_4)[:, 1]
        predict_y = clf4.predict(test_feature_matrix_4)
        # print("evaluating meta-path 4...")
        AUPR = average_precision_score(test_label_vector, predict_y_proba)
        AUC = roc_auc_score(test_label_vector, predict_y_proba)
        recall, precision, thresholds_pr = precision_recall_curve(test_label_vector, predict_y_proba)
        fpr, tpr, thresholds_roc = roc_curve(test_label_vector, predict_y_proba)
        f1 = 2 * (precision * recall) / (precision + recall)
        max_index = np.argwhere(f1 == max(f1))
        PRE = precision[max_index]
        REC = recall[max_index]
        F1 = f1[max_index]
        threshold = thresholds_pr[max_index]
        y_pre = np.copy(predict_y_proba)
        y_pre[y_pre > threshold[0][0]] = 1
        y_pre[y_pre < threshold[0][0]] = 0
        y_pre = y_pre.astype(int)
        ACC = accuracy_score(test_label_vector, y_pre)
        MCC = matthews_corrcoef(test_label_vector, y_pre)
        metric = np.array((AUPR, AUC, PRE, REC, ACC, MCC, F1))
        metric_avg[3, :] = metric_avg[3, :] + metric

        for i in range(0, m):
            ensembleScore[i] += predict_y_proba[i]

        # print("training meta-path 5...")
        clf5.fit(train_feature_matrix_5, train_label_vector)
        # print("testing meta-path 5...")
        predict_y_proba = clf5.predict_proba(test_feature_matrix_5)[:, 1]
        predict_y = clf5.predict(test_feature_matrix_5)
        # print("evaluating meta-path 5...")
        AUPR = average_precision_score(test_label_vector, predict_y_proba)
        AUC = roc_auc_score(test_label_vector, predict_y_proba)
        recall, precision, thresholds_pr = precision_recall_curve(test_label_vector, predict_y_proba)
        fpr, tpr, thresholds_roc = roc_curve(test_label_vector, predict_y_proba)
        f1 = 2 * (precision * recall) / (precision + recall)
        max_index = np.argwhere(f1 == max(f1))
        PRE = precision[max_index]
        REC = recall[max_index]
        F1 = f1[max_index]
        threshold = thresholds_pr[max_index]
        y_pre = np.copy(predict_y_proba)
        y_pre[y_pre > threshold[0][0]] = 1
        y_pre[y_pre < threshold[0][0]] = 0
        y_pre = y_pre.astype(int)
        ACC = accuracy_score(test_label_vector, y_pre)
        MCC = matthews_corrcoef(test_label_vector, y_pre)
        metric = np.array((AUPR, AUC, PRE, REC, ACC, MCC, F1))
        metric_avg[4, :] = metric_avg[4, :] + metric

        for i in range(0, m):
            ensembleScore[i] += predict_y_proba[i]

        ### ensemble all the classifiers built on above meta-paths

        for i in range(0, m):
            ensembleScore[i] = ensembleScore[i] / 5

        AUPR = average_precision_score(test_label_vector, ensembleScore)
        AUC = roc_auc_score(test_label_vector, ensembleScore)
        recall, precision, thresholds_pr = precision_recall_curve(test_label_vector, ensembleScore)
        fpr, tpr, thresholds_roc = roc_curve(test_label_vector, ensembleScore)
        f1 = 2 * (precision * recall) / (precision + recall)
        max_index = np.argwhere(f1 == max(f1))
        PRE = precision[max_index]
        REC = recall[max_index]
        F1 = f1[max_index]
        threshold = thresholds_pr[max_index]
        y_pre = np.copy(ensembleScore)
        y_pre[y_pre > threshold[0][0]] = 1
        y_pre[y_pre < threshold[0][0]] = 0
        y_pre = y_pre.astype(int)
        ACC = accuracy_score(test_label_vector, y_pre)
        MCC = matthews_corrcoef(test_label_vector, y_pre)
        metric = np.array((AUPR, AUC, PRE, REC, ACC, MCC, F1))
        metric_avg[5, :] = metric_avg[5, :] + metric

        recall, precision, thresholds_pr = precision_recall_curve(test_label_vector, ensembleScore)
        fpr, tpr, thresholds_roc = roc_curve(test_label_vector, ensembleScore)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
    
        precisions.append(interp(mean_recall, recall, precision))
        auprs.append(AUPR)
        aucs.append(AUC)
    
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = sum(aucs) / 5
    
    mean_precision = np.mean(precisions, axis=0)
    mean_aupr = sum(auprs) / 5
    
    np.savetxt('mean_fpr_EMP-SVD', mean_fpr)
    np.savetxt('mean_tpr_EMP-SVD', mean_tpr)
    np.savetxt('mean_precision_EMP-SVD', mean_precision)
    np.savetxt('mean_recall_EMP-SVD', mean_recall)
    np.savetxt('mean_auc_aupr_EMP-SVD', (mean_auc, mean_aupr))

    print("**********************************************************************************************")
    print("AUPR	AUC	PRE	REC	ACC	MCC	F1")
    print(metric_avg / kf.n_splits)
    print("**********************************************************************************************")

if __name__ == "__main__":
    drug_disease_matrix = np.loadtxt('../data/drugDiseaseInteraction.txt', delimiter='\t', dtype=int)
    drug_protein_matrix = np.loadtxt('../data/drugProteinInteraction.txt', delimiter='\t', dtype=int)
    disease_protein_matrix = np.loadtxt('../data/diseaseProteinInteraction.txt', delimiter='\t', dtype=int)
    drug_similarity_matrix = np.loadtxt('../data/drugSimilarity.txt', delimiter='\t', dtype=float)
    protein_similarity_matrix = np.loadtxt('../data/proteinSimilarity.txt', delimiter='\t', dtype=float)
    disease_similarity_matrix = np.loadtxt('../data/diseaseSimilarity.txt', delimiter='\t', dtype=float)

    print("Below are the performances of each base classifier and the final ensemble classifier EMP-SVD:")
    print("meta-path-1:drug->disease")
    print("meta-path-2:drug->protein->disease")
    print("meta-path-3:drug->protein->drug->disease")
    print("meta-path-4:drug->disease->drug->disease")
    print("meta-path-5:drug->disease->protein->disease")
    print("ensemble classifier: EMP-SVD")
 



    # for latent_feature_percent in np.arange(0.01,0.21,0.01):  


    latent_feature_percent = 0.03  ## 0.03
    print()
    print()
    print('latent_feature_percent=%s' % (str(latent_feature_percent)) )
    start = time.clock()
    EMP_SVD(drug_disease_matrix, drug_protein_matrix, disease_protein_matrix, latent_feature_percent)
    end = time.clock()
    print('Runing time:\t%s\tseconds' % (end - start))
