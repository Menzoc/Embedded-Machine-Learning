import pandas as pd
import numpy as np
import csv

def svm_to_csv(svm, class_names, csv_file):
    CSV_HEADER = ["positive_class", "negative_class", "intercept"] + list("coeff{}".format(i) for i in range(len(svm.coef_[0])))
    # resukt_left, resutl_right (top/bot)?

    lower_class_list = []
    upper_class_list = []
    for i in range(len(svm.classes_)):
        for j in range(i+1, len(svm.classes_)):
            lower_class_list.append(svm.classes_[i])
            upper_class_list.append(svm.classes_[j])

    value_list = []
    for value_i in range(len(svm.intercept_)):
        value_list.append(np.concatenate([[lower_class_list[value_i], upper_class_list[value_i],  svm.intercept_[value_i]], svm.coef_[value_i]]))
        
    svm_df = pd.DataFrame(value_list, columns=CSV_HEADER)
    svm_df[CSV_HEADER[0]] = svm_df[CSV_HEADER[0]].astype(str)
    svm_df[CSV_HEADER[1]] = svm_df[CSV_HEADER[1]].astype(str)
    svm_df[CSV_HEADER[2:]] = svm_df[CSV_HEADER[2:]].astype(float)

    pd.DataFrame([CSV_HEADER]).to_csv(csv_file, mode='w', index=False, header=False, quoting=csv.QUOTE_NONE)
    svm_df.to_csv(csv_file, mode='a', index=False, header=False, quoting=csv.QUOTE_NONNUMERIC)

    return CSV_HEADER