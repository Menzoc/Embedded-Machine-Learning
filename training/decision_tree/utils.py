import pandas as pd
import numpy as np
import csv

CSV_HEADER = ["node_id", "threshold", "feature_id", "left_children_id", "right_children_id", "class"]


def tree_to_csv(tree, class_names, csv_file):
    # node_list = np.zeros(shape=(tree.tree_.node_count, len(CSV_HEADER)))
    node_list = []
    for node_i in range(tree.tree_.node_count):
        node_list.append([node_i,
                          tree.tree_.threshold[node_i],
                          tree.tree_.feature[node_i],
                          tree.tree_.children_left[node_i],
                          tree.tree_.children_right[node_i],
                          class_names[np.argmax((tree.tree_.value[node_i]))]])

    tree_df = pd.DataFrame(node_list, columns=CSV_HEADER)
    tree_df[CSV_HEADER[0]] = tree_df[CSV_HEADER[0]].astype(int)
    tree_df[CSV_HEADER[1]] = tree_df[CSV_HEADER[1]].astype(float)
    tree_df[CSV_HEADER[2]] = tree_df[CSV_HEADER[2]].astype(int)
    tree_df[CSV_HEADER[3]] = tree_df[CSV_HEADER[3]].astype(int)
    tree_df[CSV_HEADER[4]] = tree_df[CSV_HEADER[4]].astype(int)
    tree_df[CSV_HEADER[5]] = tree_df[CSV_HEADER[5]].astype(str)

    pd.DataFrame([CSV_HEADER]).to_csv(csv_file, mode='w', index=False, header=False, quoting=csv.QUOTE_NONE)
    tree_df.to_csv(csv_file, mode='a', index=False, header=False, quoting=csv.QUOTE_NONNUMERIC)
