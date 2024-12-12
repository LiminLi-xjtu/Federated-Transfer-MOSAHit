import os
import pandas as pd
import numpy as np

# 导入CSV安装包
# import csv

source_file_path = './RNASeq_source_to_csv'
target_file_path = './RNASeq_variable_filter_csv'
flag = True
for root, dirs, files in os.walk(source_file_path):
    for file in files:
        cancer_type = file.split('.')[0].split('-')[1]
        if True:
            file_path = str(os.path.join(root, file).encode('utf-8'), 'utf-8')
            temp_class = pd.read_csv(file_path, sep=',')

            if flag:
                temp = temp_class
                flag = False
            else:
                temp = temp.append(temp_class)
print("gene_feature的数目：" + str(temp.shape[1]))
# temp = temp.dropna(axis=0, how='any')
temp = temp.dropna(axis=1, thresh=temp.shape[0] * 0.9)
print("删除None后" + " gene_feature的数目：" + str(temp.shape[1]))

print(cancer_type + "样本的数目：" + str(temp.shape[0]))
# temp = temp.dropna(axis=0, how='any')
temp = temp.dropna(axis=0, thresh=temp.shape[1] * 0.5)
print("删除None后" + "样本的数目：" + str(temp.shape[0]))

# # 中位数填补缺失值
# columns_name = list(temp.columns)
# for col in columns_name:
#     if col[:2] == 'EN':
#         temp[col] = np.squeeze(np.array(temp[[col]].fillna(temp[col].median())))

# 均值填补缺失值
# columns_name = list(temp.columns)
# for col in columns_name:
#     if col[:2] == 'EN':
#         temp[col] = np.squeeze(np.array(temp[[col]].fillna(temp[col].mean())))
columns_name = list(temp.columns)
print(columns_name[0])
mean_col_name = columns_name[1:-4]
temp[mean_col_name] = temp[mean_col_name].fillna(temp[mean_col_name].mean())


# # 最近邻填补缺失值
# from sklearn.impute import KNNImputer
# imputer = KNNImputer(n_neighbors=2)
# columns_name = list(temp.columns)
# print(columns_name[0])
# KNN_col_name = columns_name[1:-4]
# temp_knn = imputer.fit_transform(temp[KNN_col_name])
# temp[KNN_col_name] = temp_knn

var_col_name = []
for col in columns_name:
    # if col != 'label' and col != 'file_id' and col != 'file_name' and col != 'submitter_id':
    if col[: 2] == 'EN':
        with open('gene process' + '.log', 'a') as f:
            f.writelines('gene_name:' + col + '\n')
        temp[col] = temp[col].astype(float)
        var = np.var(np.log2(np.array(temp[col]).squeeze() + 1))
        mean = np.mean(np.log2(np.array(temp[col]).squeeze() + 1))

        if var > 1.2 and mean > 2:
            var_col_name.append(col)
        temp[col] = np.log2(np.array(temp[col]).squeeze() + 1).tolist()

print(var_col_name)
temp['submitter_id'] = temp['submitter_id'].str[0:12]
temp = temp.drop_duplicates('submitter_id')
var_col_name.append('label')
# var_col_name.append('file_id')
# var_col_name.append('file_name')
var_col_name.append('submitter_id')
print(temp['label'])
print("筛选后gene的数目：" + str(len(var_col_name) - 2))
df_RNASeq = temp[var_col_name]
df_RNASeq[df_RNASeq['label'].isin(["TCGA-ACC"])].to_csv(target_file_path + "/TCGA-ACC.csv", index=False, header=True)
df_RNASeq[df_RNASeq['label'].isin(["TCGA-BLCA"])].to_csv(target_file_path + "/TCGA-BLCA.csv", index=False, header=True)
df_RNASeq[df_RNASeq['label'].isin(["TCGA-BRCA"])].to_csv(target_file_path + "/TCGA-BRCA.csv", index=False, header=True)
df_RNASeq[df_RNASeq['label'].isin(["TCGA-CESC"])].to_csv(target_file_path + "/TCGA-CESC.csv", index=False, header=True)
df_RNASeq[df_RNASeq['label'].isin(["TCGA-CHOL"])].to_csv(target_file_path + "/TCGA-CHOL.csv", index=False, header=True)
df_RNASeq[df_RNASeq['label'].isin(["TCGA-COAD"])].to_csv(target_file_path + "/TCGA-COAD.csv", index=False, header=True)
df_RNASeq[df_RNASeq['label'].isin(["TCGA-DLBC"])].to_csv(target_file_path + "/TCGA-DLBC.csv", index=False, header=True)
df_RNASeq[df_RNASeq['label'].isin(["TCGA-ESCA"])].to_csv(target_file_path + "/TCGA-ESCA.csv", index=False, header=True)
df_RNASeq[df_RNASeq['label'].isin(["TCGA-GBM"])].to_csv(target_file_path + "/TCGA-GBM.csv", index=False, header=True)
df_RNASeq[df_RNASeq['label'].isin(["TCGA-HNSC"])].to_csv(target_file_path + "/TCGA-HNSC.csv", index=False, header=True)
df_RNASeq[df_RNASeq['label'].isin(["TCGA-KICH"])].to_csv(target_file_path + "/TCGA-KICH.csv", index=False, header=True)
df_RNASeq[df_RNASeq['label'].isin(["TCGA-KIRC"])].to_csv(target_file_path + "/TCGA-KIRC.csv", index=False, header=True)
df_RNASeq[df_RNASeq['label'].isin(["TCGA-KIRP"])].to_csv(target_file_path + "/TCGA-KIRP.csv", index=False, header=True)
df_RNASeq[df_RNASeq['label'].isin(["TCGA-LAML"])].to_csv(target_file_path + "/TCGA-LAML.csv", index=False, header=True)
df_RNASeq[df_RNASeq['label'].isin(["TCGA-LGG"])].to_csv(target_file_path + "/TCGA-LGG.csv", index=False, header=True)
df_RNASeq[df_RNASeq['label'].isin(["TCGA-LIHC"])].to_csv(target_file_path + "/TCGA-LIHC.csv", index=False, header=True)
df_RNASeq[df_RNASeq['label'].isin(["TCGA-LUAD"])].to_csv(target_file_path + "/TCGA-LUAD.csv", index=False, header=True)
df_RNASeq[df_RNASeq['label'].isin(["TCGA-LUSC"])].to_csv(target_file_path + "/TCGA-LUSC.csv", index=False, header=True)
df_RNASeq[df_RNASeq['label'].isin(["TCGA-MESO"])].to_csv(target_file_path + "/TCGA-MESO.csv", index=False, header=True)
df_RNASeq[df_RNASeq['label'].isin(["TCGA-OV"])].to_csv(target_file_path + "/TCGA-OV.csv", index=False, header=True)
df_RNASeq[df_RNASeq['label'].isin(["TCGA-PAAD"])].to_csv(target_file_path + "/TCGA-PAAD.csv", index=False, header=True)
df_RNASeq[df_RNASeq['label'].isin(["TCGA-PCPG"])].to_csv(target_file_path + "/TCGA-PCPG.csv", index=False, header=True)
df_RNASeq[df_RNASeq['label'].isin(["TCGA-PRAD"])].to_csv(target_file_path + "/TCGA-PRAD.csv", index=False, header=True)
df_RNASeq[df_RNASeq['label'].isin(["TCGA-READ"])].to_csv(target_file_path + "/TCGA-READ.csv", index=False, header=True)
df_RNASeq[df_RNASeq['label'].isin(["TCGA-SARC"])].to_csv(target_file_path + "/TCGA-SARC.csv", index=False, header=True)
df_RNASeq[df_RNASeq['label'].isin(["TCGA-SKCM"])].to_csv(target_file_path + "/TCGA-SKCM.csv", index=False, header=True)
df_RNASeq[df_RNASeq['label'].isin(["TCGA-STAD"])].to_csv(target_file_path + "/TCGA-STAD.csv", index=False, header=True)
df_RNASeq[df_RNASeq['label'].isin(["TCGA-TGCT"])].to_csv(target_file_path + "/TCGA-TGCT.csv", index=False, header=True)
df_RNASeq[df_RNASeq['label'].isin(["TCGA-THCA"])].to_csv(target_file_path + "/TCGA-THCA.csv", index=False, header=True)
df_RNASeq[df_RNASeq['label'].isin(["TCGA-THYM"])].to_csv(target_file_path + "/TCGA-THYM.csv", index=False, header=True)
df_RNASeq[df_RNASeq['label'].isin(["TCGA-UCEC"])].to_csv(target_file_path + "/TCGA-UCEC.csv", index=False, header=True)
df_RNASeq[df_RNASeq['label'].isin(["TCGA-UCS"])].to_csv(target_file_path + "/TCGA-UCS.csv", index=False, header=True)
df_RNASeq[df_RNASeq['label'].isin(["TCGA-UVM"])].to_csv(target_file_path + "/TCGA-UVM.csv", index=False, header=True)


