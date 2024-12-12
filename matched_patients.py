import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# 导入CSV安装包

base_path = "F:\\Pycharm_Project\\FLMOSurv\\Datasets_and_Preprocessing\\data\\"
miRNA_path = base_path + "miRNA_variable_filter_csv"
RNASeq_path = base_path + "RNASeq_variable_filter_csv"
clinical_path = base_path + "clinical"
matched_patients_path = base_path + "matched_RNASeq_miRNA_clinical"

# head_flag = True
for root, dirs, files in os.walk(miRNA_path):
    for file in files:
        cancer = file.split('-')[1].split('.')[0]
        RNASeq_file_path = str(os.path.join(RNASeq_path, file).encode('utf-8'), 'utf-8')
        clinical_file_path = str(os.path.join(clinical_path, file).encode('utf-8'), 'utf-8')
        miRNA_file_path = str(os.path.join(miRNA_path, file).encode('utf-8'), 'utf-8')

        RNASeq_dataframe = pd.read_csv(RNASeq_file_path, sep=',')

        columns_name = list(RNASeq_dataframe.columns)
        for col in columns_name:
            if col[:2] == 'EN':
                log2_val = np.log2(np.array(RNASeq_dataframe[col]).squeeze() + 1)
                RNASeq_dataframe[col] = log2_val.tolist()

        RNASeq_dataframe['submitter_id'] = RNASeq_dataframe['submitter_id'].str[0:12]
        RNASeq_dataframe = RNASeq_dataframe.drop_duplicates('submitter_id')

        clinical_dataframe = pd.read_csv(clinical_file_path, sep=',')
        clinical_dataframe['submitter_id'] = clinical_dataframe['submitter_id'].str[0:12]
        clinical_dataframe = clinical_dataframe.drop_duplicates('submitter_id')

        miRNA_dataframe = pd.read_csv(miRNA_file_path, sep=',')

        for col in columns_name:
            if col[:3] == 'hsa':
                miRNA_dataframe[col] = miRNA_dataframe[col].astype(float)
                miRNA_dataframe[col] = np.log2(np.array(miRNA_dataframe[col]).squeeze() + 1).tolist()

        miRNA_dataframe['submitter_id'] = miRNA_dataframe['submitter_id'].str[0:12]
        miRNA_dataframe = miRNA_dataframe.drop_duplicates('submitter_id')

        id_inner_list = list(set(RNASeq_dataframe['submitter_id']) & set(clinical_dataframe['submitter_id'])
                   & set(miRNA_dataframe['submitter_id']))
        print("共享ID的个数:" + str(len(id_inner_list)))
        # RNA_survival = pd.merge(RNA_dataframe[RNA_dataframe["submitter_id"].isin(id_inner_list)],
        #                         survival_dataframe[survival_dataframe["submitter_id"].isin(id_inner_list)],
        #                         how="inner", on=["submitter_id"])
        # RNA_survival = RNA_survival.sort_values(by=["submitter_id"])
        # RNA_survival.drop(["submitter_id", "label", "survival_status", "survival_time"], 1).\
        #     to_csv("TCGA_Cancer_Type/" + file.split('.')[0] + "/RNA.csv", index=False, header=False)
        # RNA_survival["survival_status"].to_csv("TCGA_Cancer_Type/" + file.split('.')[0] +
        #                                        "/ystatus.csv", index=False, header=False)
        # RNA_survival["survival_time"].to_csv("TCGA_Cancer_Type/" + file.split('.')[0] +
        #                                      "/ytime.csv", index=False, header=False)
        # print(RNA_survival.shape)

        RNASeq_dataframe = RNASeq_dataframe[RNASeq_dataframe["submitter_id"].isin(id_inner_list)].sort_values(
            by=["submitter_id"])
        print("RNASeq样本的维度:")
        print(RNASeq_dataframe.shape)

        # RNA_dataframe.drop(["submitter_id", "label"], 1).to_csv("RNA_DNA_Cancer_Type/" + file.split('.')[0] +
        #                                                         "/RNA.csv", index=False, header=False)
        # print("RNA特征的个数：" + str(len(RNA_dataframe.columns) - 2))

        clinical_dataframe = clinical_dataframe[clinical_dataframe["submitter_id"].isin(id_inner_list)]. \
            sort_values(by=["submitter_id"])
        print("clinical样本的维度:")
        print(clinical_dataframe.shape)
        # survival_dataframe["survival_status"].to_csv("RNA_DNA_Cancer_Type/" + file.split('.')[0] +
        #                                              "/ystatus.csv", index=False, header=False)
        # survival_dataframe["survival_time"].to_csv("RNA_DNA_Cancer_Type/" + file.split('.')[0] +
        #                                            "/ytime.csv", index=False, header=False)
        miRNA_dataframe = miRNA_dataframe[miRNA_dataframe["submitter_id"].isin(id_inner_list)]\
            .sort_values(by=["submitter_id"])
        print("miRNA样本的维度:")
        print(miRNA_dataframe.shape)
        print("miRNA特征的个数：" + str(len(miRNA_dataframe.columns) - 2))

        if RNASeq_dataframe.shape[0] == 0:
            continue
        if miRNA_dataframe.shape[0] == 0:
            continue
        if clinical_dataframe.shape[0] == 0:
            continue

        save_path = matched_patients_path + '\\' + cancer
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        RNASeq_dataframe.drop(["submitter_id", "label"], 1).to_csv(save_path + "/RNASeq.csv", index=False, header=True)
        miRNA_dataframe.drop(["submitter_id", "label"], 1).to_csv(save_path + "/miRNA.csv", index=False, header=True)
        clinical_dataframe.drop(["submitter_id"], 1).to_csv(save_path + "/clinical.csv", index=False, header=True)

        # RNASeq_dataframe.drop(["submitter_id", "label"], 1).to_csv(save_path + "/RNASeq.csv", index=False, header=True)
        # clinical_dataframe["survival_status"].to_csv(save_path + "/ystatus.csv", index=False, header=True)
        # clinical_dataframe["survival_time"].to_csv(save_path + "/ytime.csv", index=False, header=True)
        # miRNA_dataframe.drop(["submitter_id", "label"], 1).to_csv(save_path + "/miRNA.csv", index=False, header=True)
