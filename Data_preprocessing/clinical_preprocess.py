import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# 导入CSV安装包
import csv

source_path = "./Origin_Data/Clinical/"
target_path = "./clinical/"
for root, dirs, files in os.walk(source_path):
    for file in files:
        file_path = str(os.path.join(root, file).encode('utf-8'), 'utf-8')
        if file.split('.')[1] == 'csv':
            cancer = file.split('.')[0].split('-')[1]
            print('**********')
            temp_analysis = pd.read_csv(file_path, sep=',')

            # univariate and multivariate Cox analysis
            if cancer == 'BRCA':
                temp_analysis = temp_analysis[
                    ["submitter_id", "vital_status", "days_to_last_follow_up", "days_to_death",
                     "age", "ajcc_pathologic_stage", "ajcc_pathologic_t", "ajcc_pathologic_n", "ajcc_pathologic_m"]]
                temp_analysis.loc[(temp_analysis.vital_status == "Dead"), 'survival_status'] = 1
                temp_analysis.loc[(temp_analysis.vital_status == "Alive"), 'survival_status'] = 0
                temp_analysis.loc[
                    (temp_analysis.vital_status == "Alive") & (temp_analysis.days_to_last_follow_up != "NA"),
                    'survival_time'] \
                    = temp_analysis.days_to_last_follow_up
                temp_analysis.loc[(temp_analysis.vital_status == "Dead") & (temp_analysis.days_to_death != "NA"),
                                  'survival_time'] = temp_analysis.days_to_death
                temp_analysis["survival_status"] = temp_analysis["survival_status"].astype(float)
                temp_analysis["survival_time"] = temp_analysis["survival_time"].astype(float)
                temp_analysis = temp_analysis[temp_analysis["survival_time"] >= 0]

                temp_analysis.loc[(temp_analysis.ajcc_pathologic_stage == "Stage I"), 'grade'] = 0
                temp_analysis.loc[(temp_analysis.ajcc_pathologic_stage == "Stage IA"), 'grade'] = 0
                temp_analysis.loc[(temp_analysis.ajcc_pathologic_stage == "Stage IB"), 'grade'] = 0
                temp_analysis.loc[(temp_analysis.ajcc_pathologic_stage == "Stage II"), 'grade'] = 0
                temp_analysis.loc[(temp_analysis.ajcc_pathologic_stage == "Stage IIA"), 'grade'] = 0
                temp_analysis.loc[(temp_analysis.ajcc_pathologic_stage == "Stage IIB"), 'grade'] = 0

                temp_analysis.loc[(temp_analysis.ajcc_pathologic_stage == "Stage III"), 'grade'] = 1
                temp_analysis.loc[(temp_analysis.ajcc_pathologic_stage == "Stage IIIA"), 'grade'] = 1
                temp_analysis.loc[(temp_analysis.ajcc_pathologic_stage == "Stage IIIB"), 'grade'] = 1
                temp_analysis.loc[(temp_analysis.ajcc_pathologic_stage == "Stage IIIC"), 'grade'] = 1
                temp_analysis.loc[(temp_analysis.ajcc_pathologic_stage == "Stage IV"), 'grade'] = 1

                temp_analysis.loc[(temp_analysis.ajcc_pathologic_t == "T1"), 'stage_t'] = 0
                temp_analysis.loc[(temp_analysis.ajcc_pathologic_t == "T1a"), 'stage_t'] = 0
                temp_analysis.loc[(temp_analysis.ajcc_pathologic_t == "T1b"), 'stage_t'] = 0
                temp_analysis.loc[(temp_analysis.ajcc_pathologic_t == "T1c"), 'stage_t'] = 0
                temp_analysis.loc[(temp_analysis.ajcc_pathologic_t == "T2"), 'stage_t'] = 0
                temp_analysis.loc[(temp_analysis.ajcc_pathologic_t == "T2a"), 'stage_t'] = 0
                temp_analysis.loc[(temp_analysis.ajcc_pathologic_t == "T2b"), 'stage_t'] = 0

                temp_analysis.loc[(temp_analysis.ajcc_pathologic_t == "T3"), 'stage_t'] = 1
                temp_analysis.loc[(temp_analysis.ajcc_pathologic_t == "T3a"), 'stage_t'] = 1
                temp_analysis.loc[(temp_analysis.ajcc_pathologic_t == "T4"), 'stage_t'] = 1
                temp_analysis.loc[(temp_analysis.ajcc_pathologic_t == "T4b"), 'stage_t'] = 1
                temp_analysis.loc[(temp_analysis.ajcc_pathologic_t == "T4d"), 'stage_t'] = 1

                temp_analysis.loc[(temp_analysis.ajcc_pathologic_n == "N0"), 'stage_n'] = 0
                temp_analysis.loc[(temp_analysis.ajcc_pathologic_n == "N0 (i-)"), 'stage_n'] = 0
                temp_analysis.loc[(temp_analysis.ajcc_pathologic_n == "N0 (i+)"), 'stage_n'] = 0
                temp_analysis.loc[(temp_analysis.ajcc_pathologic_n == "N0 (mol+)"), 'stage_n'] = 0
                temp_analysis.loc[(temp_analysis.ajcc_pathologic_n == "N1"), 'stage_n'] = 0
                temp_analysis.loc[(temp_analysis.ajcc_pathologic_n == "N1a"), 'stage_n'] = 0
                temp_analysis.loc[(temp_analysis.ajcc_pathologic_n == "N1b"), 'stage_n'] = 0
                temp_analysis.loc[(temp_analysis.ajcc_pathologic_n == "N1c"), 'stage_n'] = 0
                temp_analysis.loc[(temp_analysis.ajcc_pathologic_n == "N1mi"), 'stage_n'] = 0

                temp_analysis.loc[(temp_analysis.ajcc_pathologic_n == "N2"), 'stage_n'] = 1
                temp_analysis.loc[(temp_analysis.ajcc_pathologic_n == "N2a"), 'stage_n'] = 1
                temp_analysis.loc[(temp_analysis.ajcc_pathologic_n == "N3"), 'stage_n'] = 1
                temp_analysis.loc[(temp_analysis.ajcc_pathologic_n == "N3a"), 'stage_n'] = 1
                temp_analysis.loc[(temp_analysis.ajcc_pathologic_n == "N3b"), 'stage_n'] = 1
                temp_analysis.loc[(temp_analysis.ajcc_pathologic_n == "N3c"), 'stage_n'] = 1

                temp_analysis.loc[(temp_analysis.ajcc_pathologic_m == "M0"), 'stage_m'] = 0
                temp_analysis.loc[(temp_analysis.ajcc_pathologic_m == "cM0 (i+)"), 'stage_m'] = 0
                temp_analysis.loc[(temp_analysis.ajcc_pathologic_m == "MX"), 'stage_m'] = 1
                temp_analysis.to_csv(target_path + file, index=False, header=True)
                temp_analysis = temp_analysis[["submitter_id", "survival_status", "survival_time", "age", "grade",
                                               "stage_t", "stage_n", "stage_m"]].dropna(axis=0,
                                                                                        how='any').drop_duplicates(
                    'submitter_id')
                temp_analysis.to_csv(target_path + file, index=False, header=True)
            else:
                temp_analysis = temp_analysis[
                    ["submitter_id", "vital_status", "days_to_last_follow_up", "days_to_death"]]
                temp_analysis.loc[(temp_analysis.vital_status == "Dead"), 'survival_status'] = 1
                temp_analysis.loc[(temp_analysis.vital_status == "Alive"), 'survival_status'] = 0

                temp_analysis.loc[
                    (temp_analysis.vital_status == "Alive") & (temp_analysis.days_to_last_follow_up != "NA"),
                    'survival_time'] \
                    = temp_analysis.days_to_last_follow_up

                temp_analysis.loc[(temp_analysis.vital_status == "Dead") & (temp_analysis.days_to_death != "NA"),
                                  'survival_time'] = temp_analysis.days_to_death
                # temp_analysis.loc[(temp_analysis.days_to_death == "NA"), 'days_to_death'] = 0
                temp_analysis["survival_status"] = temp_analysis["survival_status"].astype(float)
                temp_analysis["survival_time"] = temp_analysis["survival_time"].astype(float)
                temp_analysis = temp_analysis[temp_analysis["survival_time"] >= 0]
                print(temp_analysis[["submitter_id", "survival_status", "survival_time"]].dropna(axis=0,
                                                                                                 how='any').isnull().any())

                temp_analysis = temp_analysis[["submitter_id",
                                               "survival_status", "survival_time"]].dropna(axis=0,
                                                                                           how='any').drop_duplicates(
                    'submitter_id')
                temp_analysis.to_csv(target_path + file, index=False, header=True)
