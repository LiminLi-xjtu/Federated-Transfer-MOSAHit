import os
import pandas as pd
import numpy as np

# 导入CSV安装包
# import csv

# flag = True
for cancer in ['TCGA-ACC', 'TCGA-BLCA', 'TCGA-BRCA', 'TCGA-CESC', 'TCGA-CHOL', 'TCGA-COAD', 'TCGA-DLBC',
               'TCGA-ESCA', 'TCGA-GBM', 'TCGA-HNSC', 'TCGA-KICH', 'TCGA-KIRC', 'TCGA-KIRP', 'TCGA-LAML',
               'TCGA-LGG', 'TCGA-LIHC', 'TCGA-LUAD', 'TCGA-LUSC', 'TCGA-MESO', 'TCGA-OV', 'TCGA-PAAD',
               'TCGA-PCPG', 'TCGA-PRAD', 'TCGA-READ', 'TCGA-SARC', 'TCGA-SKCM', 'TCGA-STAD', 'TCGA-TGCT',
               'TCGA-THCA', 'TCGA-THYM', 'TCGA-UCEC', 'TCGA-UCS', 'TCGA-UVM']:

    source_file_path = 'Origin_Data/miRNA/' + cancer + '.mirna.tsv.gz'
    target_file_path = 'miRNA_source_to_csv/' + cancer + '.csv'
    import pandas as pd

    df = pd.read_csv(source_file_path, delimiter='\t')
    # print(df.shape)
    # print(df.iloc[:, 1:])
    # print(df.iloc[:, 1:].transpose())
    # print(df.transpose())

    # print(Ensembl_ID)
    dataframe_tran = df.transpose()
    dataframe_tran = dataframe_tran.reset_index()
    print(dataframe_tran)
    ID = dataframe_tran.iloc[0, :].tolist()
    temp_dataframe = dataframe_tran.iloc[1:, :]
    temp_dataframe.columns = ID
    print(temp_dataframe)
    temp_dataframe.to_csv(target_file_path, index=False, header=True)
