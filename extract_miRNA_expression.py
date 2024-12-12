import os
import os.path
import gzip
import numpy as np
import pandas as pd

# # 导入CSV安装包
# import csv
# 导入json包
import json

# # 创建文件对象
# file_write_name = open('tcga_table_acc.csv', 'w', newline='', encoding='utf-8')
# # 基于文件对象构建 csv写入对象
# csv_writer = csv.writer(file_write_name)

path = []
path.append("./Origin_Data/miRNA/TCGA-ACC")
path.append("./Origin_Data/miRNA/TCGA-BLCA")
path.append("./Origin_Data/miRNA/TCGA-BRCA")
path.append("./Origin_Data/miRNA/TCGA-CESC")
path.append("./Origin_Data/miRNA/TCGA-CHOL")
path.append("./Origin_Data/miRNA/TCGA-COAD")
path.append("./Origin_Data/miRNA/TCGA-DLBC")
path.append("./Origin_Data/miRNA/TCGA-ESCA")
path.append("./Origin_Data/miRNA/TCGA-GBM")
path.append("./Origin_Data/miRNA/TCGA-HNSC")
path.append("./Origin_Data/miRNA/TCGA-KICH")
path.append("./Origin_Data/miRNA/TCGA-KIRC")
path.append("./Origin_Data/miRNA/TCGA-KIRP")
path.append("./Origin_Data/miRNA/TCGA-LAML")
path.append("./Origin_Data/miRNA/TCGA-LGG")
path.append("./Origin_Data/miRNA/TCGA-LIHC")
path.append("./Origin_Data/miRNA/TCGA-LUAD")
path.append("./Origin_Data/miRNA/TCGA-LUSC")
path.append("./Origin_Data/miRNA/TCGA-MESO")
path.append("./Origin_Data/miRNA/TCGA-OV")
path.append("./Origin_Data/miRNA/TCGA-PAAD")
path.append("./Origin_Data/miRNA/TCGA-PCPG")
path.append("./Origin_Data/miRNA/TCGA-PRAD")
path.append("./Origin_Data/miRNA/TCGA-READ")
path.append("./Origin_Data/miRNA/TCGA-SARC")
path.append("./Origin_Data/miRNA/TCGA-SKCM")
path.append("./Origin_Data/miRNA/TCGA-STAD")
path.append("./Origin_Data/miRNA/TCGA-TGCT")
path.append("./Origin_Data/miRNA/TCGA-THCA")
path.append("./Origin_Data/miRNA/TCGA-THYM")
path.append("./Origin_Data/miRNA/TCGA-UCEC")
path.append("./Origin_Data/miRNA/TCGA-UCS")
path.append("./Origin_Data/miRNA/TCGA-UVM")

for type_path in path:
    print(len(type_path.split('/')))
    label = type_path.split('/')[3]
    print(label)
    metadata = open(type_path + "/metadata.cart.2022-05-26.json", encoding='utf-8')
    json_data = json.load(metadata)
    file_name_list = []
    file_id_list = []
    submitter_id_list = []
    for list_data in json_data:
        file_name = list_data['file_name']
        if file_name is None:
            file_name = "NA"
        file_name_list.append(file_name)
        file_id = list_data['file_id']
        if file_id is None:
            file_id = "NA"
        file_id_list.append(file_id)
        submitter_id = list_data['associated_entities'][0]['entity_submitter_id']
        if submitter_id is None:
            submitter_id = "NA"
        submitter_id_list.append(submitter_id)
    data = {"file_name": file_name_list, "file_id": file_id_list, "submitter_id": submitter_id_list}
    meta_dataframe = pd.DataFrame(data)
    count = 0
    flag = True
    for root, dirs, files in os.walk(type_path + "/harmonized/Transcriptome_Profiling/miRNA_Expression_Quantification"):
        for file in files:
            file_path = str(os.path.join(root, file).encode('utf-8'), 'utf-8')
            col_list_value = []
            if file_path[-3:] == 'txt':
                # print(file_path)
                file_id = file_path.split('/')[7]
                # print(file_id)
                file_name = file_path.split('/')[8]
                # print(file_name)
                if flag:
                    col_list_name = []
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f.readlines():
                            col_name = line.split('\t')[0]
                        # print(col_name)
                            if col_name[:3] == 'hsa':
                                col_list_name.append(col_name)
                    col_list_name.append('label')
                    col_list_name.append('file_id')
                    col_list_name.append('file_name')
                    # csv_writer.writerow(col_list_name)
                    gene_dataframe = pd.DataFrame(columns=col_list_name)
                    flag = False
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f.readlines():
                        col_name = line.split('\t')[0]
                        if col_name[:3] == 'hsa':
                            col_value = line.split('\t')[1].replace('\n', '')
                            if col_value is None:
                                col_value = "NA"
                            # print(col_value)
                            col_list_value.append(col_value)
                    col_list_value.append(label)
                    col_list_value.append(file_id)
                    col_list_value.append(file_name)
                    # csv_writer.writerow(col_list_value)
                    count = count + 1
                    gene_dataframe.loc[count] = col_list_value
    print(label + ":" + str(count))
    #             print(gene_dataframe)
    # if count == 5:
    #     break
    # print(count)
    result = pd.merge(gene_dataframe, meta_dataframe, how="inner", on=['file_id', 'file_name'])
    result.to_csv("./miRNA_source_to_csv/" + label + ".csv")
