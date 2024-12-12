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
path.append("F:\\TCGA\\TCGA_官方网站\\Raw data\\gz_source_document\\miRNA\\TCGA-ACC")
path.append("F:\\TCGA\\TCGA_官方网站\\Raw data\\gz_source_document\\miRNA\\TCGA-BLCA")
path.append("F:\\TCGA\\TCGA_官方网站\\Raw data\\gz_source_document\\miRNA\\TCGA-BRCA")
path.append("F:\\TCGA\\TCGA_官方网站\\Raw data\\gz_source_document\\miRNA\\TCGA-CESC")
path.append("F:\\TCGA\\TCGA_官方网站\\Raw data\\gz_source_document\\miRNA\\TCGA-CHOL")
path.append("F:\\TCGA\\TCGA_官方网站\\Raw data\\gz_source_document\\miRNA\\TCGA-COAD")
path.append("F:\\TCGA\\TCGA_官方网站\\Raw data\\gz_source_document\\miRNA\\TCGA-DLBC")
path.append("F:\\TCGA\\TCGA_官方网站\\Raw data\\gz_source_document\\miRNA\\TCGA-ESCA")
path.append("F:\\TCGA\\TCGA_官方网站\\Raw data\\gz_source_document\\miRNA\\TCGA-GBM")
path.append("F:\\TCGA\\TCGA_官方网站\\Raw data\\gz_source_document\\miRNA\\TCGA-HNSC")
path.append("F:\\TCGA\\TCGA_官方网站\\Raw data\\gz_source_document\\miRNA\\TCGA-KICH")
path.append("F:\\TCGA\\TCGA_官方网站\\Raw data\\gz_source_document\\miRNA\\TCGA-KIRC")
path.append("F:\\TCGA\\TCGA_官方网站\\Raw data\\gz_source_document\\miRNA\\TCGA-KIRP")
path.append("F:\\TCGA\\TCGA_官方网站\\Raw data\\gz_source_document\\miRNA\\TCGA-LAML")
path.append("F:\\TCGA\\TCGA_官方网站\\Raw data\\gz_source_document\\miRNA\\TCGA-LGG")
path.append("F:\\TCGA\\TCGA_官方网站\\Raw data\\gz_source_document\\miRNA\\TCGA-LIHC")
path.append("F:\\TCGA\\TCGA_官方网站\\Raw data\\gz_source_document\\miRNA\\TCGA-LUAD")
path.append("F:\\TCGA\\TCGA_官方网站\\Raw data\\gz_source_document\\miRNA\\TCGA-LUSC")
path.append("F:\\TCGA\\TCGA_官方网站\\Raw data\\gz_source_document\\miRNA\\TCGA-MESO")
path.append("F:\\TCGA\\TCGA_官方网站\\Raw data\\gz_source_document\\miRNA\\TCGA-OV")
path.append("F:\\TCGA\\TCGA_官方网站\\Raw data\\gz_source_document\\miRNA\\TCGA-PAAD")
path.append("F:\\TCGA\\TCGA_官方网站\\Raw data\\gz_source_document\\miRNA\\TCGA-PCPG")
path.append("F:\\TCGA\\TCGA_官方网站\\Raw data\\gz_source_document\\miRNA\\TCGA-PRAD")
path.append("F:\\TCGA\\TCGA_官方网站\\Raw data\\gz_source_document\\miRNA\\TCGA-READ")
path.append("F:\\TCGA\\TCGA_官方网站\\Raw data\\gz_source_document\\miRNA\\TCGA-SARC")
path.append("F:\\TCGA\\TCGA_官方网站\\Raw data\\gz_source_document\\miRNA\\TCGA-SKCM")
path.append("F:\\TCGA\\TCGA_官方网站\\Raw data\\gz_source_document\\miRNA\\TCGA-STAD")
path.append("F:\\TCGA\\TCGA_官方网站\\Raw data\\gz_source_document\\miRNA\\TCGA-TGCT")
path.append("F:\\TCGA\\TCGA_官方网站\\Raw data\\gz_source_document\\miRNA\\TCGA-THCA")
path.append("F:\\TCGA\\TCGA_官方网站\\Raw data\\gz_source_document\\miRNA\\TCGA-THYM")
path.append("F:\\TCGA\\TCGA_官方网站\\Raw data\\gz_source_document\\miRNA\\TCGA-UCEC")
path.append("F:\\TCGA\\TCGA_官方网站\\Raw data\\gz_source_document\\miRNA\\TCGA-UCS")
path.append("F:\\TCGA\\TCGA_官方网站\\Raw data\\gz_source_document\\miRNA\\TCGA-UVM")
target_path = 'F:\\Pycharm_Project\\FGCNSurv\\Datasets_and_Preprocessing\\data\\miRNA_source_to_csv\\'

for type_path in path:
    label = type_path.split('\\')[6]
    print(label)
    metadata = open(type_path + "\\metadata.cart.2022-05-26.json", encoding='utf-8')
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
    for root, dirs, files in os.walk(type_path + "\\harmonized\\Transcriptome_Profiling\\miRNA_Expression_Quantification"):
        for file in files:
            file_path = str(os.path.join(root, file).encode('utf-8'), 'utf-8')
            col_list_value = []
            if file_path[-3:] == 'txt':
                # print(file_path)
                file_id = file_path.split('\\')[9]
                # print(file_id)
                file_name = file_path.split('\\')[10]
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
    result.to_csv(target_path + label + ".csv")
