import pandas as pd
from sklearn.model_selection import KFold
import os
import sys

sys.path.append("../..")
import numpy as np
import torch
import random
import math
# import argparse
# import json
from model.neural_network import fmmo_survival_Analysis
# import torch.utils.data as Data
from sklearn import preprocessing
from FL_Opera import federated_averging, sync_models


def CIndex(pred, ytime_test, ystatus_test, num_category, time_interval):
    concord = 0.
    total = 0.
    N_test = ystatus_test.shape[0]
    ytime_test = np.squeeze(ytime_test)
    ystatus_test = np.squeeze(ystatus_test)
    theta = np.squeeze(pred)
    for i in range(N_test):
        if ystatus_test[i] == 1:
            if ytime_test[i] < time_interval * 20:
                hitting_time = int(ytime_test[i] / time_interval)
            if time_interval * 20 < ytime_test[i]:
                hitting_time = int((ytime_test[i] - time_interval * 20) / 183 + 20)
            if hitting_time >= num_category - 1:
                hitting_time = num_category - 1
            for j in range(N_test):
                if ytime_test[j] > ytime_test[i]:
                    total = total + 1
                    if np.sum(theta[j, :hitting_time + 1]) < np.sum(theta[i, :hitting_time + 1]):
                        concord = concord + 1
                    elif np.sum(theta[j, :hitting_time + 1]) == np.sum(theta[i, :hitting_time + 1]):
                        concord = concord + 0.5

    return concord / total


if __name__ == '__main__':
    # import time
    # time.sleep(3600*3.5)

    train_rate = 0.8
    train_lr = 2e-4
    num_category = 30
    time_interval = 90
    delta = 0.1
    lamda = 0.1
    target_lamda = 0.1

    base_path = "./Data_preprocessing"
    TRAIN_PATH_Site_1 = base_path + "/Meta_Train_Data/Site_1"
    dir_res = os.listdir(TRAIN_PATH_Site_1)
    Flag = True
    for i in range(len(dir_res)):
        if Flag:
            temp_train_path = TRAIN_PATH_Site_1 + '/' + dir_res[i]
            RNASeq_arr_1 = np.loadtxt(fname=temp_train_path + '/RNASeq.csv', delimiter=",", skiprows=1)
            miRNA_arr_1 = np.loadtxt(fname=temp_train_path + '/miRNA.csv', delimiter=",", skiprows=1)
            ytime_arr_1 = np.loadtxt(fname=temp_train_path + '/ytime.csv', delimiter=",", skiprows=1)
            ystatus_arr_1 = np.loadtxt(fname=temp_train_path + '/ystatus.csv', delimiter=",", skiprows=1)
            Flag = False
        else:
            temp_train_path = TRAIN_PATH_Site_1 + '/' + dir_res[i]
            temp_RNASeq_arr = np.loadtxt(fname=temp_train_path + '/RNASeq.csv', delimiter=",", skiprows=1)
            RNASeq_arr_1 = np.concatenate((RNASeq_arr_1, temp_RNASeq_arr), axis=0)
            temp_miRNA_arr = np.loadtxt(fname=temp_train_path + '/miRNA.csv', delimiter=",", skiprows=1)
            miRNA_arr_1 = np.concatenate((miRNA_arr_1, temp_miRNA_arr), axis=0)
            temp_ytime_arr = np.loadtxt(fname=temp_train_path + '/ytime.csv', delimiter=",", skiprows=1)
            ytime_arr_1 = np.concatenate((ytime_arr_1, temp_ytime_arr), axis=0)
            temp_ystatus_arr = np.loadtxt(fname=temp_train_path + '/ystatus.csv', delimiter=",", skiprows=1)
            ystatus_arr_1 = np.concatenate((ystatus_arr_1, temp_ystatus_arr), axis=0)

    standard_scaler = preprocessing.StandardScaler()
    RNASeq_arr_1 = standard_scaler.fit_transform(RNASeq_arr_1)
    miRNA_arr_1 = standard_scaler.fit_transform(miRNA_arr_1)

    ystatus_train_dead_index_1 = np.argwhere(ystatus_arr_1 == 1)
    ystatus_train_alive_index_1 = np.argwhere(ystatus_arr_1 == 0)

    ystatus_train_dead_1 = ystatus_arr_1[ystatus_train_dead_index_1,]
    ystatus_train_alive_1 = ystatus_arr_1[ystatus_train_alive_index_1,]
    RNASeq_train_dead_1 = RNASeq_arr_1[ystatus_train_dead_index_1,]
    RNASeq_train_alive_1 = RNASeq_arr_1[ystatus_train_alive_index_1,]
    miRNA_train_dead_1 = miRNA_arr_1[ystatus_train_dead_index_1,]
    miRNA_train_alive_1 = miRNA_arr_1[ystatus_train_alive_index_1,]
    ytime_train_dead_1 = ytime_arr_1[ystatus_train_dead_index_1,]
    ytime_train_alive_1 = ytime_arr_1[ystatus_train_alive_index_1,]

    TRAIN_PATH_Site_2 = base_path + "/Meta_Train_Data/Site_2"

    dir_res = os.listdir(TRAIN_PATH_Site_2)
    Flag = True
    for i in range(len(dir_res)):
        if Flag:
            temp_train_path = TRAIN_PATH_Site_2 + '/' + dir_res[i]
            RNASeq_arr_2 = np.loadtxt(fname=temp_train_path + '/RNASeq.csv', delimiter=",", skiprows=1)
            miRNA_arr_2 = np.loadtxt(fname=temp_train_path + '/miRNA.csv', delimiter=",", skiprows=1)
            ytime_arr_2 = np.loadtxt(fname=temp_train_path + '/ytime.csv', delimiter=",", skiprows=1)
            ystatus_arr_2 = np.loadtxt(fname=temp_train_path + '/ystatus.csv', delimiter=",", skiprows=1)
            Flag = False
        else:
            temp_train_path = TRAIN_PATH_Site_2 + '/' + dir_res[i]
            temp_RNASeq_arr = np.loadtxt(fname=temp_train_path + '/RNASeq.csv', delimiter=",", skiprows=1)
            RNASeq_arr_2 = np.concatenate((RNASeq_arr_2, temp_RNASeq_arr), axis=0)
            temp_miRNA_arr = np.loadtxt(fname=temp_train_path + '/miRNA.csv', delimiter=",", skiprows=1)
            miRNA_arr_2 = np.concatenate((miRNA_arr_2, temp_miRNA_arr), axis=0)
            temp_ytime_arr = np.loadtxt(fname=temp_train_path + '/ytime.csv', delimiter=",", skiprows=1)
            ytime_arr_2 = np.concatenate((ytime_arr_2, temp_ytime_arr), axis=0)
            temp_ystatus_arr = np.loadtxt(fname=temp_train_path + '/ystatus.csv', delimiter=",", skiprows=1)
            ystatus_arr_2 = np.concatenate((ystatus_arr_2, temp_ystatus_arr), axis=0)

    standard_scaler = preprocessing.StandardScaler()
    RNASeq_arr_2 = standard_scaler.fit_transform(RNASeq_arr_2)
    miRNA_arr_2 = standard_scaler.fit_transform(miRNA_arr_2)

    ystatus_train_dead_index_2 = np.argwhere(ystatus_arr_2 == 1)
    ystatus_train_alive_index_2 = np.argwhere(ystatus_arr_2 == 0)

    ystatus_train_dead_2 = ystatus_arr_2[ystatus_train_dead_index_2,]
    ystatus_train_alive_2 = ystatus_arr_2[ystatus_train_alive_index_2,]
    RNASeq_train_dead_2 = RNASeq_arr_2[ystatus_train_dead_index_2,]
    RNASeq_train_alive_2 = RNASeq_arr_2[ystatus_train_alive_index_2,]
    miRNA_train_dead_2 = miRNA_arr_2[ystatus_train_dead_index_2,]
    miRNA_train_alive_2 = miRNA_arr_2[ystatus_train_alive_index_2,]
    ytime_train_dead_2 = ytime_arr_2[ystatus_train_dead_index_2,]
    ytime_train_alive_2 = ytime_arr_2[ystatus_train_alive_index_2,]

    TRAIN_PATH_Site_3 = base_path + "/Meta_Train_Data/Site_3"

    dir_res = os.listdir(TRAIN_PATH_Site_3)
    Flag = True
    for i in range(len(dir_res)):
        if Flag:
            temp_train_path = TRAIN_PATH_Site_3 + '/' + dir_res[i]
            RNASeq_arr_3 = np.loadtxt(fname=temp_train_path + '/RNASeq.csv', delimiter=",", skiprows=1)
            miRNA_arr_3 = np.loadtxt(fname=temp_train_path + '/miRNA.csv', delimiter=",", skiprows=1)
            ytime_arr_3 = np.loadtxt(fname=temp_train_path + '/ytime.csv', delimiter=",", skiprows=1)
            ystatus_arr_3 = np.loadtxt(fname=temp_train_path + '/ystatus.csv', delimiter=",", skiprows=1)
            Flag = False
        else:
            temp_train_path = TRAIN_PATH_Site_3 + '/' + dir_res[i]
            temp_RNASeq_arr = np.loadtxt(fname=temp_train_path + '/RNASeq.csv', delimiter=",", skiprows=1)
            RNASeq_arr_3 = np.concatenate((RNASeq_arr_3, temp_RNASeq_arr), axis=0)
            temp_miRNA_arr = np.loadtxt(fname=temp_train_path + '/miRNA.csv', delimiter=",", skiprows=1)
            miRNA_arr_3 = np.concatenate((miRNA_arr_3, temp_miRNA_arr), axis=0)
            temp_ytime_arr = np.loadtxt(fname=temp_train_path + '/ytime.csv', delimiter=",", skiprows=1)
            ytime_arr_3 = np.concatenate((ytime_arr_3, temp_ytime_arr), axis=0)
            temp_ystatus_arr = np.loadtxt(fname=temp_train_path + '/ystatus.csv', delimiter=",", skiprows=1)
            ystatus_arr_3 = np.concatenate((ystatus_arr_3, temp_ystatus_arr), axis=0)

    standard_scaler = preprocessing.StandardScaler()
    RNASeq_arr_3 = standard_scaler.fit_transform(RNASeq_arr_3)
    miRNA_arr_3 = standard_scaler.fit_transform(miRNA_arr_3)

    ystatus_train_dead_index_3 = np.argwhere(ystatus_arr_3 == 1)
    ystatus_train_alive_index_3 = np.argwhere(ystatus_arr_3 == 0)

    ystatus_train_dead_3 = ystatus_arr_3[ystatus_train_dead_index_3,]
    ystatus_train_alive_3 = ystatus_arr_3[ystatus_train_alive_index_3,]
    RNASeq_train_dead_3 = RNASeq_arr_3[ystatus_train_dead_index_3,]
    RNASeq_train_alive_3 = RNASeq_arr_3[ystatus_train_alive_index_3,]
    miRNA_train_dead_3 = miRNA_arr_3[ystatus_train_dead_index_3,]
    miRNA_train_alive_3 = miRNA_arr_3[ystatus_train_alive_index_3,]
    ytime_train_dead_3 = ytime_arr_3[ystatus_train_dead_index_3,]
    ytime_train_alive_3 = ytime_arr_3[ystatus_train_alive_index_3,]

    # TRAIN_PATH_Site_4 = base_path + "/Meta_Train_Data/Site_4"
    # dir_res = os.listdir(TRAIN_PATH_Site_4)
    # Flag = True
    # for i in range(len(dir_res)):
    #     if Flag:
    #         temp_train_path = TRAIN_PATH_Site_4 + '/' + dir_res[i]
    #         RNASeq_arr_4 = np.loadtxt(fname=temp_train_path + '/RNASeq.csv', delimiter=",", skiprows=1)
    #         miRNA_arr_4 = np.loadtxt(fname=temp_train_path + '/miRNA.csv', delimiter=",", skiprows=1)
    #         ytime_arr_4 = np.loadtxt(fname=temp_train_path + '/ytime.csv', delimiter=",", skiprows=1)
    #         ystatus_arr_4 = np.loadtxt(fname=temp_train_path + '/ystatus.csv', delimiter=",", skiprows=1)
    #         Flag = False
    #     else:
    #         temp_train_path = TRAIN_PATH_Site_3 + '/' + dir_res[i]
    #         temp_RNASeq_arr = np.loadtxt(fname=temp_train_path + '/RNASeq.csv', delimiter=",", skiprows=1)
    #         RNASeq_arr_4 = np.concatenate((RNASeq_arr_4, temp_RNASeq_arr), axis=0)
    #         temp_miRNA_arr = np.loadtxt(fname=temp_train_path + '/miRNA.csv', delimiter=",", skiprows=1)
    #         miRNA_arr_4 = np.concatenate((miRNA_arr_4, temp_miRNA_arr), axis=0)
    #         temp_ytime_arr = np.loadtxt(fname=temp_train_path + '/ytime.csv', delimiter=",", skiprows=1)
    #         ytime_arr_4 = np.concatenate((ytime_arr_4, temp_ytime_arr), axis=0)
    #         temp_ystatus_arr = np.loadtxt(fname=temp_train_path + '/ystatus.csv', delimiter=",", skiprows=1)
    #         ystatus_arr_4 = np.concatenate((ystatus_arr_4, temp_ystatus_arr), axis=0)
    #
    # standard_scaler = preprocessing.StandardScaler()
    # RNASeq_arr_4 = standard_scaler.fit_transform(RNASeq_arr_4)
    # miRNA_arr_4 = standard_scaler.fit_transform(miRNA_arr_4)

    Target_PATH = base_path + "/Meta_Target_Data/BLCA"
    RNASeq_feature = np.loadtxt(fname=Target_PATH + "/RNASeq.csv", delimiter=",", skiprows=1)
    miRNA_feature = np.loadtxt(fname=Target_PATH + "/miRNA.csv", delimiter=",", skiprows=1)
    ytime = np.loadtxt(fname=Target_PATH + "/ytime.csv", delimiter=",", skiprows=1)
    ystatus = np.loadtxt(fname=Target_PATH + "/ystatus.csv", delimiter=",", skiprows=1)

    standard_scaler = preprocessing.StandardScaler()
    RNASeq_feature = standard_scaler.fit_transform(RNASeq_feature)
    miRNA_feature = standard_scaler.fit_transform(miRNA_feature)

    max_cind_list = []
    for k_num in range(20):
        random.seed(k_num)
        torch.manual_seed(k_num)
        np.random.seed(k_num)
        noise_level = 0.05
        ystatus_dead_index = np.squeeze(np.argwhere(ystatus == 1))
        ystatus_censor_index = np.squeeze(np.argwhere(ystatus == 0))

        ystatus_dead = ystatus[ystatus_dead_index,]
        ystatus_censor = ystatus[ystatus_censor_index,]
        RNASeq_dead = RNASeq_feature[ystatus_dead_index,]
        RNASeq_censor = RNASeq_feature[ystatus_censor_index,]
        miRNA_dead = miRNA_feature[ystatus_dead_index,]
        miRNA_censor = miRNA_feature[ystatus_censor_index,]
        ytime_dead = ytime[ystatus_dead_index,]
        ytime_censor = ytime[ystatus_censor_index,]

        dead_range = range(RNASeq_dead.shape[0])
        ind_dead_train = random.sample(dead_range, math.floor(RNASeq_dead.shape[0] * train_rate))
        RNASeq_dead_train = RNASeq_dead[ind_dead_train,]
        print(RNASeq_dead_train.shape)
        miRNA_dead_train = miRNA_dead[ind_dead_train,]
        ytime_dead_train = ytime_dead[ind_dead_train,]
        ystatus_dead_train = ystatus_dead[ind_dead_train,]

        ind_dead_rest = [i for i in dead_range if i not in ind_dead_train]
        RNASeq_dead_rest = RNASeq_dead[ind_dead_rest,]
        miRNA_dead_rest = miRNA_dead[ind_dead_rest,]
        ytime_dead_rest = ytime_dead[ind_dead_rest,]
        ystatus_dead_rest = ystatus_dead[ind_dead_rest,]

        censor_range = range(RNASeq_censor.shape[0])
        ind_censor_train = random.sample(censor_range, math.floor(RNASeq_censor.shape[0] * train_rate))
        print(ind_censor_train)
        RNASeq_censor_train = RNASeq_censor[ind_censor_train,]
        miRNA_censor_train = miRNA_censor[ind_censor_train,]
        ytime_censor_train = ytime_censor[ind_censor_train,]
        ystatus_censor_train = np.squeeze(ystatus_censor[ind_censor_train,])

        RNASeq_train = np.concatenate((RNASeq_dead_train, RNASeq_censor_train), axis=0)
        miRNA_train = np.concatenate((miRNA_dead_train, miRNA_censor_train), axis=0)
        ystatus_train = np.squeeze(np.concatenate((ystatus_dead_train, ystatus_censor_train), axis=0))
        ytime_train = np.squeeze(np.concatenate((ytime_dead_train, ytime_censor_train), axis=0))

        ind_censor_rest = [i for i in censor_range if i not in ind_censor_train]
        RNASeq_censor_rest = RNASeq_censor[ind_censor_rest,]
        miRNA_censor_rest = miRNA_censor[ind_censor_rest,]
        ytime_censor_rest = np.squeeze(ytime_censor[ind_censor_rest,])
        ystatus_censor_rest = np.squeeze(ystatus_censor[ind_censor_rest,])

        RNASeq_val = np.concatenate((RNASeq_dead_rest, RNASeq_censor_rest), axis=0)
        miRNA_val = np.concatenate((miRNA_dead_rest, miRNA_censor_rest), axis=0)
        ytime_val = np.squeeze(np.concatenate((ytime_dead_rest, ytime_censor_rest), axis=0))
        ystatus_val = np.squeeze(np.concatenate((ystatus_dead_rest, ystatus_censor_rest), axis=0))

        flmo_model = fmmo_survival_Analysis()
        site1_model = fmmo_survival_Analysis()
        site1_model.load_state_dict(flmo_model.state_dict())  # copy? looks okay
        optimizer_site1 = torch.optim.Adam([{'params': site1_model.parameters()}, ], lr=train_lr, weight_decay=5e-4)

        site2_model = fmmo_survival_Analysis()
        site2_model.load_state_dict(flmo_model.state_dict())  # copy? looks okay
        optimizer_site2 = torch.optim.Adam([{'params': site2_model.parameters()}, ], lr=train_lr, weight_decay=5e-4)

        site3_model = fmmo_survival_Analysis()
        site3_model.load_state_dict(flmo_model.state_dict())  # copy? looks okay
        optimizer_site3 = torch.optim.Adam([{'params': site3_model.parameters()}, ], lr=train_lr, weight_decay=5e-4)

        site4_model = fmmo_survival_Analysis()
        site4_model.load_state_dict(flmo_model.state_dict())  # copy? looks okay
        optimizer_site4 = torch.optim.Adam([{'params': site4_model.parameters()}, ], lr=train_lr, weight_decay=5e-4)

        server_model = fmmo_survival_Analysis()
        max_cind = 0.0
        best_iter = 0
        Aux_TRAIN_NUM = 300
        TRAIN_NUM = np.min(np.array([300, math.floor(RNASeq_feature.shape[0] * train_rate)]))
        dead_sample_rate = 0.45
        alive_sample_rate = 0.55
        for iter in range(100):
            print(iter)
            site1_model.train()
            site2_model.train()
            site3_model.train()
            site4_model.train()
            lst = []
            # index = np.squeeze(np.arange(0, RNASeq_arr_1.shape[0]))
            # fir_index = np.random.choice(index, size=200, replace=False)
            # RNASeq_site1_train_tensor = torch.tensor(RNASeq_arr_1[fir_index,], dtype=torch.float)
            # miRNA_site1_train_tensor = torch.tensor(miRNA_arr_1[fir_index,], dtype=torch.float)
            # ystatus_site1_train_tensor = torch.tensor(ystatus_arr_1[fir_index,], dtype=torch.float)
            # ytime_site1_train_tensor = torch.tensor(ytime_arr_1[fir_index,], dtype=torch.float)

            ind_train_dead_1 = np.random.choice(range(RNASeq_train_dead_1.shape[0]), int(Aux_TRAIN_NUM * dead_sample_rate))
            ind_train_alive_1 = np.random.choice(range(RNASeq_train_alive_1.shape[0]), int(Aux_TRAIN_NUM * alive_sample_rate))
            # print(ind)

            RNASeq_batch_train_dead_1 = RNASeq_train_dead_1[ind_train_dead_1,]
            miRNA_batch_train_dead_1 = miRNA_train_dead_1[ind_train_dead_1,]
            ystatus_batch_train_dead_1 = ystatus_train_dead_1[ind_train_dead_1,]
            ytime_batch_train_dead_1 = ytime_train_dead_1[ind_train_dead_1,]

            RNASeq_batch_train_alive_1 = RNASeq_train_alive_1[ind_train_alive_1,]
            miRNA_batch_train_alive_1 = miRNA_train_alive_1[ind_train_alive_1,]
            ystatus_batch_train_alive_1 = ystatus_train_alive_1[ind_train_alive_1,]
            ytime_batch_train_alive_1 = ytime_train_alive_1[ind_train_alive_1,]

            RNASeq_batch_train_1 = np.concatenate((RNASeq_batch_train_dead_1, RNASeq_batch_train_alive_1), axis=0)
            miRNA_batch_train_1 = np.concatenate((miRNA_batch_train_dead_1, miRNA_batch_train_alive_1), axis=0)
            ystatus_batch_train_1 = np.concatenate((ystatus_batch_train_dead_1, ystatus_batch_train_alive_1), axis=0)
            ytime_batch_train_1 = np.concatenate((ytime_batch_train_dead_1, ytime_batch_train_alive_1), axis=0)

            RNASeq_site1_train_tensor = torch.tensor(RNASeq_batch_train_1, dtype=torch.float)
            miRNA_site1_train_tensor = torch.tensor(miRNA_batch_train_1, dtype=torch.float)
            ystatus_site1_train_tensor = torch.tensor(ystatus_batch_train_1, dtype=torch.float)
            ytime_site1_train_tensor = torch.tensor(ytime_batch_train_1, dtype=torch.float)

            row_num = RNASeq_site1_train_tensor.shape[0]
            theta = site1_model.get_survival_result(RNASeq_site1_train_tensor, miRNA_site1_train_tensor)
            mask_hit_batch_train = torch.tensor(np.zeros([row_num, num_category]),
                                                dtype=torch.float)
            for i in range(row_num):
                if ytime_site1_train_tensor[i] < time_interval * 20:
                    hitting_time = int(ytime_site1_train_tensor[i] / time_interval)
                if time_interval * 20 < ytime_site1_train_tensor[i]:
                    hitting_time = int((ytime_site1_train_tensor[i]-time_interval * 20)/183 + 20)
                if hitting_time >= num_category - 1:
                    hitting_time = num_category - 1
                if ystatus_site1_train_tensor[i] != 0:  # not censored
                    mask_hit_batch_train[i, hitting_time] = 1
                else:  # censored
                    mask_hit_batch_train[i, hitting_time:] = 1

            loss_1 = - torch.mean(torch.log(torch.sum(mask_hit_batch_train * theta, dim=1) + 1e-8))

            R_matrix_batch_train = torch.tensor(
                np.zeros([row_num, row_num], dtype=int), dtype=torch.float)
            for i in range(row_num):
                R_matrix_batch_train[i,] = torch.tensor(
                    np.array(list(map(int, (ytime_site1_train_tensor >= ytime_site1_train_tensor[i])))))

            mask_rank_batch_train = torch.tensor(np.zeros([row_num, num_category]), dtype=torch.float)
            for i in range(row_num):
                if ytime_site1_train_tensor[i] < time_interval * 20:
                    hitting_time = int(ytime_site1_train_tensor[i] / time_interval)
                if time_interval * 20 < ytime_site1_train_tensor[i]:
                    hitting_time = int((ytime_site1_train_tensor[i]-time_interval * 20)/183 + 20)
                if hitting_time >= num_category - 1:
                    hitting_time = num_category - 1
                mask_rank_batch_train[i, :hitting_time + 1] = 1

            CIF = torch.sum(theta * mask_rank_batch_train, dim=1)
            Rank_matrix_batch_train = torch.tensor(torch.zeros([row_num, row_num], dtype=torch.float),
                                                   dtype=torch.float)
            for i in range(row_num):
                Rank_matrix_batch_train[i, :] = torch.exp(-(CIF[i] -
                                                            torch.matmul(theta,
                                                                         mask_rank_batch_train[i, :])) / torch.sum(
                    mask_rank_batch_train[i, :]) / delta)

            loss_2 = torch.mean(torch.mul(torch.sum(R_matrix_batch_train * Rank_matrix_batch_train, dim=1),
                                          torch.reshape(ystatus_site1_train_tensor, [row_num])))

            loss = loss_1 + lamda * loss_2
            # Update meta-parameters
            optimizer_site1.zero_grad()
            loss.backward()
            optimizer_site1.step()
            # end = time.time()
            # print("1 iteration time:", end - start)
            lst.append(site1_model)

            # index = np.squeeze(np.arange(0, RNASeq_arr_2.shape[0]))
            # fir_index = np.random.choice(index, size=200, replace=False)
            # RNASeq_site2_train_tensor = torch.tensor(RNASeq_arr_2[fir_index,], dtype=torch.float)
            # miRNA_site2_train_tensor = torch.tensor(miRNA_arr_2[fir_index,], dtype=torch.float)
            # ystatus_site2_train_tensor = torch.tensor(ystatus_arr_2[fir_index,], dtype=torch.float)
            # ytime_site2_train_tensor = torch.tensor(ytime_arr_2[fir_index,], dtype=torch.float)

            ind_train_dead_2 = np.random.choice(range(RNASeq_train_dead_2.shape[0]), int(Aux_TRAIN_NUM * dead_sample_rate))
            ind_train_alive_2 = np.random.choice(range(RNASeq_train_alive_2.shape[0]), int(Aux_TRAIN_NUM * alive_sample_rate))
            # print(ind)

            RNASeq_batch_train_dead_2 = RNASeq_train_dead_2[ind_train_dead_2,]
            miRNA_batch_train_dead_2 = miRNA_train_dead_2[ind_train_dead_2,]
            ystatus_batch_train_dead_2 = ystatus_train_dead_2[ind_train_dead_2,]
            ytime_batch_train_dead_2 = ytime_train_dead_2[ind_train_dead_2,]

            RNASeq_batch_train_alive_2 = RNASeq_train_alive_2[ind_train_alive_2,]
            miRNA_batch_train_alive_2 = miRNA_train_alive_2[ind_train_alive_2,]
            ystatus_batch_train_alive_2 = ystatus_train_alive_2[ind_train_alive_2,]
            ytime_batch_train_alive_2 = ytime_train_alive_2[ind_train_alive_2,]

            RNASeq_batch_train_2 = np.concatenate((RNASeq_batch_train_dead_2, RNASeq_batch_train_alive_2), axis=0)
            miRNA_batch_train_2 = np.concatenate((miRNA_batch_train_dead_2, miRNA_batch_train_alive_2), axis=0)
            ystatus_batch_train_2 = np.concatenate((ystatus_batch_train_dead_2, ystatus_batch_train_alive_2), axis=0)
            ytime_batch_train_2 = np.concatenate((ytime_batch_train_dead_2, ytime_batch_train_alive_2), axis=0)

            RNASeq_site2_train_tensor = torch.tensor(RNASeq_batch_train_2, dtype=torch.float)
            miRNA_site2_train_tensor = torch.tensor(miRNA_batch_train_2, dtype=torch.float)
            ystatus_site2_train_tensor = torch.tensor(ystatus_batch_train_2, dtype=torch.float)
            ytime_site2_train_tensor = torch.tensor(ytime_batch_train_2, dtype=torch.float)

            row_num = RNASeq_site2_train_tensor.shape[0]
            theta = site2_model.get_survival_result(RNASeq_site2_train_tensor, miRNA_site2_train_tensor)
            mask_hit_batch_train = torch.tensor(np.zeros([row_num, num_category]),
                                                dtype=torch.float)
            for i in range(row_num):
                if ytime_site2_train_tensor[i] < time_interval * 20:
                    hitting_time = int(ytime_site2_train_tensor[i] / time_interval)
                if time_interval * 20 < ytime_site2_train_tensor[i]:
                    hitting_time = int((ytime_site2_train_tensor[i]-time_interval * 20)/183 + 20)
                if hitting_time >= num_category - 1:
                    hitting_time = num_category - 1
                if ystatus_site2_train_tensor[i] != 0:  # not censored
                    mask_hit_batch_train[i, hitting_time] = 1
                else:  # censored
                    mask_hit_batch_train[i, hitting_time:] = 1

            loss_1 = - torch.mean(torch.log(torch.sum(mask_hit_batch_train * theta, dim=1) + 1e-8))

            R_matrix_batch_train = torch.tensor(
                np.zeros([row_num, row_num], dtype=int), dtype=torch.float)
            for i in range(row_num):
                R_matrix_batch_train[i,] = torch.tensor(
                    np.array(list(map(int, (ytime_site2_train_tensor >= ytime_site2_train_tensor[i])))))

            mask_rank_batch_train = torch.tensor(np.zeros([row_num, num_category]), dtype=torch.float)
            for i in range(row_num):
                if ytime_site2_train_tensor[i] < time_interval * 20:
                    hitting_time = int(ytime_site2_train_tensor[i] / time_interval)
                if time_interval * 20 < ytime_site2_train_tensor[i]:
                    hitting_time = int((ytime_site2_train_tensor[i]-time_interval * 20)/183 + 20)
                if hitting_time >= num_category - 1:
                    hitting_time = num_category - 1
                mask_rank_batch_train[i, :hitting_time + 1] = 1

            CIF = torch.sum(theta * mask_rank_batch_train, dim=1)
            Rank_matrix_batch_train = torch.tensor(torch.zeros([row_num, row_num], dtype=torch.float),
                                                   dtype=torch.float)
            for i in range(row_num):
                Rank_matrix_batch_train[i, :] = torch.exp(-(CIF[i] -
                                                            torch.matmul(theta,
                                                                         mask_rank_batch_train[i, :])) / torch.sum(
                    mask_rank_batch_train[i, :]) / delta)

            loss_2 = torch.mean(torch.mul(torch.sum(R_matrix_batch_train * Rank_matrix_batch_train, dim=1),
                                          torch.reshape(ystatus_site2_train_tensor, [row_num])))

            loss = loss_1 + lamda * loss_2
            # Update meta-parameters
            optimizer_site2.zero_grad()
            loss.backward()
            optimizer_site2.step()
            # end = time.time()
            # print("1 iteration time:", end - start)
            lst.append(site2_model)

            # index = np.squeeze(np.arange(0, RNASeq_arr_3.shape[0]))
            # fir_index = np.random.choice(index, size=200, replace=False)
            # RNASeq_site3_train_tensor = torch.tensor(RNASeq_arr_3[fir_index,], dtype=torch.float)
            # miRNA_site3_train_tensor = torch.tensor(miRNA_arr_3[fir_index,], dtype=torch.float)
            # ystatus_site3_train_tensor = torch.tensor(ystatus_arr_3[fir_index,], dtype=torch.float)
            # ytime_site3_train_tensor = torch.tensor(ytime_arr_3[fir_index,], dtype=torch.float)

            ind_train_dead_3 = np.random.choice(range(RNASeq_train_dead_3.shape[0]), int(Aux_TRAIN_NUM * dead_sample_rate))
            ind_train_alive_3 = np.random.choice(range(RNASeq_train_alive_3.shape[0]), int(Aux_TRAIN_NUM * alive_sample_rate))
            # print(ind)

            RNASeq_batch_train_dead_3 = RNASeq_train_dead_3[ind_train_dead_3,]
            miRNA_batch_train_dead_3 = miRNA_train_dead_3[ind_train_dead_3,]
            ystatus_batch_train_dead_3 = ystatus_train_dead_3[ind_train_dead_3,]
            ytime_batch_train_dead_3 = ytime_train_dead_3[ind_train_dead_3,]

            RNASeq_batch_train_alive_3 = RNASeq_train_alive_3[ind_train_alive_3,]
            miRNA_batch_train_alive_3 = miRNA_train_alive_3[ind_train_alive_3,]
            ystatus_batch_train_alive_3 = ystatus_train_alive_3[ind_train_alive_3,]
            ytime_batch_train_alive_3 = ytime_train_alive_3[ind_train_alive_3,]

            RNASeq_batch_train_3 = np.concatenate((RNASeq_batch_train_dead_3, RNASeq_batch_train_alive_3), axis=0)
            miRNA_batch_train_3 = np.concatenate((miRNA_batch_train_dead_3, miRNA_batch_train_alive_3), axis=0)
            ystatus_batch_train_3 = np.concatenate((ystatus_batch_train_dead_3, ystatus_batch_train_alive_3), axis=0)
            ytime_batch_train_3 = np.concatenate((ytime_batch_train_dead_3, ytime_batch_train_alive_3), axis=0)

            RNASeq_site3_train_tensor = torch.tensor(RNASeq_batch_train_3, dtype=torch.float)
            miRNA_site3_train_tensor = torch.tensor(miRNA_batch_train_3, dtype=torch.float)
            ystatus_site3_train_tensor = torch.tensor(ystatus_batch_train_3, dtype=torch.float)
            ytime_site3_train_tensor = torch.tensor(ytime_batch_train_3, dtype=torch.float)

            row_num = RNASeq_site3_train_tensor.shape[0]
            theta = site3_model.get_survival_result(RNASeq_site3_train_tensor, miRNA_site3_train_tensor)
            mask_hit_batch_train = torch.tensor(np.zeros([row_num, num_category]),
                                                dtype=torch.float)
            for i in range(row_num):
                if ytime_site3_train_tensor[i] < time_interval * 20:
                    hitting_time = int(ytime_site3_train_tensor[i] / time_interval)
                if time_interval * 20 < ytime_site3_train_tensor[i]:
                    hitting_time = int((ytime_site3_train_tensor[i]-time_interval * 20)/183 + 20)
                if hitting_time >= num_category - 1:
                    hitting_time = num_category - 1
                if ystatus_site3_train_tensor[i] != 0:  # not censored
                    mask_hit_batch_train[i, hitting_time] = 1
                else:  # censored
                    mask_hit_batch_train[i, hitting_time:] = 1

            loss_1 = - torch.mean(torch.log(torch.sum(mask_hit_batch_train * theta, dim=1) + 1e-8))

            R_matrix_batch_train = torch.tensor(
                np.zeros([row_num, row_num], dtype=int), dtype=torch.float)
            for i in range(row_num):
                R_matrix_batch_train[i,] = torch.tensor(
                    np.array(list(map(int, (ytime_site3_train_tensor >= ytime_site3_train_tensor[i])))))

            mask_rank_batch_train = torch.tensor(np.zeros([row_num, num_category]), dtype=torch.float)
            for i in range(row_num):
                if ytime_site3_train_tensor[i] < time_interval * 20:
                    hitting_time = int(ytime_site3_train_tensor[i] / time_interval)
                if time_interval * 20 < ytime_site3_train_tensor[i]:
                    hitting_time = int((ytime_site3_train_tensor[i]-time_interval * 20)/183 + 20)
                if hitting_time >= num_category - 1:
                    hitting_time = num_category - 1
                mask_rank_batch_train[i, :hitting_time + 1] = 1

            CIF = torch.sum(theta * mask_rank_batch_train, dim=1)
            Rank_matrix_batch_train = torch.tensor(torch.zeros([row_num, row_num], dtype=torch.float),
                                                   dtype=torch.float)
            for i in range(row_num):
                Rank_matrix_batch_train[i, :] = torch.exp(-(CIF[i] -
                                                            torch.matmul(theta,
                                                                         mask_rank_batch_train[i, :])) / torch.sum(
                    mask_rank_batch_train[i, :]) / delta)

            loss_2 = torch.mean(torch.mul(torch.sum(R_matrix_batch_train * Rank_matrix_batch_train, dim=1),
                                          torch.reshape(ystatus_site3_train_tensor, [row_num])))

            loss = loss_1 + lamda * loss_2
            # Update meta-parameters
            optimizer_site3.zero_grad()
            loss.backward()
            optimizer_site3.step()
            # end = time.time()
            # print("1 iteration time:", end - start)
            lst.append(site3_model)

            # index = np.squeeze(np.arange(0, RNASeq_train.shape[0]))
            # fir_index = np.random.choice(index, size=200, replace=False)
            # RNASeq_train_tensor = torch.tensor(RNASeq_train[fir_index,], dtype=torch.float)
            # miRNA_train_tensor = torch.tensor(miRNA_train[fir_index,], dtype=torch.float)
            # ystatus_train_tensor = torch.tensor(ystatus_train[fir_index,], dtype=torch.float)
            # ytime_train_tensor = torch.tensor(ytime_train[fir_index,], dtype=torch.float)

            ind_train_dead = np.random.choice(range(RNASeq_dead_train.shape[0]), int(TRAIN_NUM * dead_sample_rate))
            ind_train_alive = np.random.choice(range(RNASeq_censor_train.shape[0]), int(TRAIN_NUM * alive_sample_rate))
            # print(ind)

            RNASeq_batch_train_dead = RNASeq_dead_train[ind_train_dead,]
            miRNA_batch_train_dead = miRNA_dead_train[ind_train_dead,]
            ystatus_batch_train_dead = ystatus_dead_train[ind_train_dead,]
            ytime_batch_train_dead = ytime_dead_train[ind_train_dead,]

            RNASeq_batch_train_alive = RNASeq_censor_train[ind_train_alive,]
            miRNA_batch_train_alive = miRNA_censor_train[ind_train_alive,]
            ystatus_batch_train_alive = ystatus_censor_train[ind_train_alive,]
            ytime_batch_train_alive = ytime_censor_train[ind_train_alive,]

            RNASeq_batch_train = np.concatenate((RNASeq_batch_train_dead, RNASeq_batch_train_alive), axis=0)
            miRNA_batch_train = np.concatenate((miRNA_batch_train_dead, miRNA_batch_train_alive), axis=0)
            ystatus_batch_train = np.concatenate((ystatus_batch_train_dead, ystatus_batch_train_alive), axis=0)
            ytime_batch_train = np.concatenate((ytime_batch_train_dead, ytime_batch_train_alive), axis=0)

            RNASeq_train_tensor = torch.tensor(RNASeq_batch_train, dtype=torch.float)
            miRNA_train_tensor = torch.tensor(miRNA_batch_train, dtype=torch.float)
            ystatus_train_tensor = torch.tensor(ystatus_batch_train, dtype=torch.float)
            ytime_train_tensor = torch.tensor(ytime_batch_train, dtype=torch.float)
            row_num = RNASeq_train_tensor.shape[0]
            theta = site4_model.get_survival_result(RNASeq_train_tensor, miRNA_train_tensor)
            mask_hit_batch_train = torch.tensor(np.zeros([row_num, num_category]),
                                                dtype=torch.float)
            for i in range(row_num):
                if ytime_train_tensor[i] < time_interval*20:
                    hitting_time = int(ytime_train_tensor[i] / time_interval)
                if time_interval*20 < ytime_train_tensor[i]:
                    hitting_time = int((ytime_train_tensor[i]-time_interval*20)/183 + 20)
                if hitting_time >= num_category - 1:
                    hitting_time = num_category - 1
                if ystatus_train_tensor[i] != 0:  # not censored
                    mask_hit_batch_train[i, hitting_time] = 1
                else:  # censored
                    mask_hit_batch_train[i, hitting_time:] = 1

            loss_1 = - torch.mean(torch.log(torch.sum(mask_hit_batch_train * theta, dim=1) + 1e-8))

            R_matrix_batch_train = torch.tensor(
                np.zeros([row_num, row_num], dtype=int), dtype=torch.float)
            for i in range(row_num):
                R_matrix_batch_train[i,] = torch.tensor(
                    np.array(list(map(int, (ytime_train_tensor >= ytime_train_tensor[i])))))

            mask_rank_batch_train = torch.tensor(np.zeros([row_num, num_category]), dtype=torch.float)
            for i in range(row_num):
                if ytime_train_tensor[i] < time_interval*20:
                    hitting_time = int(ytime_train_tensor[i] / time_interval)
                if time_interval*20 < ytime_train_tensor[i]:
                    hitting_time = int((ytime_train_tensor[i]-time_interval*20)/183 + 20)
                if hitting_time >= num_category - 1:
                    hitting_time = num_category - 1
                mask_rank_batch_train[i, :hitting_time + 1] = 1

            CIF = torch.sum(theta * mask_rank_batch_train, dim=1)
            Rank_matrix_batch_train = torch.tensor(torch.zeros([row_num, row_num], dtype=torch.float),
                                                   dtype=torch.float)
            for i in range(row_num):
                Rank_matrix_batch_train[i, :] = torch.exp(-(CIF[i] -
                                                            torch.matmul(theta,
                                                                         mask_rank_batch_train[i, :])) / torch.sum(
                    mask_rank_batch_train[i, :]) / delta)

            loss_2 = torch.mean(torch.mul(torch.sum(R_matrix_batch_train * Rank_matrix_batch_train, dim=1),
                                          torch.reshape(ystatus_train_tensor, [row_num])))

            loss = loss_1 + target_lamda * loss_2
            # Update meta-parameters
            optimizer_site4.zero_grad()
            loss.backward()
            optimizer_site4.step()
            # end = time.time()
            # print("1 iteration time:", end - start)
            lst.append(site4_model)
            server_model, _ = federated_averging(server_model, lst, noise_level)
            sync_models(server_model, lst)

            server_model.eval()

            eval_model = fmmo_survival_Analysis(p=0)
            eval_model.load_state_dict(server_model.state_dict())  # copy? looks okay

            eval_model.eval()
            RNASeq_train_tensor = torch.tensor(RNASeq_train, dtype=torch.float)
            miRNA_train_tensor = torch.tensor(miRNA_train, dtype=torch.float)
            pred_train = eval_model.get_survival_result(RNASeq_train_tensor, miRNA_train_tensor)

            cind_train = CIndex(pred_train.detach().numpy(), ytime_train, ystatus_train, num_category, time_interval)
            # print("训练集C_index:" + str(cind_train))
            RNASeq_val_tensor = torch.tensor(RNASeq_val, dtype=torch.float)
            miRNA_val_tensor = torch.tensor(miRNA_val, dtype=torch.float)
            pred_val = eval_model.get_survival_result(RNASeq_val_tensor, miRNA_val_tensor)
            cind_val = CIndex(pred_val.detach().numpy(), ytime_val, ystatus_val, num_category, time_interval)
            with open('Cox_Result_lr5e-05/Cox_Val/Regularizer_eval_randomseed' + str(k_num) + '.log',
                      'a') as f:
                f.writelines(
                    'Iteration:' + str(iter) + ",cind:" + str(cind_val) + '\n')
            if cind_train - cind_val > 0.05:
                if cind_val >= max_cind:
                    max_cind = cind_val
                    best_iter = iter
                    count = 0
                else:
                    count = count + 1

        max_cind_list.append(max_cind)
        with open('Cox_Result_lr5e-05/Cox_Val/Regularizer_eval_AVE_Cind' + '.log', 'a') as f:
            f.writelines(str(best_iter) + ',' + str(max_cind) + '\n')
    with open('Cox_Result_lr5e-05/Cox_Val/Regularizer_eval_AVE_Cind' + '.log', 'a') as f:
        f.writelines("max cind ave:" + str(np.mean(np.array(max_cind_list))) + '\n')
        f.writelines("max cind std:" + str(np.std(np.array(max_cind_list))) + '\n')
