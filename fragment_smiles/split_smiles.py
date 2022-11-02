'''
!/usr/bin/env python
-*- coding:utf-8 -*-
@ProjectName  :Code
@FileName  :Split_SMILES.py
@Time      :2021/7/1 16:12
@Author    :sylershao
'''
import pickle

import pandas as pd
import numpy as np
import time
# import treePlotter
from rdkit import Chem


def split_smiles(smiles):
    smiles_level = 0  # 读取层级
    smiles_level_max = 0
    index_level_stat_dict = {}  # {index:[w:这是字母,level,(-1,0,1，由上上一个字母到本字母到区别，默认为0)]
    for i, w in enumerate(smiles):
        state = 0
        if i == 0: state = 1

        if w == '(':
            smiles_level += 1
            if smiles_level_max < smiles_level: smiles_level_max = smiles_level
            state = 1
        if w == ')':
            smiles_level -= 1
            state = -1

        index_level_stat_dict[i] = [w, smiles_level, state]

    # print(index_level_stat_dict)

    # 输出level
    split_result = []

    split_str = ''

    for i in range(len(smiles)):
        # print('index_level_stat_dict[{}]'.format(i),index_level_stat_dict[i])

        if index_level_stat_dict[i][2] == 0:
            split_result[index_level_stat_dict[i][1]][-1] = split_result[index_level_stat_dict[i][1]][-1] + \
                                                            index_level_stat_dict[i][0]
        elif index_level_stat_dict[i][2] == 1:
            if len(split_result) == index_level_stat_dict[i][1]:
                split_result.append([index_level_stat_dict[i][0]])
            else:
                split_result[index_level_stat_dict[i][1]].append(index_level_stat_dict[i][0])
        elif index_level_stat_dict[i][2] == -1:
            split_result[index_level_stat_dict[i][1]][-1] = split_result[index_level_stat_dict[i][1]][-1] + 'R' + \
                                                            index_level_stat_dict[i][0]
    # print(split_result)

    for i, _ in enumerate(split_result):
        for j, ww in enumerate(split_result[i]):
            if '(' in ww: ww = ww.replace('(', '')
            if ')' in ww: ww = ww.replace(')', '')
            split_result[i][j] = ww

    # print(split_result)

    # 返回用于作图的dict
    split_result_dict = {}
    dict_i = {}
    for i in range(len(split_result)):
        _i = len(split_result) - 1 - i
        # print('_i',_i)

        R_count = 0

        dict_i_temp = {}
        for j, a in enumerate(split_result[_i]):
            # dict_i[j] = a
            dict_i_temp[j] = a
            if 'R' in a:
                count_w = 0
                for w in a:
                    if w == 'R': count_w += 1

                dict_i_temp[j] = {}
                dict_a = {}
                dict_i_temp[j][a] = {}
                for c in range(R_count, R_count + count_w, 1):
                    dict_i_temp[j][a][c - R_count] = dict_i[c]
                R_count += count_w

        dict_i = dict_i_temp
        #
        # print(dict_i)
        # print(dict_i_temp)
    return dict_i_temp, split_result

    # for j,b in enumerate(split_result[i]):


def plot_tree_test():
    a = {'no_surfacing':
             {'L1': {'flippers':
                         {'R0': 'no', 'R1': 'yes'}},
              'L0': {'flippers':
                         {'R0': 'why', 'R1': 'no'}},
              'L2': 'yes'}
         }
    b = {'Cc1cccRcc1': {
        0: {'-c2nc3ccccc3cRn2C/C=C/c2cccRcc2\r\nasdasd': {0: '=O', 1: {'CRCN3CRCOc4ccccc43': {0: '=O', 1: '=O'}}}}}}
    # treePlotter.createPlot(b, 'test3')


# def Statistics_for_splitlist(result_list): #统计出现次数


def main():
    times = time.time()
    data = pd.read_csv('../dataset/BindingDB/train.csv', index_col=0)
    SMILESs = list(data['SMILES'])
    # Label = list(data['Label'])
    # # 建立二维list存放分割结果
    # split_result_to_pkl = [SMILESs, GI50_Data, []]
    # Statistics_result_dict = {}
    # split_result_to_csv = pd.DataFrame()
    # split_result_to_csv['SMILES'] = SMILESs
    # split_result_to_csv['Label'] = Label
    # for i in range(500):
    #     split_result_to_csv['split_{}'.format(i)] = [np.nan]*len(split_result_to_csv)

    # split_result_to_csv.columns = split_result_to_csv.columns

    df_smiles_list = pd.DataFrame(columns=['index', 'sub_smiles'])

    for smiles_index, smiles in enumerate(SMILESs):
        print('smiles', smiles_index, smiles)
        # print('timecost',(time.time()-times))

        if smiles == smiles:  # 知识点 numpy特性  np.nan != np.nan  (判断是否为  np.nan)
            # SMILES canonicalization | SMILES to Canonical SMILES

            mol = Chem.MolFromSmiles(smiles)
            try:
                canonical_smi = Chem.MolToSmiles(mol)
            except:
                continue
            if not smiles == canonical_smi:  # 展示 canonicalization 效果
                print('canonical_smi', canonical_smi)
            # 置换符号
            if 'Cl' in canonical_smi: canonical_smi = canonical_smi.replace('Cl', 'L')
            if 'Br' in canonical_smi: canonical_smi = canonical_smi.replace('Br', 'P')

            # 分割smiles
            result_dict, result_list = split_smiles(canonical_smi)
            print(result_list)

            # 分割的基础上进行bpe算法
            smiles_list = []
            for layer in result_list:
                smiles_list.extend(layer)
            # print(smiles_list)
            df_smiles = pd.DataFrame()
            df_smiles['index'] = [smiles_index] * len(smiles_list)
            df_smiles['sub_smiles'] = smiles_list

            df_smiles_list = pd.concat([df_smiles_list, df_smiles])
            print(df_smiles_list)

        # if smiles_index == 1000:break

    df_smiles_list.to_csv('train.sub_smiles.csv')
    df_smiles_list['sub_smiles'].to_csv('train.sub_smiles.noindex.csv', index=None, header=None)


if __name__ == "__main__":
    main()
