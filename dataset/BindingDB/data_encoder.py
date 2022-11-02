import pandas as pd
import collections
import numpy as np
import torch

val_df = pd.read_csv("val.sub_smiles.csv")
train_df = pd.read_csv("train_split.sub_smiles.csv")
test_df = pd.read_csv("test_split.sub_smiles.csv")
# print(val_df['sub_smiles'])

val_df_list = list(val_df['sub_smiles'])
val_df_index = list(val_df['index'])

train_df_list = list(train_df['sub_smiles'])
train_df_index = list(train_df['index'])

test_df_list = list(test_df['sub_smiles'])
test_df_index = list(test_df['index'])

sub_word_list = val_df_list+test_df_list+train_df_list
# print(sub_word_list)

word2dict = collections.Counter(sub_word_list)

demo = list(word2dict.keys())
print(demo)

dict_index_word = {}
for index, word in enumerate(demo):
    dict_index_word[word] = index

print(dict_index_word)
# dict_index_word = dict(demo3, demo2)
# print(word2dict)
# print(dict_index_word)


def encoder(data_list, data_index):
    res = []
    for j in range(data_index[-1]+1):
        word_list = []
        # print(data_index[-1])
        for i in range(len(data_index)):
            if data_index[i] == j:
                word_list.append(int(dict_index_word[data_list[i]]))
            # else:
            #     break
        res.append(word_list)
    # res = torch.nn.utils.rnn.pad_sequence(res, batch_first=True, padding_value=50)
    print(res.__len__())
    return res
#
#
train_res = encoder(train_df_list, train_df_index)

test_res = encoder(test_df_list, test_df_index)
#
val_res = encoder(val_df_list, val_df_index)
# print(val_res)
#
df_train_encoder = pd.DataFrame()
df_train_encoder['smiles_encoder'] = train_res
df_train_encoder.to_pickle('train_split_sub_smiles.pickle')

df_test_encoder = pd.DataFrame()
df_test_encoder['smiles_encoder'] = test_res
df_test_encoder.to_pickle('test_split_sub_smiles.pickle')

df_val_encoder = pd.DataFrame()
df_val_encoder['smiles_encoder'] = val_res
df_val_encoder.to_pickle('val_sub_smiles.pickle')




