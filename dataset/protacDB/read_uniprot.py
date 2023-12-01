import pandas as pd
import pysam as sam
import numpy as np

data = pd.read_csv("protac.csv", sep=",")
print(data.keys())

fa = sam.Fastafile("uniprot-compressed.fasta")
print(fa.filename)

protein_dict = dict()
for item in fa.references:
    # print(str(item))
    demo = fa.fetch(item)
    # print(demo)
    name = item.split("|")

    protein_dict[name[1]] = demo

print(protein_dict)

uniprot_list = list(data['Uniprot'])
smiles_list = list(data['Smiles'])
kd_list = np.array(list(data['Kd (nM, Protac to Target)']))

print(uniprot_list.__len__())
print(smiles_list.__len__())
sequence_list = []
print(uniprot_list)
for item2 in np.array(uniprot_list):
    # print(item2)
    if item2 != "nan":
        sequence_list.append(protein_dict[item2])
    else:
        sequence_list.append("unk")
# print(sequence_list)

res_seq = []
res_com = []
res_label = []
for i in range(len(uniprot_list)):
    if sequence_list[i] != 'unk':
        if kd_list[i] != 'nan':
            if kd_list[i] != ' ':
                if "/" not in kd_list[i]:
                    res_seq.append(sequence_list[i])
                    res_com.append(smiles_list[i])
                    res_label.append(kd_list[i])

print(res_com.__len__())
print(res_seq.__len__())
print(res_label)

res = []
for item in uniprot_list:
    if item not in res:
        res.append(item)

protacDB_res = []

for j in range(len(res_label)):
    protacDB_part = []
    protacDB_part.append(res_com[j])
    protacDB_part.append(res_seq[j])
    print(res_label[j])
    if float(res_label[j]) < 10:
        protacDB_part.append(0)
    else:
        protacDB_part.append(1)
    protacDB_res.append(protacDB_part)

print(protacDB_res)
#

res_protac = pd.DataFrame(data=protacDB_res, columns=["SMILES", "Target Sequence", "Label"])
res_protac.insert(loc=0, column="Unnamed: 0", value=list(range(len(res_protac))))
# print(train)
res_protac.to_csv("../DUDE/res_protac.csv")

# print(res)
# print(res.__len__())

# uniprot_list_out = open("uniprot_list.txt", 'w')
# for i in range(len(res)):
#     uniprot_list_out.write(str(res[i]))
#     uniprot_list_out.write(' ')
