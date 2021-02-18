import os
import pickle
from sklearn.preprocessing import PowerTransformer, quantile_transform, normalize
from rdkit import rdBase, Chem, DataStructs
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem, PandasTools, Descriptors, rdmolfiles
from rdkit.ML.Descriptors import MoleculeDescriptors
import collections
import pandas as pd
import numpy as np

def plus_count(smile):
    return smile.count("+")

def minus_count(smile):
    return smile.count("-")


def num_element(smile, element="c"):
    """
    原子の数を数えるだけ
    微妙に上がるとのこと
    """
    count = 0
    smile += "a"
    for index, s in enumerate(smile):
        if element == "c":
            #要塩素の場合
            if (s == "c" or s == "C") and smile[index+1] != "l":
                count += 1
        elif element == "o":
            if s == "o" or s == "O":
                count += 1
        elif element == "n":
            if s == "n" or s == "N":
                count += 1
        elif element == "s":
            if s == "S" or s == "s":
                count += 1
        elif element == "cl":
            if (s == "l" and smile[index-1] == "C") or (s == "l" and smile[index-1] == "c"):
                count += 1
        elif element == "br":
            if (s == "B" and smile[index+1] == "r") or (s == "b" and smile[index+1] == "r"):
                count += 1
        elif element == "f":
            if s == "F" or s == "f":
                count += 1
        elif element == "i":
            if s == "I" or s == "i":
                count += 1
        elif element == "na":
            if (s == "N" and smile[index+1] == "a") or (s == "n" and smile[index+1] == "a"):
                count += 1
        
    if element=="pt":
        count = str(smile).count("Pt")
        if count > 0:
            count = 1
    elif element=="sn":
        count = str(smile).count("Sn")
        if count > 0:
            count = 1
    elif element=="@":
        count = str(smile).count("@")
        if count > 0:
            count = 1

    return count 

def len_smile(smile):
    return len(str(smile).replace("#", "").replace("[", "").replace("]","").replace("{","").replace("}","").replace("(","").replace(")","").replace("+","").replace("-","").replace("-","").replace("=",""))

def judge_symmetry(smile):
    mol = Chem.MolFromSmiles(smile)
    rankAtoms = list(rdmolfiles.CanonicalRankAtoms(mol, breakTies=False))
    rankAtoms_counter = collections.Counter(rankAtoms)
    symmetry = len(rankAtoms_counter.values()) / len(rankAtoms)
    #print(len(rankAtoms_counter.values()) / len(rankAtoms))
    return symmetry


def calc_ARR(smile):
    '''
    各結合のうち芳香族性の結合の割合を算出
    '''
    mh = Chem.MolFromSmiles(smile)
    m = Chem.RemoveHs(mh)
    num_bonds = m.GetNumBonds()
    num_aromatic_bonds = 0
    for bond in m.GetBonds():
        if bond.GetIsAromatic():
            num_aromatic_bonds += 1
    if num_bonds==0:
        print(smile) # C, [He]
        ARR = 0
    else:
        ARR = num_aromatic_bonds/num_bonds
    return ARR

class TransformDataset:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def fill(self, fill_type="max"):
        #print("Null Count", self.data.isnull().sum().sum())
        self.data = self.data.replace([[np.inf, -np.inf], None])
        pt = PowerTransformer()
        for col in self.data.columns:
            if col=="SMILES" or col=="IC50 (nM)":
                continue
            if self.data[col].isnull().values.sum()==self.data[col].shape[0]:
                self.data[col] = 0
                continue
            if fill_type=="max":
                MM = 1E6
                self.data[col] = self.data[col].fillna(1E6).astype(np.float32)
                self.data[col] = np.nan_to_num(self.data[col], nan=1E6, posinf=1E6, neginf=-1E6) #重要
            elif fill_type=="zero":
                self.data[col] = self.data[col].fillna(1E6).astype(np.float32)
                self.data[col] = np.nan_to_num(self.data[col], nan=1E6, posinf=1E6, neginf=-1E6) #重要
            elif fill_type=="mean":
                self.data[col] = self.data[col].astype(np.float32)
                self.data[col] = self.data[col].fillna(self.data[col].mean()).astype(np.float32)
                self.data[col] = np.nan_to_num(self.data[col], nan=0, posinf=+2, neginf=-2) #重要
            elif fill_type=="median":
                self.data[col] = self.data[col].fillna(self.data[col].median()).astype(np.float32)
            self.data[col] = self.data[col].astype(np.float32)
        
        return self.data
    
    def scale(self, scale_type="normal"):
        for col in self.data.columns:
            if col=="SMILES" or col=="IC50 (nM)":
                continue
            if len(self.data[col].unique())==1:
                continue
            avg = self.data[col].mean()
            std = self.data[col].std()
            self.data[col] = (self.data[col] - avg) / std
        return self.data
    
    def transform(self):
        self.data["num_c"] = self.data["SMILES"].apply(num_element, element="c")
        self.data["num_o"] = self.data["SMILES"].apply(num_element, element="o")
        self.data["num_n"] = self.data["SMILES"].apply(num_element, element="n")
        self.data["num_s"] = self.data["SMILES"].apply(num_element, element="s")
        self.data["num_pt"] = self.data["SMILES"].apply(num_element, element="pt")
        self.data["num_sn"] = self.data["SMILES"].apply(num_element, element="sn")
        self.data["num_@"] = self.data["SMILES"].apply(num_element, element="@")
        self.data["num_f"] = self.data["SMILES"].apply(num_element, element="f")
        self.data["num_br"] = self.data["SMILES"].apply(num_element, element="br")
        self.data["num_cl"] = self.data["SMILES"].apply(num_element, element="cl")
        self.data["num_i"] = self.data["SMILES"].apply(num_element, element="i")

        self.data["Sum_ATSC0_p_Z"] = (self.data["ATSC0p"].astype(np.float32) + self.data["ATSC0Z"].astype(np.float32))
        self.data["Sum_ATSC0_p_m"] = (self.data["ATSC0p"].astype(np.float32) + self.data["ATSC0m"].astype(np.float32))
        self.data["Sum_ATSC0_m_Z"] = (self.data["ATSC0m"].astype(np.float32) + self.data["ATSC0Z"].astype(np.float32))

        self.data["Diff_ATSC0_p_Z"] = (self.data["ATSC0p"].astype(np.float32) - self.data["ATSC0Z"].astype(np.float32))
        self.data["Diff_ATSC0_p_m"] = (self.data["ATSC0p"].astype(np.float32) - self.data["ATSC0m"].astype(np.float32))
        self.data["Diff_ATSC0_m_Z"] = (self.data["ATSC0m"].astype(np.float32) - self.data["ATSC0Z"].astype(np.float32))

        self.data["Diff_SlogP_VSA1_2"] = (self.data["SlogP_VSA2"].astype(np.float32) - self.data["SlogP_VSA1"].astype(np.float32))
        return self.data


    def radius_similarity(self, r=2):
        '''
        訓練データ, または文献中の特徴的な構造をもつ化合物を類似度判定の対象に使用
        https://proceedings.neurips.cc/paper/2015/file/f9be311e65d81a9ad8150a60844bb94c-Paper.pdf
        '''
        path = './src/similarity/similarity_smiles.txt'

        with open(path, 'r') as f:
            similarity_list = f.readlines()
        similarity_mols = [Chem.MolFromSmiles(str(smi)) for smi in similarity_list]
        mols = [Chem.MolFromSmiles(smi) for smi in self.data["SMILES"]]

        '''
        AllChem.GetMorganFingerprintAsBitVectによる類似度判定
        '''
        
        similarity_morgan = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=r, nBits=2048) for mol in similarity_mols]
        morgan_fps = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=r, nBits=2048) for mol in mols]
        for index, sim_mor in enumerate(similarity_morgan):
            self.data["TanimotoSimilarity_radius{}_index{}".format(r, index+1)] = DataStructs.BulkTanimotoSimilarity(sim_mor, morgan_fps[0:])
            self.data["DiceSimilarity_radius{}_index{}".format(r, index+1)] = DataStructs.BulkDiceSimilarity(sim_mor, morgan_fps[0:])

        '''
        AllChem.GetMACCSKeysFingerprintによる類似度判定
        '''
        maccs_lists = [AllChem.GetMACCSKeysFingerprint(mol) for mol in similarity_mols]
        maccs_fps = [AllChem.GetMACCSKeysFingerprint(mol) for mol in mols]
        for index, sim_mor in enumerate(maccs_lists):
            self.data["maccs_TanimotoSimilarity_index{}".format(index+1)] = DataStructs.BulkTanimotoSimilarity(sim_mor, maccs_fps[0:])
            self.data["maccs_DiceSimilarity_index{}".format(index+1)] = DataStructs.BulkDiceSimilarity(sim_mor, maccs_fps[0:])

        morgan_lists = [AllChem.GetMorganFingerprint(mol, radius=r) for mol in similarity_mols]
        morgan_fps = [AllChem.GetMorganFingerprint(mol, radius=r) for mol in mols]
        for index, sim_mor in enumerate(morgan_lists):
            self.data["morgan_TanimotoSimilarity_radius{}_index{}".format(r, index+1)] = DataStructs.BulkTanimotoSimilarity(sim_mor, morgan_fps[0:])
            self.data["morgan_DiceSimilarity_radius{}_index{}".format(r, index+1)] = DataStructs.BulkDiceSimilarity(sim_mor, morgan_fps[0:])

        return self.data
            





