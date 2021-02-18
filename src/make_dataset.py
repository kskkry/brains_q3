from operator import index
import numpy as np
import pandas as pd
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors 
from rdkit.ML.Descriptors import MoleculeDescriptors
from multiprocessing import freeze_support
from rdkit import Chem
from mordred import Calculator, descriptors

class RdkitDataset:
    def __init__(self, data: pd.DataFrame) -> None:
        '''
        train
        data: columns [smiles, IC50 (nM)]

        test
        data: columns [smiles]
        '''
        self.data = data
        self.names = [x[0] for x in Descriptors._descList]

    def feature_vector(self, smiles):
        """
        SMILESから特徴量を作成し、listで返す
        """
        # cis-trans 情報は RDKitがうまく解釈できない場合があるので除去する
        smiles = smiles.replace('/', '').replace('\\', '')
        # SMILESから Molオブジェクト (RDKitで分子を表すオブジェクト) を作成する
        mol = AllChem.MolFromSmiles(smiles)

        rdkit_info = MoleculeDescriptors.MolecularDescriptorCalculator(self.names)
        rdkit_data = rdkit_info.CalcDescriptors(mol)
        return list(rdkit_data)

    def transform(self)-> pd.DataFrame:
        smiles_data = [self.feature_vector(smiles) for smiles in self.data["SMILES"]]
        rdkit_df = pd.DataFrame(smiles_data, columns=self.names).astype(np.float32)
        rdkit_df = pd.concat([self.data, rdkit_df], axis=1)
        return rdkit_df


class MordredDataset:
    def __init__(self, data) -> None:
        '''
        上手く機能しなかった
        下記のメソッドを使う方がよい
        '''
        self.data = data
    
    def transform(self, train_data=True)-> pd.DataFrame:
        """
        https://github.com/mordred-descriptor/mordred/blob/develop/examples/030-multiple_mol-multiple_desc.py
        """
        freeze_support()
        mols = [Chem.MolFromSmiles(smi) for smi in self.data["SMILES"]]

        # Create Calculator
        calc = Calculator(descriptors)

        # pandas method calculate multiple molecules (return pandas DataFrame)
        mordred_df = calc.pandas(mols)

        mordred_df = mordred_df.astype(str)
        masks = mordred_df.apply(lambda d: d.str.contains('[a-zA-Z]' ,na=False))
        mordred_df[masks] = None
        mordred_df = pd.concat([self.data, mordred_df], axis=1)

        return mordred_df

def to_mordred(df):
    """
    https://github.com/mordred-descriptor/mordred/blob/develop/examples/030-multiple_mol-multiple_desc.py
    """
    freeze_support()
    mols = [Chem.MolFromSmiles(smi) for smi in df["SMILES"]]
    calc = Calculator(descriptors)

    # pandas method calculate multiple molecules (return pandas DataFrame)
    """
    https://github.com/mordred-descriptor/mordred/blob/develop/mordred/_base/calculator.py
    Calculator.pandas parameter : 
    """

    mordred_df = calc.pandas(mols, quiet=True)

    mordred_df = mordred_df.astype(str).replace(" ", "")
    masks = mordred_df.apply(lambda d: d.str.contains('[a-zA-Z]' ,na=False))
    mordred_df[masks] = None

    all_data = pd.concat([df, mordred_df], axis=1)
    return all_data

