from numpy.core.numeric import NaN
import pandas as pd
import numpy as np
from multiprocessing import freeze_support
from rdkit import Chem
from mordred import Calculator, descriptors

if __name__ == "__main__":
    """
    https://github.com/mordred-descriptor/mordred/blob/develop/examples/030-multiple_mol-multiple_desc.py
    """
    freeze_support()

    data = pd.read_csv("./datasets/dataset.csv")
    mols = [Chem.MolFromSmiles(smi) for smi in data["SMILES"]]

    # Create Calculator
    '''
    ignore_3D=False : (, 1826)
    ignore_3D=True  : (, 1613)
    '''
    calc = Calculator(descriptors, ignore_3D=True)

    # pandas method calculate multiple molecules (return pandas DataFrame)
    print("calc pandas")
    '''
    quiet=Falseにするとtqdmによるプログレスバーが現れるが同時に処理速度が圧倒的に遅くなった
    '''
    mordred_df = calc.pandas(mols, quiet=True)

    mordred_df = mordred_df.astype(str)
    masks = mordred_df.apply(lambda d: d.str.contains('[a-zA-Z]' ,na=False))
    mordred_df[masks] = None
    mordred_df = pd.concat([data, mordred_df], axis=1)
    
    mordred_df.to_csv("./datasets/mordred_train_dataset.csv", index=False)

    rdkit_data = pd.read_csv("./datasets/rdkit_dataset.csv")
    same_cols = []
    for rd_col in rdkit_data.columns:
        for mo_col in mordred_df.columns:
            if str(rd_col)==str(mo_col):
                same_cols.append(rd_col)
                break

    print(same_cols)



