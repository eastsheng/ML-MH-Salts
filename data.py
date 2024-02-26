import numpy as np
import pandas as pd

def read_data(f):
	df = pd.read_excel(f,header=1)
	df = df[['Brine', '%wt of salt',
			'CaBr2', 'CaCl2', 'K2CO3', 'KBr', 'KCl', 'KHCOO', 'MgBr2', 'MgCl2',
			'Na2SO4', 'NaBr', 'NaCl', 'NaHCOO', 'NaI', 'NH4Cl', 'ZnBr2', 'Na⁺',
			'K⁺', 'NH₄⁺', 'Zn⁺', 'Ca⁺⁺', 'Mg⁺⁺', 'Cl⁻', 'Br⁻', 'I⁻', 'HCOO⁻',
			'CO₃⁻⁻', 'SO₄⁻⁻', 'P (MPa)', 'T (K)']]
	return df

def removing_duplicates(df):
	subset_columns = df.columns[:30]
	duplicate_mask = df.duplicated(subset=subset_columns, keep='first')
	df = df[~duplicate_mask]
	return df



if __name__ == '__main__':
	f = "./data/1-s2.0-S1364032122009844-mmc1.xlsx"
	df = read_data(f)
	df = removing_duplicates(df)
	print(df)