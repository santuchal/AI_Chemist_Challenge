import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
import gzip
from rdkit.Chem import MolFromSmiles, rdMolDescriptors
from rdkit.Chem.Descriptors import CalcMolDescriptors
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesRegressor
import streamlit as st

class Molecule:
	def __init__(self, smiles: str):
		if not smiles :
			print("Empty smiles are given")
			sys.exit()
		self.smiles = smiles
		self.mol = MolFromSmiles(smiles)
	def descriptor_generator(self):
		return CalcMolDescriptors(self.mol)

SMI = st.text_input('Input SMILE', 'O=Cc1ccc(Cl)cc1')
st.write('The input SMILE is', str(SMI))
mol = MolFromSmiles(SMI)

formula = rdMolDescriptors.CalcMolFormula(MolFromSmiles(SMI))
descriptors = CalcMolDescriptors(mol)
descriptors_dataframe = pd.DataFrame([list(descriptors.values())], columns= list(descriptors.keys()))
st.markdown(''':rainbow[ABSORPTION]''')
st.markdown(''':orange[LGBMRegressor]''')


with gzip.GzipFile('model/Absorption_Lipophilicity_Prediction_lgbm_model.joblib.gz', 'rb') as fa:  
   Absorption_Lipophilicity_Prediction_lgbm_model = joblib.load(fa)

st.write("Absorption Lipophilicity Result for LGBM Regressor : ", round(Absorption_Lipophilicity_Prediction_lgbm_model.predict(descriptors_dataframe)[0],4))


with gzip.GzipFile('model/Absorption_Lipophilicity_Prediction_etr_model.joblib.gz', 'rb') as fe:  
   Absorption_Lipophilicity_Prediction_etr_model = joblib.load(fe)

st.markdown(''':orange[ExtraTreesRegressor]''')
st.write("Absorption_Lipophilicity Result for ExtraTreesRegressor : ", round(Absorption_Lipophilicity_Prediction_etr_model.predict(descriptors_dataframe)[0],4))


with gzip.GzipFile('model/Absorption_Lipophilicity_Prediction_rf_model.joblib.gz', 'rb') as fi:  
   Absorption_Lipophilicity_Prediction_rf_model = joblib.load(fi)


st.markdown(''':orange[RandomForestRegressor]''')
st.write("Absorption_Lipophilicity Result for RandomForestRegressor : ", round(Absorption_Lipophilicity_Prediction_rf_model.predict(descriptors_dataframe)[0],4))


with gzip.GzipFile('model/Absorption_Lipophilicity_Prediction_rf_model_optimised.joblib.gz', 'rb') as fo:  
   Absorption_Lipophilicity_Prediction_rf_model_optimised = joblib.load(fo)

st.markdown(''':orange[RandomForestRegressor Optimised]''')
st.write("Absorption_Lipophilicity Result for Optimised RandomForestRegressor : ", round(Absorption_Lipophilicity_Prediction_rf_model_optimised.predict(descriptors_dataframe)[0],4))

with gzip.GzipFile('model/Absorption_Lipophilicity_Prediction_etr_model_optimised.joblib.gz', 'rb') as fu:  
   Absorption_Lipophilicity_Prediction_etr_model_optimised = joblib.load(fu)

st.markdown(''':orange[ExtraTreesRegressor Optimised]''')
st.write("Absorption_Lipophilicity Result for Optimised ExtraTreesRegressor : ", round(Absorption_Lipophilicity_Prediction_etr_model_optimised.predict(descriptors_dataframe)[0],4))

st.markdown(''':orange[LGBMRegressor Optimised]''')
with gzip.GzipFile('model/Absorption_Lipophilicity_Prediction_lgbm_model_optimised.joblib.gz', 'rb') as fb:  
   Absorption_Lipophilicity_Prediction_lgbm_model_optimised = joblib.load(fb)

st.write("Absorption Lipophilicity Result for Optimised LGBM Regressor : ", round(Absorption_Lipophilicity_Prediction_lgbm_model_optimised.predict(descriptors_dataframe)[0],4))
