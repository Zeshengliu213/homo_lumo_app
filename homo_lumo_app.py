import streamlit as st
import pickle
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from PIL import Image
import os
import pickle

def load_model():
    # 获取当前脚本所在的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构造模型文件的相对路径
    model_path = os.path.join(script_dir, 'xgb_homo_lumo_model.pkl')
    
    # 加载模型
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model

model = load_model()

# 生成KP指纹
def generate_kp_fingerprints(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    kp_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=4860)
    return list(kp_fp)

# 生成分子图像
#def generate_molecule_image(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        img = Draw.MolToImage(mol, size=(300, 300))  # 生成分子结构图
        return img
    else:
        return None

# 预测HOMO和LUMO
def predict_homo_lumo(smiles, model):
    fp = generate_kp_fingerprints(smiles)
    if fp is None:
        return "Invalid SMILES string.", None
    fp_df = pd.DataFrame([fp])  # 将指纹转换为DataFrame格式
    predictions = model.predict(fp_df)
    homo_pred = round(predictions[0][0], 2)  # HOMO预测值，保留两位小数
    lumo_pred = round(predictions[0][1], 2)  # LUMO预测值，保留两位小数
    return homo_pred, lumo_pred

# Streamlit应用界面
st.title('HOMO-LUMO Predictor with Molecule Image')

# 用户输入SMILES码
smiles_input = st.text_input('Enter a SMILES code:')

# 当用户输入SMILES码时，进行预测并显示分子图像
if st.button('Predict'):
    if smiles_input:
        try:
            homo, lumo = predict_homo_lumo(smiles_input, model)
            #molecule_image = generate_molecule_image(smiles_input)
            
            if lumo is not None:
                # 输出预测值
                st.write(f'Predicted HOMO: {homo:.2f}')
                st.write(f'Predicted LUMO: {lumo:.2f}')
                
                # 输出分子结构图像
                #if molecule_image:
                    #st.image(molecule_image, caption='Molecular Structure')
                #else:
                    #st.write("Invalid SMILES string. Could not generate molecule image.")
            else:
                st.write(homo)  # 如果 SMILES 无效，显示错误信息
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.error('Please enter a valid SMILES code.')
