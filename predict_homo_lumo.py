import pickle
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

# 加载模型
def load_model():
    with open(r'd:\Working relevant\OneDrive - Linköpings universitet\Python study\Machine learning\Github\Machine learning-Predict Eg\xgb_homo_lumo_model.pkl', 'rb') as f:
        model = pickle.load(f)
        

    return model

# 生成KP指纹
def generate_kp_fingerprints(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    kp_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=4860)
    return list(kp_fp)

# 定义预测函数
def predict_homo_lumo(smiles, model):
    fp = generate_kp_fingerprints(smiles)
    
    if fp is None:
        return "Invalid SMILES string."
    
    fp_df = pd.DataFrame([fp])  # 将指纹转换为DataFrame格式
    
    # 使用模型预测
    predictions = model.predict(fp_df)
    
    # 输出预测结果
    homo_pred = predictions[0][0]  # HOMO预测值
    lumo_pred = predictions[0][1]  # LUMO预测值
    
    return homo_pred, lumo_pred

# 主程序
if __name__ == "__main__":
    # 加载模型
    model = load_model()
    
    # 提示用户输入SMILES码
    smiles_input = input("Enter a SMILES code: ")
    
    # 调用预测函数
    homo, lumo = predict_homo_lumo(smiles_input, model)
    
    # 输出预测结果
    print(f'Predicted HOMO: {homo}')
    print(f'Predicted LUMO: {lumo}')
