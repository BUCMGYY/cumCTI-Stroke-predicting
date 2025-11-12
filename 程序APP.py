import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from scipy import sparse
import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

# ===================== 你的范围与顺序（直接用你刚输出的） =====================
feature_ranges = {
    'cumCTI': {'type': 'numerical', 'min': 19.6991388, 'max': 37.51311244, 'default': 26.18794029},
    'CTI control levels': {'type': 'categorical', 'options': [0, 1, 2, 3], 'default': 2},
    'depression category': {'type': 'categorical', 'options': [0, 1], 'default': 1},
    'social isolation': {'type': 'categorical', 'options': [0, 1], 'default': 0},
    'marital status': {'type': 'categorical', 'options': [0, 1, 2], 'default': 2},
    'sleep duration': {'type': 'categorical', 'options': [0, 1, 2], 'default': 1},
    'regional category': {'type': 'categorical', 'options': [0, 1, 2], 'default': 1},
    'cooking fuel use': {'type': 'categorical', 'options': [0, 1], 'default': 1},
    'residence': {'type': 'categorical', 'options': [0, 1], 'default': 0},
    'education status': {'type': 'categorical', 'options': [0, 1], 'default': 0},
    'stomach digestive disease': {'type': 'categorical', 'options': [0, 1], 'default': 0}
}

DISPLAY_ORDER = [
    "cumCTI",
    "CTI control levels",
    "depression category",
    "social isolation",
    "marital status",
    "sleep duration",
    "regional category",
    "cooking fuel use",
    "residence",
    "education status",
    "stomach digestive disease",
]

# 显示名 -> 训练时内部列名（与你训练的列保持一致）
DISPLAY_TO_INTERNAL = {
    "cumCTI": "cumCTI",
    "CTI control levels": "CTI_control_levels",
    "depression category": "Depression_category",
    "social isolation": "social_isolation",
    "marital status": "Marital_status",
    "sleep duration": "Sleeping_time",
    "regional category": "Regional_Category" if "Regional_Category" in [] else "Regional_Category".replace("_Category","_category"),  # 防呆：若你列名首字母大小写不同，请按需改回去
    "cooking fuel use": "Cooking_fuel_use",
    "residence": "Residence",
    "education status": "Education_status",
    "stomach digestive disease": "Stomach",
}
# 如果上面 Regional 的名字和你训练时完全一致是 "Regional_category"，请把上一行替换回：
DISPLAY_TO_INTERNAL["regional category"] = "Regional_category"

# 训练时预处理器接收的内部列顺序（保持与你贴出来的 internal_order 一致）
internal_order = [
    'CTI_control_levels',
    'Cooking_fuel_use',
    'Depression_category',
    'Education_status',
    'Marital_status',
    'Regional_category',
    'Residence',
    'Sleeping_time',
    'Stomach',
    'cumCTI',
    'social_isolation'
]
# =========================================================================

st.title("cumCTI-Stroke RF Predictor with SHAP")

# 加载整条 Pipeline（preprocess + rf）
pipe = joblib.load('rf_model.pkl')
preprocess = pipe.named_steps['preprocess']
rf = pipe.named_steps['rf']

# 动态生成输入项
st.header("Enter the following feature values:")
feature_values_display = []
for feature in DISPLAY_ORDER:
    props = feature_ranges[feature]
    if props["type"] == "numerical":
        value = st.number_input(
            label=f"{feature} ({props['min']} - {props['max']})",
            min_value=float(props["min"]),
            max_value=float(props["max"]),
            value=float(props["default"]),
        )
    else:
        value = st.selectbox(
            label=f"{feature} (Select a value)",
            options=props.get("options", []),
            index=(props.get("options", []).index(props.get("default")) if props.get("options") and props.get("default") in props["options"] else 0)
        )
    feature_values_display.append(value)

# 映射为模型需要的内部列，并对齐列顺序
row_internal = {DISPLAY_TO_INTERNAL[d]: v for d, v in zip(DISPLAY_ORDER, feature_values_display)}
X_user = pd.DataFrame([row_internal], columns=internal_order)

# 预测与 SHAP（严格按你示例的写法组织变量名）
if st.button("Predict"):
    # 预处理 → 得到模型真实输入
    Xt = preprocess.transform(X_user)
    if sparse.issparse(Xt):
        Xt = Xt.toarray()

    # 与你示例一致的变量名
    features = Xt
    model = rf

    # —— 你的示例：预测 + 概率 ——
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    probability = predicted_proba[int(predicted_class)] * 100

    # —— 你的示例：文本图（保留 Times New Roman；若无该字体最多告警，不影响运行）——
    text = f"Based on feature values, predicted possibility of Stroke is {probability:.2f}%"
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(
        0.5, 0.5, text,
        fontsize=16, ha='center', va='center',
        fontname='Times New Roman',
        transform=ax.transAxes
    )
    ax.axis('off')
    plt.savefig("prediction_text.png", bbox_inches='tight', dpi=300)
    st.image("prediction_text.png")

    # —— 你的示例：SHAP ——
    explainer = shap.TreeExplainer(model)

    raw_shap = explainer.shap_values(features)
    expected = explainer.expected_value

    # 统一成 (n_samples, n_features, n_classes) 形状，确保下面示例调用不爆维度错
    if isinstance(raw_shap, list):
        shap_values = np.stack(raw_shap, axis=2)                  # list[类] -> 3D
        expected_vec = np.array(expected if isinstance(expected, (list, np.ndarray)) else [expected]*shap_values.shape[2])
    else:
        shap_values = raw_shap[:, :, None]                        # 二分类 2D -> 3D
        expected_vec = np.array(expected if isinstance(expected, (list, np.ndarray)) else [expected])

    # one-hot 后的特征名
    try:
        feature_names = preprocess.get_feature_names_out()
    except Exception:
        feature_names = np.array([f"f{i}" for i in range(features.shape[1])])

    # 生成 SHAP 力图（保持你的函数签名）
    class_index = int(predicted_class)
    # 第三个参数必须与模型特征维度一致——用模型的 one-hot 特征，而不是原始输入列
    shap.force_plot(
        expected_vec[min(class_index, shap_values.shape[2]-1)],
        shap_values[0, :, min(class_index, shap_values.shape[2]-1)],
        pd.Series(features[0], index=feature_names),
        matplotlib=True
    )
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")
