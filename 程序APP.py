import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from scipy import sparse

# ===================== 你的范围与顺序（直接用你刚输出的） =====================
feature_ranges = {
    'cumCTI': {'type': 'numerical', 'min': 19.6991388, 'max': 37.51311244, 'default': 26.18794029},
    'CTI control levels': {'type': 'categorical', 'min': 0.0, 'max': 3.0, 'default': 2.0},
    'depression category': {'type': 'categorical', 'min': 0.0, 'max': 1.0, 'default': 1.0},
    'social isolation': {'type': 'categorical', 'min': 0.0, 'max': 1.0, 'default': 0.0},
    'marital status': {'type': 'categorical', 'min': 0.0, 'max': 2.0, 'default': 2.0},
    'sleep duration': {'type': 'categorical', 'min': 0.0, 'max': 2.0, 'default': 0.5},
    'regional category': {'type': 'categorical', 'min': 0.0, 'max': 2.0, 'default': 1.0},
    'cooking fuel use': {'type': 'categorical', 'min': 0.0, 'max': 1.0, 'default': 1.0},
    'residence': {'type': 'categorical', 'min': 0.0, 'max': 1.0, 'default': 0.0},
    'education status': {'type': 'categorical', 'min': 0.0, 'max': 1.0, 'default': 0.0},
    'stomach digestive disease': {'type': 'categorical', 'min': 0.0, 'max': 1.0, 'default': 0.0}
}

# 显示名的固定顺序（确保 UI 顺序一致）
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

# 显示名 -> 训练时内部列名（与你的训练列一致）
DISPLAY_TO_INTERNAL = {
    "cumCTI": "cumCTI",
    "CTI control levels": "CTI_control_levels",
    "depression category": "Depression_category",
    "social isolation": "social_isolation",
    "marital status": "Marital_status",
    "sleep duration": "Sleeping_time",
    "regional category": "Regional_category",
    "cooking fuel use": "Cooking_fuel_use",
    "residence": "Residence",
    "education status": "Education_status",
    "stomach digestive disease": "Stomach",
}

# 训练时预处理器接收的内部列顺序（来自你刚才的 internal_order）
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

st.title("cumCTI-Stroke KNN Predictor with SHAP")

# ===== 改动1：加载整条 Pipeline（preprocess + rf）=====
pipe = joblib.load('rf_model.pkl')
preprocess = pipe.named_steps['preprocess']
rf = pipe.named_steps['rf']

# ===== 动态生成输入项（保留你原先的写法与风格）=====
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
        # 这里理论上不会走到，因为你给的都是 numerical；保留以防后续扩展
        value = st.selectbox(
            label=f"{feature} (Select a value)",
            options=props.get("options", []),
        )
    feature_values_display.append(value)

# ===== 映射为模型需要的内部列，并对齐列顺序 =====
row_internal = {DISPLAY_TO_INTERNAL[d]: v for d, v in zip(DISPLAY_ORDER, feature_values_display)}
X_user = pd.DataFrame([row_internal], columns=internal_order)

# ===== 预测与 SHAP 可视化 =====
if st.button("Predict"):
    # 模型预测（整条 Pipeline）
    predicted_proba = pipe.predict_proba(X_user)[0]
    pred_label = pipe.predict(X_user)[0]

    # 取“正类=1”的概率（若无标签1，则退而取预测类概率）
    classes = rf.classes_
    if 1 in classes:
        pos_idx = int(np.where(classes == 1)[0][0])
    else:
        pos_idx = int(np.where(classes == pred_label)[0][0])

    probability = predicted_proba[pos_idx] * 100.0

    # 显示预测结果（沿用你的 Matplotlib 文本渲染）
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

    # ===== SHAP：在预处理后的空间解释 RF =====
    Xt = preprocess.transform(X_user)
    if sparse.issparse(Xt):
        Xt = Xt.toarray()

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(Xt)  # list: 每个类别一个数组 [n_samples, n_features]

    # one-hot 展开后的特征名（若不可用则用 f0..fn）
    try:
        feature_names = preprocess.get_feature_names_out()
    except Exception:
        feature_names = np.array([f"f{i}" for i in range(Xt.shape[1])])

    # 选择用于展示的类别：与概率显示一致
    sv_row = shap_values[pos_idx][0]
    shap.force_plot(
        explainer.expected_value[pos_idx],
        sv_row,
        pd.Series(Xt[0], index=feature_names),
        matplotlib=True,
        show=False
    )
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=600)
    st.image("shap_force_plot.png")
