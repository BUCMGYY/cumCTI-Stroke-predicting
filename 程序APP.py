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
    "regional category": "Regional_category",
    "cooking fuel use": "Cooking_fuel_use",
    "residence": "Residence",
    "education status": "Education_status",
    "stomach digestive disease": "Stomach",
}

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
    # ===== 预测 =====
    predicted_proba = pipe.predict_proba(X_user)[0]
    pred_label = pipe.predict(X_user)[0]

    # 优先取“正类=1”的概率；如果模型标签不是0/1，则取预测类概率
    classes = rf.classes_
    if 1 in classes:
        pos_idx = int(np.where(classes == 1)[0][0])
    else:
        pos_idx = int(np.where(classes == pred_label)[0][0])

    probability = float(predicted_proba[pos_idx] * 100.0)

    # ===== 文本结果（避免 Times New Roman 告警，改用默认可用字体）=====
    text = f"Based on feature values, predicted possibility of Stroke is {probability:.2f}%"
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(
        0.5, 0.5, text,
        fontsize=16, ha='center', va='center',
        # fontname='Times New Roman',  # 不再强制，不然容器里会报警
        transform=ax.transAxes
    )
    ax.axis('off')
    plt.savefig("prediction_text.png", bbox_inches='tight', dpi=300)
    st.image("prediction_text.png")

    # ===== SHAP（兼容二分类/多分类返回格式；使用 raw 输出，避免版本冲突）=====
    Xt = preprocess.transform(X_user)
    from scipy import sparse as _sp

    if _sp.issparse(Xt):
        Xt = Xt.toarray()

    # 展开后特征名（one-hot 后）
    try:
        feature_names = preprocess.get_feature_names_out()
    except Exception:
        feature_names = np.array([f"f{i}" for i in range(Xt.shape[1])])

    # 不要传 model_output / feature_perturbation，防止报错
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(Xt)
    expected = explainer.expected_value

    # --- 兼容不同返回格式 ---
    if isinstance(shap_values, list):
        # 多分类：按你上面用于展示概率的类索引 pos_idx 取
        sv_row = shap_values[pos_idx][0]
        base = expected[pos_idx] if isinstance(expected, (list, np.ndarray)) else expected
    else:
        # 二分类某些版本会直接返回 (n_samples, n_features)
        sv_row = shap_values[0]
        base = expected if not isinstance(expected, (list, np.ndarray)) else expected[0]

    shap.force_plot(
        base,
        sv_row,
        pd.Series(Xt[0], index=feature_names),
        matplotlib=True,
        show=False
    )
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=600)
    st.image("shap_force_plot.png")


