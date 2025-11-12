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

st.title("cumCTI-Stroke KNN Predictor with SHAP")  # 文案随你，模型实际是 RF

# ===== 加载整条 Pipeline（preprocess + rf）=====
pipe = joblib.load('rf_model.pkl')
preprocess = pipe.named_steps['preprocess']
rf = pipe.named_steps['rf']

# ===== 动态生成输入项 =====
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
            index=(props["options"].index(props["default"]) if props.get("options") and props.get("default") in props["options"] else 0),
        )
    feature_values_display.append(value)

# ===== 映射为模型需要的内部列，并对齐列顺序 =====
row_internal = {DISPLAY_TO_INTERNAL[d]: v for d, v in zip(DISPLAY_ORDER, feature_values_display)}
X_user = pd.DataFrame([row_internal], columns=internal_order)

# ===== 预测与 SHAP 可视化（按你示例代码的变量名来） =====
if st.button("Predict"):
    # 1) 预测（示例变量名）
    predicted_proba = pipe.predict_proba(X_user)[0]
    pred_label = pipe.predict(X_user)[0]
    predicted_class = int(pred_label)  # 供你的示例代码使用

    probability = predicted_proba[predicted_class] * 100.0

    # 2) 文本渲染（按你示例）
    text = f"Based on feature values, predicted possibility of Stroke is {probability:.2f}%"
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(
        0.5, 0.5, text,
        fontsize=16, ha='center', va='center',
        fontname='Times New Roman',   # 若容器无该字体，最多告警；你坚持用我就不改
        transform=ax.transAxes
    )
    ax.axis('off')
    plt.savefig("prediction_text.png", bbox_inches='tight', dpi=300)
    st.image("prediction_text.png")

    # 3) SHAP —— 严格用你的 force_plot 调用，但保证传入“单样本、一维”向量
    #    （并在 matplotlib=True 不被支持时，自动回退到 HTML 版本）
    #    a) 先拿到模型真实输入（one-hot 后）
    Xt = preprocess.transform(X_user)
    if sparse.issparse(Xt):
        Xt = Xt.toarray()

    #    b) 与示例一致的变量名
    features = Xt           # 你示例里用的变量名
    model = rf              # 你示例里的 model

    explainer = shap.TreeExplainer(model)
    raw_shap = explainer.shap_values(features)
    expected = explainer.expected_value

    #    c) 统一为 (n_features,) 的单样本一维向量（否则 force_plot 会当成“多样本”）
    if isinstance(raw_shap, list):
        # 多分类：先取当前类别的 (n_samples, n_features)，再取第0个样本
        sv_vec = np.array(raw_shap[predicted_class][0]).ravel()
        base = expected[predicted_class] if isinstance(expected, (list, np.ndarray)) else expected
    else:
        # 二分类返回 (n_samples, n_features)
        sv_vec = np.array(raw_shap[0]).ravel()
        base = expected if not isinstance(expected, (list, np.ndarray)) else expected[0]

    # one-hot 后特征名
    try:
        feature_names = preprocess.get_feature_names_out()
    except Exception:
        feature_names = np.array([f"f{i}" for i in range(features.shape[1])])

    try:
        # 你示例里的 force_plot（matplotlib=True）
        shap.force_plot(
            base,
            sv_vec,
            pd.Series(features[0], index=feature_names),
            matplotlib=True
        )
        plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
        st.image("shap_force_plot.png")
    except NotImplementedError:
        # 某些 shap 版本不支持 matplotlib=True；回退到 HTML 强制图
        plot = shap.force_plot(
            base,
            sv_vec,
            pd.Series(features[0], index=feature_names),
            matplotlib=False
        )
        import streamlit.components.v1 as components
        components.html(shap.getjs() + plot.html(), height=400, scrolling=True)
