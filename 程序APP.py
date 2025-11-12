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

# Streamlit 界面
st.title("CumCTI: Predicting Future Stroke Progression with SHAP Visualization")

# 动态生成输入项
st.header("Enter the following feature values:")
feature_values = []
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        value = st.number_input(
            label=f"{feature} ({properties['min']} - {properties['max']})",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
        )
    elif properties["type"] == "categorical":
        value = st.selectbox(
            label=f"{feature} (Select a value)",
            options=properties["options"],
        )
    feature_values.append(value)

# 转换为模型输入格式
features = np.array([feature_values])

# 预测与 SHAP 可视化
if st.button("Predict"):
    # 模型预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 提取预测的类别概率
    probability = predicted_proba[predicted_class] * 100

    # 显示预测结果，使用 Matplotlib 渲染指定字体
    text = f"Based on feature values, predicted possibility of Stroke is {probability:.2f}%"
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(
        0.5, 0.5, text,
        fontsize=16,
        ha='center', va='center',
        fontname='Times New Roman',
        transform=ax.transAxes
    )
    ax.axis('off')
    plt.savefig("prediction_text.png", bbox_inches='tight', dpi=300)
    st.image("prediction_text.png")

    # 计算 SHAP 值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_ranges.keys()))

    # 生成 SHAP 力图
    class_index = predicted_class  # 当前预测类别
    shap_fig = shap.force_plot(
        explainer.expected_value[class_index],
        shap_values[:,:,class_index],
        pd.DataFrame([feature_values], columns=feature_ranges.keys()),
        matplotlib=True,
    )
    # 保存并显示 SHAP 图
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")

