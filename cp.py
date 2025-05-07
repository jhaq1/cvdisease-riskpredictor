#%% -*- coding: utf-8 -*-
"""
Created on Sat May 4, 2025
Capstone Project - EN.585.771.81.SP25
Cardiovascular Disease Risk Predictor
@author: Junaid Haq
"""

#%% Imports

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import streamlit as st
st.set_page_config(layout="wide")

#%% Data Import and Preprocessing

df = pd.read_csv('heart_data.csv')
df = df.drop(['index', 'id'], axis=1)
df['age'] = df['age']/356
df['cholesterol'] = (df['cholesterol'] == 3).astype(int)
df['gluc'] = (df['gluc'] == 3).astype(int)

x = df.drop('cardio', axis=1)
y = df['cardio']

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
x_train, x_val, y_train, y_val = train_test_split(x_scaled, y, test_size=0.2, random_state=42, stratify=y)

#%% Define Neural Network and Train the Model

@st.cache_resource
def train_model(x_train, y_train, x_val, y_val):
    def sensitivity(y_true, y_pred):
        y_true = K.cast(y_true, 'float32')
        true_pos = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        poss_pos = K.sum(K.round(K.clip(y_true, 0, 1)))
        return true_pos / (poss_pos + K.epsilon())

    def specificity(y_true, y_pred):
        y_true = K.cast(y_true, 'float32')
        true_neg = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
        poss_neg = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
        return true_neg / (poss_neg + K.epsilon())

    model = models.Sequential([
        layers.Dense(16, activation='relu', input_shape=(x_train.shape[1],)),
        layers.Dense(8, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', sensitivity, specificity])
    hist = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=38, batch_size=32, verbose=0)
    
    return model, hist


model, hist = train_model(x_train, y_train, x_val, y_val)

#%% Precompute ROC

y_pred_proba = model.predict(x_val).ravel()
fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
roc_auc = auc(fpr, tpr)

#%% Web App

st.markdown(
    """
    <div style="display: flex; align-items: center;">
        <img src='https://images.emojiterra.com/google/noto-emoji/unicode-16.0/color/svg/1fac0.svg' style='width:80px; height:80px; margin-right:10px;'/>
        <h1 style='margin: 0; white-space: nowrap;'>Cardiovascular Disease Risk Predictor</h1>
    </div>
    """,
    unsafe_allow_html=True
)

user_col, divider_col, metrics_col = st.columns([1.5,.1,3.4])

with user_col:
    st.header("Patient Health Data")
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.slider("Age (years)", 20, 80, 39)
    height = st.slider("Height (cm)", 120, 220, 162)
    weight = st.slider("Weight (kg)", 40, 200, 85)
    ap_hi = st.slider("Systolic BP (mmHg)", 90, 200, 120)
    ap_lo = st.slider("Diastolic BP (mmHg)", 60, 140, 80)
    cholesterol_level = st.selectbox("Total cholesterol level", ["Normal", "Above Normal", "Well Above Normal"])
    gluc_level = st.selectbox("Total glucose level", ["Normal", "Above Normal", "Well Above Normal"])
    smoke = st.selectbox("Do you smoke?", ["Yes", "No"])
    alco = st.selectbox("Do you consume alcohol?", ["Yes", "No"])
    active = st.selectbox("Are you physically active?", ["Yes", "No"])

with divider_col:
    st.markdown("<div style='height:100%; border-left: 10px solid #ccc;'></div>", unsafe_allow_html=True)

# Map inputs
user_input = pd.DataFrame([{
    "age": age / 356,
    "gender": 1 if gender == "Male" else 0,
    "height": height,
    "weight": weight,
    "ap_hi": ap_hi,
    "ap_lo": ap_lo,
    "smoke": 1 if smoke == "Yes" else 0,
    "alco": 1 if alco == "Yes" else 0,
    "active": 1 if active == "Yes" else 0,
    "cholesterol": 1 if cholesterol_level == "Well Above Normal" else 0,
    "gluc": 1 if gluc_level == "Well Above Normal" else 0,
}])

user_scaled = scaler.transform(user_input.reindex(columns=x.columns))
prediction = model.predict(user_scaled)[0][0]

def draw_neural_net(ax, left, right, bottom, top, layer_sizes):
    ax.axis('off')
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom) / float(max(layer_sizes))
    h_spacing = (right - left) / float(n_layers - 1)

    # Store positions for annotations
    node_positions = {}

    for i, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2. + (top + bottom) / 2.
        node_positions[i] = []
        for j in range(layer_size):
            x = left + i * h_spacing
            y = layer_top - j * v_spacing

            # Determine color based on layer
            if i == 0:
                color = 'skyblue'  # Input layer
            elif i == n_layers - 1:
                color = 'lightcoral'  # Output layer
            else:
                color = 'lightgray'  # Hidden layers

            circle = plt.Circle((x, y), 0.02, color=color, ec='k', zorder=4)
            ax.add_artist(circle)
            node_positions[i].append((x, y))

        # Add layer label
        if i == 0:
            ax.text(x, top + 0.05, 'Input Layer', ha='center', fontsize=9, color='blue')
        elif i == n_layers - 1:
            ax.text(x, top + 0.05, 'Output Layer', ha='center', fontsize=9, color='red')
        else:
            ax.text(x, top + 0.05, f'Hidden Layer {i}', ha='center', fontsize=9, color='gray')

    # Draw connections
    for i in range(n_layers - 1):
        for a in node_positions[i]:
            for b in node_positions[i + 1]:
                line = plt.Line2D([a[0], b[0]], [a[1], b[1]], c='k', lw=0.5, zorder=1)
                ax.add_artist(line)

# Determine background color
if prediction < 0.3:
    bg_color = "#d0f0c0"
    txt = """
    **Risk Assessment: 🟢 Low**

    Your predicted risk of developing cardiovascular disease is low. This does *not* mean you are entirely free from risk, but rather that—based on the information provided—your likelihood is currently minimal.

    Maintaining a heart-healthy lifestyle is still essential. We recommend continuing regular checkups, staying physically active, following a balanced diet, and avoiding smoking or excessive alcohol consumption.

    *This result is a prediction based on statistical modeling and does not constitute a medical diagnosis. Please consult your healthcare provider for personalized medical advice.*
    """
elif prediction < 0.5:
    bg_color = "#fff7b3"
    txt = """
    **Risk Assessment: 🟡 Moderate**
    
    Your predicted risk of developing cardiovascular disease is moderate. This suggests that certain factors in your profile may be elevating your risk level, though it does *not* confirm the presence of disease.
    
    It may be beneficial to consult your primary care physician or a cardiovascular specialist to further evaluate your health. Lifestyle modifications—such as improved nutrition, increased physical activity, stress reduction, and smoking cessation—could help lower your risk.
    
    *This is a model-based prediction and not a clinical diagnosis. Any medical decisions should be made in partnership with your healthcare provider.*
    """

else:
    bg_color = "#f5b5b5"
    txt = """
    **Risk Assessment: 🔴 High**
    
    Your predicted risk of developing cardiovascular disease is high. This means the model has identified a strong statistical likelihood based on your current health data.
    
    We strongly encourage you to schedule a comprehensive evaluation with a licensed medical professional or cardiologist as soon as possible. Early intervention can be crucial in reducing long-term risk and improving outcomes.
    
    *Please note: This is not a diagnostic tool. It is a risk estimate generated by a machine learning model and should not replace medical advice, diagnosis, or treatment. Always consult your doctor before making healthcare decisions.*
    """

with metrics_col:
    with st.container():  # Helps control layout height better
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
            <div style='
                padding: 1rem;
                border-radius: 10px;
                background-color: {bg_color};
                display: flex;
                align-items: center;
                justify-content: flex-start;
            '>
                <div style="display: flex; align-items: center;">
                    <div style="font-size: 1.75rem; font-weight: 600; margin-right: 1rem;">
                        Predicted Risk:
                    </div>
                    <div style="font-size: 1.75rem; font-weight: 600;">
                        {prediction:.2%}
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Right-aligned expander
        st.markdown("<div style='text-align: right;'>", unsafe_allow_html=True)
        with st.expander("ℹ️ What does my predicted risk mean?"):
            st.markdown(txt)
        st.markdown("</div>", unsafe_allow_html=True)

    # Create layout
    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)

    with row1_col1:
        #st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### Neural Network Schematic")
        fig, ax = plt.subplots(figsize=(6,3))
        ax.axis('off')
        draw_neural_net(ax, .1, .9, .1, .9, [x_train.shape[1], 16, 8, 1])
        st.pyplot(fig)

    with row1_col2:
        #st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### Feature Correlation")
        label_map = {
            "active": "Exercise",
            "smoke": "Smoking",
            "height": "Height",
            "alco": "Alcohol",
            "gender": "Gender",
            "ap_hi": "Systolic BP",
            "ap_lo": "Diastolic BP",
            "gluc": "Glucose",
            "weight": "Weight",
            "cholesterol": "Cholesterol",
            "age": "Age"
            }
        correlations = x.corrwith(y)
        correlations.rename(index=label_map, inplace=True)
        correlations = correlations.sort_values()        
        fig_corr = plt.figure(figsize=(6,3))
        sns.barplot(x=correlations.values, y=correlations.index, palette='coolwarm')
        plt.xlabel("Correlation with Risk Modeling")
        plt.ylabel("")
        st.pyplot(fig_corr)

    # --- Row 2 ---
    with row2_col1:
        st.markdown("##### Training Metrics by Epoch")
        option = st.selectbox("", ["Accuracy & Loss", "Sensitivity & Specificity"])
        if option == "Accuracy & Loss":
            fig_acc_loss, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 3), sharex=True)
            ax1.plot(hist.history['accuracy'], label='Train', color='tab:blue')
            ax1.plot(hist.history['val_accuracy'], label='Validation', color='tab:orange')
            ax1.set_ylabel("Accuracy")
            ax1.legend()
            
            ax2.plot(hist.history['loss'], label='Train', color='tab:blue')
            ax2.plot(hist.history['val_loss'], label='Validation', color='tab:orange')
            ax2.set_ylabel("Loss")
            ax2.set_xlabel("Epoch")
            ax2.legend()
            st.pyplot(fig_acc_loss)
        else:
            fig_sens_spec, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 3), sharex=True)
            ax1.plot(hist.history['sensitivity'], label='Train', color='tab:blue')
            ax1.plot(hist.history['val_sensitivity'], label='Validation', color='tab:orange')
            ax1.set_ylabel("Sensitivity")
            ax1.legend()
            
            ax2.plot(hist.history['specificity'], label='Train', color='tab:blue')
            ax2.plot(hist.history['val_specificity'], label='Validation', color='tab:orange')
            ax2.set_ylabel("Specificity")
            ax2.set_xlabel("Epoch")
            ax2.legend()
            st.pyplot(fig_sens_spec)

    with row2_col2:
        st.markdown("##### ROC Curve")
        fig_roc = plt.figure(figsize=(6,3.5))
        plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.2f}')
        plt.fill_between(fpr, tpr - 0.02, tpr + 0.02, alpha=0.2)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('1 - Specificity')
        plt.ylabel('Sensitivity')
        plt.legend()
        st.pyplot(fig_roc)

