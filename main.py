import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os

# --- Configuration ---
st.set_page_config(page_title="Dashboard Scoring Crédit", layout="wide")
st.title("📊 Dashboard Scoring Crédit - Prototype")

# --- Chargement données ---
@st.cache_data
def load_data():
    data_path = os.path.join("Simulations", "Data", "features_for_prediction.csv")
    if not os.path.exists(data_path):
        st.error(f"Le fichier {data_path} est introuvable.")
        st.stop()
    df = pd.read_csv(data_path)
    return df

df = load_data()

numeric_cols = df.select_dtypes(include="number").columns.tolist()
client_ids = df["SK_ID_CURR"].unique()

# --- Chargement modèle ---
@st.cache_resource
def load_model():
    model_path = os.path.join("Simulations", "Best_model", "lgbm_pipeline1.pkl")
    if not os.path.exists(model_path):
        st.error("Fichier modèle introuvable.")
        st.stop()
    model_bundle = joblib.load(model_path)
    return model_bundle

model_bundle = load_model()
pipeline = model_bundle["pipeline"]
expected_features = model_bundle["features"]
model = pipeline.steps[-1][1]

# --- Sélection client ---
client_id = st.selectbox("🔎 Sélectionnez un client", client_ids)
client_data = df[df["SK_ID_CURR"] == client_id].iloc[0]

# --- Prédiction ---
X_client = df[df["SK_ID_CURR"] == client_id][expected_features].copy().apply(pd.to_numeric, errors='coerce').fillna(0)

pred_proba = model.predict_proba(X_client)[0][1]  # Probabilité défaut
decision = "Refus" if pred_proba >= 0.5 else "Accord"

col1, col2 = st.columns([1, 2])
with col1:
    st.metric("Probabilité défaut (%)", f"{pred_proba*100:.2f}%")
    if decision == "Accord":
        st.success(f"Décision : {decision}")
    else:
        st.error(f"Décision : {decision}")

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pred_proba*100,
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "green" if pred_proba < 0.5 else "red"},
            'steps': [
                {'range': [0, 50], 'color': 'green'},
                {'range': [50, 100], 'color': 'red'}
            ],
            'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 50}
        }
    ))
    st.plotly_chart(gauge, use_container_width=True)

with col2:
    st.markdown("### Caractéristiques du client")
    st.dataframe(client_data[numeric_cols])

# --- SHAP ---
booster = model.booster_ if hasattr(model, "booster_") else model
explainer = shap.TreeExplainer(booster)

st.markdown("---")
st.markdown("## Explication SHAP Globale")

X_all = df[expected_features].copy().apply(pd.to_numeric, errors='coerce').fillna(0)
shap_vals_global = explainer.shap_values(X_all)
shap_vals_global_use = shap_vals_global[1] if isinstance(shap_vals_global, list) else shap_vals_global

fig_global, ax = plt.subplots()
shap.summary_plot(shap_vals_global_use, X_all, plot_type="bar", show=False, max_display=10)
st.pyplot(fig_global)

st.markdown("## Explication SHAP Locale")
shap_values_local = explainer(X_client)
local_exp = shap_values_local[1][0] if isinstance(shap_values_local, list) else shap_values_local[0]

fig_local = plt.figure()
shap.waterfall_plot(local_exp, show=False)
st.pyplot(fig_local)

# --- Analyse univariée ---
st.markdown("---")
st.markdown("## Analyse univariée")

var_uni = st.selectbox("Variable à comparer", numeric_cols, key="var_uni")
fig_uni = px.histogram(df, x=var_uni, nbins=30, title=f"Distribution de {var_uni}")
fig_uni.add_vline(x=client_data[var_uni], line_dash="dash", line_color="red", annotation_text="Client")
st.plotly_chart(fig_uni, use_container_width=True)

# --- Analyse bivariée ---
st.markdown("---")
st.markdown("## Analyse bivariée")

var_x = st.selectbox("Feature X", numeric_cols, index=0, key="var_x")
var_y = st.selectbox("Feature Y", numeric_cols, index=1, key="var_y")

fig_bi = px.scatter(df, x=var_x, y=var_y, title=f"{var_x} vs {var_y}", opacity=0.5)
fig_bi.add_scatter(x=[client_data[var_x]], y=[client_data[var_y]], mode='markers',
                   marker=dict(color='red', size=15), name="Client")
st.plotly_chart(fig_bi, use_container_width=True)

st.subheader("✏️ Modification des Informations du Client")

modifiable_client_data = X_client.copy()
modified_features = {}

st.write("Modifiez les valeurs des variables ci-dessous puis cliquez sur 'Recalculer le score'.")

for feature in numeric_cols:
    min_val = float(df[feature].min())
    max_val = float(df[feature].max())
    default_val = float(client_data[feature])
    step_val = (max_val - min_val) / 100 if max_val != min_val else 0.01

    modified_features[feature] = st.slider(
        label=f"{feature}",
        min_value=min_val,
        max_value=max_val,
        value=default_val,
        step=step_val
    )

# Mise à jour du dataframe avec les nouvelles valeurs
for feature, value in modified_features.items():
    modifiable_client_data[feature] = value

if st.button("Recalculer le score avec les valeurs modifiées"):
    new_pred = model.predict_proba(modifiable_client_data)[0][1]
    st.write(f"Nouvelle probabilité de défaut : {new_pred:.3f}%")

    new_decision = "✅ Accord" if new_pred < 0.5 else "❌ Refus"
    st.markdown(f"### {new_decision}")

    new_shap_values_local = explainer(modifiable_client_data)
    new_local_explanation = (
        new_shap_values_local[1][0] if isinstance(new_shap_values_local, list) else new_shap_values_local[0]
    )

    fig_new_local = plt.figure()
    shap.waterfall_plot(new_local_explanation, show=False)
    st.pyplot(fig_new_local)

    st.caption(
        """
        **📝 Explication :** Ce graphique montre l'impact des nouvelles valeurs de features sur la prédiction mise à jour.  
        🔹 Les valeurs positives augmentent le risque de défaut,  
        🔹 Les valeurs négatives le réduisent.  
        """
    )
