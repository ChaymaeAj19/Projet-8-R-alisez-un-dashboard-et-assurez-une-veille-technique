import streamlit as st
import pandas as pd
import numpy as np
import requests
import shap
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import os

# === CONFIG ===
st.set_page_config(layout="wide")
API_URL = "https://projet-7-implementation.onrender.com"
MODEL_PATH = "Simulations/Best_model/lgbm_pipeline1.pkl"
DATA_PATH = "Simulations/Data/features_for_prediction.csv"

# === CHARGEMENT MODELE ET DONNEES ===
@st.cache_resource
def load_model_and_data():
    model_bundle = joblib.load(MODEL_PATH)
    pipeline = model_bundle["pipeline"]
    features = model_bundle["features"]
    df = pd.read_csv(DATA_PATH)
    return pipeline, features, df

pipeline, expected_features, df_all_clients = load_model_and_data()
model = pipeline.steps[-1][1]
explainer = shap.TreeExplainer(model)

# === SIDEBAR: Sélection ou ajout client ===
st.sidebar.header("🔎 Client")
client_ids = df_all_clients["SK_ID_CURR"].tolist()
selected_id = st.sidebar.selectbox("Sélectionnez un client", client_ids)
new_client = st.sidebar.checkbox("➕ Ajouter un nouveau client")

# === NOUVEAU CLIENT ===
if new_client:
    st.title("📋 Nouveau client")
    new_data = {}
    for col in expected_features:
        val = st.number_input(f"{col}", value=float(df_all_clients[col].mean()))
        new_data[col] = val
    if st.button("💾 Enregistrer et prédire"):
        df_temp = pd.DataFrame([new_data])
        proba = pipeline.predict_proba(df_temp)[0][1]
        shap_values = explainer.shap_values(df_temp)
        st.success(f"Score de crédit : {proba:.2%}")
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap.Explanation(
            values=shap_values[1][0],
            base_values=explainer.expected_value[1],
            data=df_temp.iloc[0],
            feature_names=df_temp.columns
        ), show=False)
        st.pyplot(fig)
        st.stop()

# === CLIENT EXISTANT ===
client_data = df_all_clients[df_all_clients["SK_ID_CURR"] == selected_id]
X_client = client_data[expected_features].copy().apply(pd.to_numeric, errors='coerce').fillna(0)

# === API CALL ===
@st.cache_data(show_spinner=False)
def get_api_prediction(client_id):
    try:
        response = requests.post(f"{API_URL}/predict", json={"SK_ID_CURR": int(client_id)})
        return response.json()
    except:
        return {"error": "Erreur de connexion à l'API"}

result = get_api_prediction(selected_id)
score = result.get("probability")

# === AFFICHAGE SCORE ===
st.title(f"📊 Scoring Client : {selected_id}")
col1, col2 = st.columns([1, 2])

with col1:
    if score is not None:
        st.metric("Score de crédit (%)", f"{round(score, 2)}%")
        st.success("✅ Accord") if score >= 50 else st.error("❌ Refus")
        fig = px.bar_polar(
            r=[score, 100 - score],
            theta=["Score", "Distance au seuil"],
            color=["Score", "Distance au seuil"],
            color_discrete_sequence=px.colors.sequential.Plasma_r
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Erreur lors de la récupération du score.")
        st.write(result)

with col2:
    st.markdown("#### 📄 Caractéristiques client")
    editable_client = st.data_editor(client_data, num_rows="fixed")
    if st.button("Mettre à jour client"):
        st.info("⚠️ Sauvegarde locale non implémentée — à ajouter si besoin.")

# === SHAP LOCAL ===
st.subheader("📌 Explication locale (SHAP)")
shap_values = explainer.shap_values(X_client)
explanation = shap.Explanation(
    values=shap_values[1][0],
    base_values=explainer.expected_value[1],
    data=X_client.iloc[0],
    feature_names=X_client.columns
)
fig_local, ax = plt.subplots()
shap.plots.waterfall(explanation, show=False)
st.pyplot(fig_local)

# === SHAP GLOBAL ===
st.subheader("🌍 SHAP global : Variables les + importantes")
X_all = df_all_clients[expected_features].copy().apply(pd.to_numeric, errors='coerce').fillna(0)
shap_vals_global = explainer.shap_values(X_all)
fig_global, ax = plt.subplots()
shap.summary_plot(shap_vals_global[1], X_all, plot_type="bar", show=False)
st.pyplot(fig_global)

# === ANALYSE UNIVARIEE ===
st.subheader("📈 Comparaison du client avec la population")
selected_var = st.selectbox("Variable à comparer", expected_features)
fig_uni = px.histogram(df_all_clients, x=selected_var, nbins=30)
fig_uni.add_vline(x=client_data[selected_var].values[0], line_color="red", annotation_text="Client", line_dash="dash")
st.plotly_chart(fig_uni, use_container_width=True)

# === ANALYSE BI-VARIEE ===
st.subheader("📊 Analyse bi-variée")
col_x, col_y = st.columns(2)
feature_x = col_x.selectbox("Variable X", expected_features)
feature_y = col_y.selectbox("Variable Y", expected_features)

fig_bi = px.scatter(df_all_clients, x=feature_x, y=feature_y, opacity=0.5)
fig_bi.add_scatter(x=[client_data[feature_x].values[0]],
                   y=[client_data[feature_y].values[0]],
                   mode='markers',
                   marker=dict(size=12, color='red'),
                   name="Client")
st.plotly_chart(fig_bi, use_container_width=True)

# === FIN ===
st.markdown("---")
st.caption("Prototype P7 - Dashboard Scoring Crédit | API + SHAP local")
