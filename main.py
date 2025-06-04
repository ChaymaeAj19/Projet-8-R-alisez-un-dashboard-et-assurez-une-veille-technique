import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import os

# === Configuration de la page ===
st.set_page_config(page_title="Dashboard Scoring Crédit", layout="wide")

# === Chargement du modèle et des données ===
model_path = os.path.join("Simulations", "Best_model", "lgbm_pipeline1.pkl")
data_path = os.path.join("Simulations", "Data", "features_for_prediction.csv")

@st.cache_resource
def load_model():
    model_bundle = joblib.load(model_path)
    pipeline = model_bundle['pipeline']
    expected_features = model_bundle['features']
    model = pipeline.steps[-1][1]
    return pipeline, expected_features, model

@st.cache_data
def load_data():
    return pd.read_csv(data_path)

pipeline, expected_features, model = load_model()
data = load_data()

X_all = data[expected_features].copy().apply(pd.to_numeric, errors='coerce').fillna(0)

explainer = shap.Explainer(model, X_all)

# === TABS ===
tabs = st.tabs(["Score & SHAP", "Modifier Client", "Nouveau Client", "Comparaison", "Analyse bi-variée"])

# === TAB 1 ===
with tabs[0]:
    st.title("🔍 Score & SHAP pour un client existant")
    client_id = st.selectbox("Sélectionnez un client", data['SK_ID_CURR'].unique())
    client_data = data[data["SK_ID_CURR"] == client_id]
    X_client = client_data[expected_features].copy().apply(pd.to_numeric, errors='coerce').fillna(0)

    proba = pipeline.predict_proba(X_client)[0][1]
    decision = "✅ Prêt accordé" if proba < 0.5 else "❌ Prêt refusé"

    st.metric("Score de crédit (%)", f"{proba*100:.2f}%", help="Probabilité de défaut")
    st.markdown(f"### Décision : <span style='color:{'green' if proba < 0.5 else 'red'}'>{decision}</span>", unsafe_allow_html=True)

    st.subheader("Explication locale (SHAP)")
    explanation = explainer(X_client)
    fig, ax = plt.subplots()
    shap.plots.waterfall(explanation[0], show=False)
    st.pyplot(fig)

    st.subheader("Explication globale (SHAP)")
    fig2, ax2 = plt.subplots()
    shap.summary_plot(explainer.shap_values(X_all), X_all, plot_type="bar", show=False)
    st.pyplot(fig2)

# === TAB 2 ===
with tabs[1]:
    st.title("✏️ Modifier un client")
    client_id = st.selectbox("Client à modifier", data['SK_ID_CURR'].unique(), key="edit")
    client_data = data[data['SK_ID_CURR'] == client_id][expected_features].iloc[0]

    edited_data = {}
    for col in expected_features:
        edited_data[col] = st.number_input(col, value=float(client_data[col]))

    edited_df = pd.DataFrame([edited_data])
    edited_df = edited_df.apply(pd.to_numeric, errors='coerce').fillna(0)

    proba = pipeline.predict_proba(edited_df)[0][1]
    decision = "✅ Prêt accordé" if proba < 0.5 else "❌ Prêt refusé"

    st.metric("Nouveau score de crédit", f"{proba*100:.2f}%")
    st.markdown(f"### Décision : <span style='color:{'green' if proba < 0.5 else 'red'}'>{decision}</span>", unsafe_allow_html=True)

    explanation = explainer(edited_df)
    fig, ax = plt.subplots()
    shap.plots.waterfall(explanation[0], show=False)
    st.pyplot(fig)

# === TAB 3 ===
with tabs[2]:
    st.title("🆕 Nouveau client")
    new_data = {}
    for col in expected_features:
        new_data[col] = st.number_input(f"{col}", key=f"new_{col}")

    new_df = pd.DataFrame([new_data]).apply(pd.to_numeric, errors='coerce').fillna(0)
    if st.button("Calculer le score"):
        proba = pipeline.predict_proba(new_df)[0][1]
        decision = "✅ Prêt accordé" if proba < 0.5 else "❌ Prêt refusé"
        st.metric("Score nouveau client", f"{proba*100:.2f}%")
        st.markdown(f"### Décision : <span style='color:{'green' if proba < 0.5 else 'red'}'>{decision}</span>", unsafe_allow_html=True)

        explanation = explainer(new_df)
        fig, ax = plt.subplots()
        shap.plots.waterfall(explanation[0], show=False)
        st.pyplot(fig)

# === TAB 4 ===
with tabs[3]:
    st.title("📊 Comparaison avec la population")
    var = st.selectbox("Choisissez une variable", expected_features)
    fig = px.histogram(data, x=var, nbins=30, title=f"Distribution de {var}")
    st.plotly_chart(fig)

# === TAB 5 ===
with tabs[4]:
    st.title("🔍 Analyse bi-variée")
    col1 = st.selectbox("Feature 1", expected_features, key="var1")
    col2 = st.selectbox("Feature 2", expected_features, key="var2")
    fig = px.scatter(data, x=col1, y=col2, color="SK_ID_CURR", title=f"{col1} vs {col2}")
    st.plotly_chart(fig)

# === Footer ===
st.markdown("---")
st.caption("Prototype complet - Scoring client avec SHAP et édition dynamique")
