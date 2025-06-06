import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import shap
import matplotlib.pyplot as plt
import os

# --- Configuration ---
st.set_page_config(page_title="Dashboard Scoring Crédit", layout="wide")
st.title("📊 Dashboard Scoring Crédit - Prototype")

API_URL = "https://projet-7-implementation.onrender.com"

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

# --- Sélection client ---
client_id = st.selectbox("🔎 Sélectionnez un client", client_ids)
client_data = df[df["SK_ID_CURR"] == client_id].iloc[0]

# --- Fonction API prédiction ---
def predict_api(data_dict):
    try:
        if "SK_ID_CURR" in data_dict:
            payload = {"SK_ID_CURR": int(data_dict["SK_ID_CURR"])}
        else:
            payload = {"data": data_dict}
        response = requests.post(f"{API_URL}/predict", json=payload)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# --- Affichage score & jauge ---
res = predict_api({"SK_ID_CURR": client_id})
score = res.get("probability", None)
if score is None:
    st.error("Erreur récupération score depuis l'API.")
    st.stop()

# Seuil ajusté à 5%
decision = "Accord" if score < 5 else "Refus"

col1, col2 = st.columns([1, 2])
with col1:
    st.metric("Score crédit (%)", f"{score:.2f}%")
    if decision == "Accord":
        st.success(f"Décision : {decision}")
    else:
        st.error(f"Décision : {decision}")

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "green" if score < 5 else "red"},
            'steps': [
                {'range': [0, 5], 'color': 'green'},
                {'range': [5, 100], 'color': 'red'}
            ],
            'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 5}
        }
    ))
    st.plotly_chart(gauge, use_container_width=True)

with col2:
    st.markdown("### Caractéristiques du client")
    st.dataframe(client_data[numeric_cols])

# --- SHAP local & global (réels) ---
st.markdown("---")
st.markdown("## Interprétabilité (Feature importance)")

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

X_all = df[expected_features].copy().apply(pd.to_numeric, errors='coerce').fillna(0)
X_client = df[df["SK_ID_CURR"] == client_id][expected_features].copy().apply(pd.to_numeric, errors='coerce').fillna(0)

booster = model.booster_ if hasattr(model, "booster_") else model
explainer = shap.TreeExplainer(booster)

st.markdown("### 🔍 Importance locale SHAP (réelle)")
explanation = explainer(X_client)
fig_local, ax = plt.subplots()
shap.plots.waterfall(explanation[0], show=False)
st.pyplot(fig_local)

st.markdown("### 🌍 Importance globale SHAP (réelle)")
shap_vals_global = explainer.shap_values(X_all)
shap_vals_global_use = shap_vals_global[1] if isinstance(shap_vals_global, list) else shap_vals_global
fig_global, ax = plt.subplots()
shap.summary_plot(shap_vals_global_use, X_all, plot_type="bar", show=False, max_display=10)
st.pyplot(fig_global)

# --- Histogramme univarié ---
st.markdown("---")
st.markdown("## Comparaison univariée")

var_uni = st.selectbox("Variable à comparer", numeric_cols)

fig_uni = px.histogram(df, x=var_uni, nbins=30, title=f"Distribution de {var_uni}")
fig_uni.add_vline(x=client_data[var_uni], line_dash="dash", line_color="red", annotation_text="Client")
st.plotly_chart(fig_uni, use_container_width=True)

# --- Analyse bi-variée ---
st.markdown("---")
st.markdown("## Analyse bivariée")

var_x = st.selectbox("Feature X", numeric_cols, index=0, key="var_x")
var_y = st.selectbox("Feature Y", numeric_cols, index=1, key="var_y")

fig_bi = px.scatter(df, x=var_x, y=var_y, title=f"Analyse bivariée : {var_x} vs {var_y}", opacity=0.5)
fig_bi.add_scatter(x=[client_data[var_x]], y=[client_data[var_y]], mode='markers',
                   marker=dict(color='red', size=15), name="Client")
st.plotly_chart(fig_bi, use_container_width=True)

# --- Formulaire modification client + recalcul dynamique ---
st.markdown("---")
st.markdown("## Modifier les informations du client")

with st.form("edit_form"):
    edited_features = {}
    for feat in numeric_cols:
        val = st.number_input(feat, value=float(client_data[feat]), format="%.4f")
        edited_features[feat] = val
    submit_edit = st.form_submit_button("Recalculer score")

if submit_edit:
    res_edit = predict_api(edited_features)
    score_edit = res_edit.get("probability", None)
    if score_edit is not None:
        st.success(f"Score recalculé : {score_edit:.2f}%")
        decision_edit = "Accord" if score_edit < 5 else "Refus"
        if decision_edit == "Accord":
            st.success(f"Décision : {decision_edit}")
        else:
            st.error(f"Décision : {decision_edit}")
    else:
        st.error("Erreur lors de la prédiction du score modifié.")

# --- Fin ---
st.markdown("---")
st.caption("Dashboard fonctionnel avec score, interprétabilité SHAP réelle, comparaisons et édition client.")
