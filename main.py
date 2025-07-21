import streamlit as st
import pandas as pd
import requests
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os

# --- Config page ---
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
def predict_api(data_dict, use_id=True):
    try:
        if use_id and "SK_ID_CURR" in data_dict:
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

decision = "Accord" if score < 50 else "Refus"

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
            'bar': {'color': "green" if score < 50 else "red"},
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

# --- SHAP local & global ---
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

# --- Tableau éditable pour modification client ---
st.markdown("---")
st.markdown("## Modifier les informations du client")

# Préparer dataframe modifiable (client uniquement, variables numériques)
editable_df = pd.DataFrame(client_data[numeric_cols]).T.reset_index()
editable_df.columns = ["Variable", "Valeur"]

# --- Tableau éditable pour modification client ---
st.markdown("## Modifier les informations du client")

editable_df = pd.DataFrame({
    "Variable": numeric_cols,
    "Valeur": [client_data[col] for col in numeric_cols]
})

edited_df = st.data_editor(editable_df, num_rows="dynamic", use_container_width=True)

# Appliquer les modifications
data_for_pred = client_data.copy()
for _, row in edited_df.iterrows():
    var = row["Variable"]
    val = row["Valeur"]
    if var in data_for_pred.index:
        try:
            data_for_pred[var] = float(val)
        except:
            pass

data_for_pred_df = pd.DataFrame([data_for_pred[expected_features]])

if st.button("Recalculer le score avec les valeurs modifiées"):
    new_pred = model.predict_proba(data_for_pred_df)[0][1] * 100  # en %
    st.write(f"Nouvelle probabilité de défaut : {new_pred:.2f}%")

    new_decision = "Accord" if new_pred < 50 else "Refus"
    if new_decision == "Accord":
        st.success(f"Décision : {new_decision}")
    else:
        st.error(f"Décision : {new_decision}")

    new_shap_values_local = explainer(data_for_pred_df)
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

# --- Fin ---
st.markdown("---")
st.caption("Dashboard fonctionnel avec score, interprétabilité SHAP réelle, comparaisons et édition client via tableau.")
