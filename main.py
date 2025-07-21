import streamlit as st
import pandas as pd
import requests

API_URL = "https://projet-7-implementation.onrender.com"

@st.cache_data
def load_data():
    return pd.read_csv("Simulations/Data/features_for_prediction.csv")

def predict_api(data_dict, use_id=True):
    """
    Envoie les données au endpoint API pour obtenir la prédiction.
    Si use_id=True, on envoie {"SK_ID_CURR": id}, sinon {"data": data_dict}.
    """
    try:
        if use_id and "SK_ID_CURR" in data_dict:
            payload = {"SK_ID_CURR": int(data_dict["SK_ID_CURR"])}
        else:
            payload = {"data": data_dict}
        response = requests.post(f"{API_URL}/predict", json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def main():
    st.title("📊 Dashboard Crédit Accessible avec API")

    data = load_data()

    client_id = st.selectbox("🔍 Sélectionnez un client (SK_ID_CURR)", data["SK_ID_CURR"].unique())

    client_data = data[data["SK_ID_CURR"] == client_id].iloc[0]

    st.subheader("Informations du client sélectionné")
    st.write(client_data)

    # Prédiction via API avec SK_ID_CURR
    res = predict_api({"SK_ID_CURR": client_id})
    score = res.get("probability", None)
    if score is None:
        st.error(f"Erreur récupération score depuis l'API : {res.get('error', 'Erreur inconnue')}")
        st.stop()
    st.metric("Probabilité de défaut (%)", f"{score * 100:.2f}%")
    decision = "✅ Approuvé" if score < 0.5 else "❌ Refusé"
    st.markdown(f"### Décision : {decision}")

    st.markdown("---")
    st.subheader("✏️ Modifier les variables du client et recalculer le score")

    with st.form("modification_form"):
        edited_features = {}
        for col in data.columns:
            if col != "SK_ID_CURR":
                val = st.number_input(label=col, value=float(client_data[col]), format="%.4f")
                edited_features[col] = val
        submit = st.form_submit_button("Recalculer le score")

    if submit:
        res_edit = predict_api(edited_features, use_id=False)
        score_edit = res_edit.get("probability", None)
        if score_edit is not None:
            st.success(f"Nouvelle probabilité de défaut : {score_edit * 100:.2f}%")
            new_decision = "✅ Approuvé" if score_edit < 0.5 else "❌ Refusé"
            st.markdown(f"### Nouvelle décision : {new_decision}")
        else:
            st.error(f"Erreur lors de la prédiction modifiée : {res_edit.get('error', 'Erreur inconnue')}")

if __name__ == "__main__":
    main()
