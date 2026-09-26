import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Prédiction temps de livraison", layout="wide")

# --- Chargement des artefacts ---
@st.cache_resource
def load_artifacts():
    model = joblib.load('models/random_forest_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    feature_columns = joblib.load('models/feature_columns.joblib')
    metrics = joblib.load('models/metrics.joblib')
    return model, scaler, feature_columns, metrics

@st.cache_data
def load_data():
    return pd.read_csv('data/processed/food_delivery_clean.csv')

model, scaler, feature_columns, metrics = load_artifacts()
df = load_data()

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

st.title("🚴 Prédiction du temps de livraison")

tab1, tab2, tab3 = st.tabs(["Prédiction", "Visualisation des données", "Performance du modèle"])

# ============ TAB 1 : PREDICTION ============
with tab1:
    st.subheader("Renseignez les informations de la commande")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Livreur**")
        age = st.slider("Âge du livreur", 15, 50, 30)
        ratings = st.slider("Note du livreur", 1.0, 5.0, 4.7, step=0.1)
        vehicle_condition = st.selectbox("État du véhicule (0=mauvais, 3=excellent)", [0, 1, 2, 3], index=1)
        multiple_deliveries = st.selectbox("Livraisons multiples simultanées", [0, 1, 2, 3], index=1)

    with col2:
        st.markdown("**Trajet**")
        dist_mode = st.radio("Distance", ["Saisir directement (km)", "Calculer depuis coordonnées"])
        if dist_mode == "Saisir directement (km)":
            distance_km = st.number_input("Distance (km)", 0.5, 25.0, 8.0)
        else:
            r_lat = st.number_input("Latitude restaurant", value=19.0)
            r_lon = st.number_input("Longitude restaurant", value=76.0)
            d_lat = st.number_input("Latitude livraison", value=19.1)
            d_lon = st.number_input("Longitude livraison", value=76.1)
            distance_km = haversine(r_lat, r_lon, d_lat, d_lon)
            st.info(f"Distance calculée : {distance_km:.2f} km")

        order_hour = st.slider("Heure de la commande", 0, 23, 19)
        is_weekend = st.checkbox("Commande le weekend")

    with col3:
        st.markdown("**Conditions**")
        weather = st.selectbox("Météo", ["Sunny", "Cloudy", "Fog", "Stormy", "Sandstorms", "Windy", "Missing"])
        traffic = st.selectbox("Trafic", ["Low", "Medium", "High", "Jam", "Missing"])
        vehicle_type = st.selectbox("Type de véhicule", ["motorcycle", "scooter", "electric_scooter", "bicycle"])
        city = st.selectbox("Type de zone", ["Metropolitian", "Urban", "Semi-Urban"])
        festival = st.checkbox("Jour de festival")

    if st.button("Prédire le temps de livraison", type="primary"):
        # Construction du vecteur de features, aligné sur feature_columns
        row = pd.DataFrame([np.zeros(len(feature_columns))], columns=feature_columns)

        row['delivery_person_age'] = age
        row['delivery_person_ratings'] = ratings
        row['vehicle_condition'] = vehicle_condition
        row['multiple_deliveries'] = multiple_deliveries
        row['distance_km'] = distance_km
        row['order_hour'] = order_hour
        row['is_weekend'] = int(is_weekend)
        row['festival'] = int(festival)

        traffic_order = {'Low': 0, 'Medium': 1, 'High': 2, 'Jam': 3, 'Missing': -1}
        row['traffic'] = traffic_order[traffic]

        weather_col = f'weather_{weather}'
        if weather_col in row.columns:
            row[weather_col] = True

        vehicle_col = f'vehicle_type_{vehicle_type}'
        if vehicle_col in row.columns:
            row[vehicle_col] = True

        city_col = f'city_{city}'
        if city_col in row.columns:
            row[city_col] = True

        # Standardisation des colonnes numériques (mêmes que l'entraînement)
        cols_to_scale = ['delivery_person_age', 'delivery_person_ratings', 'distance_km', 'order_hour']
        row[cols_to_scale] = scaler.transform(row[cols_to_scale])

        prediction = model.predict(row)[0]

        st.success(f"### ⏱️ Temps de livraison estimé : **{prediction:.0f} minutes**")
        st.caption(f"Marge d'erreur moyenne du modèle : ± {metrics['mae']:.1f} minutes")

# ============ TAB 2 : VISUALISATION DES DONNÉES ============
with tab2:
    st.subheader("Exploration du dataset")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        ax.hist(df['delivery_time_min'], bins=30, edgecolor='black')
        ax.set_title("Distribution du temps de livraison")
        ax.set_xlabel("Minutes")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        df.boxplot(column='delivery_time_min', by='traffic', ax=ax)
        ax.set_title("Temps de livraison par niveau de trafic")
        plt.suptitle("")
        st.pyplot(fig)

    st.markdown("**Aperçu des données**")
    st.dataframe(df.head(20))

# ============ TAB 3 : PERFORMANCE DU MODÈLE ============
with tab3:
    st.subheader("Performance du modèle (Random Forest optimisé)")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE", f"{metrics['mae']:.2f} min")
    col2.metric("RMSE", f"{metrics['rmse']:.2f} min")
    col3.metric("R² (test)", f"{metrics['r2_test']:.3f}")
    col4.metric("R² ajusté", f"{metrics['r2_adjusted']:.3f}")

    st.markdown(f"""
    **Interprétation métier** : le modèle se trompe en moyenne de **{metrics['mae']:.1f} minutes**
    sur l'estimation du temps de livraison, et explique environ **{metrics['r2_test']*100:.0f}%**
    de la variabilité observée dans les données.
    """)

    fig, ax = plt.subplots()
    ax.bar(['Train', 'Test'], [metrics['r2_train'], metrics['r2_test']])
    ax.set_ylabel("R²")
    ax.set_title("Comparaison R² train vs test")
    ax.set_ylim(0, 1)
    st.pyplot(fig)