# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from demand_model2 import DemandPredictor
import os

api_key = 'AIzaSyA6B65C29gm4QFT-AT81K-EngPgxvYOW6s'

# --- KONFIGURACJA STRONY ---
st.set_page_config(page_title="Price Optimizer", layout="wide")
st.title("Price Optimizer & Demand Forecaster")


# --- 1. INICJALIZACJA MODELU (CACHED) ---
@st.cache_resource
def get_model(force_retrain=False):
    predictor = DemandPredictor(data_path='data/')

    # Sprawdzamy czy model istnieje i czy nie wymuszamy treningu
    model_loaded = False
    if not force_retrain:
        model_loaded = predictor.load_saved_model()

    if model_loaded:
        print("Model wczytany z dysku.")
    else:
        # Jeśli nie ma modelu lub wymuszono trening
        with st.spinner():
            predictor.load_data()
            predictor.train()
            predictor.save_model()  # Zapisujemy na przyszłość

    return predictor

try:
    # Wywołujemy funkcję
    predictor = get_model()
    st.success("System gotowy do pracy!")
except Exception as e:
    st.error(f"Błąd inicjalizacji: {e}")
    st.stop()


with st.sidebar:
    st.subheader("📊 Jakość Modelu (Błędy)")
    if hasattr(predictor, 'metrics') and predictor.metrics:
        # R2 Score
        r2 = predictor.metrics.get('r2', 0)
        st.metric("R² Score (Dopasowanie)", f"{r2:.2%}")

        # MAE
        st.metric("MAE (Średni błąd w szt.)", f"{predictor.metrics.get('mae', 0):.2f}")

        # RMSE
        st.metric("RMSE (Błąd pierwiastkowy)", f"{predictor.metrics.get('rmse', 0):.2f}")

    else:
        st.warning("Brak zapisanych metryk. Przetrenuj model.")

# --- 2. INTERFEJS UŻYTKOWNIKA ---
st.divider()
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📦 Wybór Produktu")
    product_list = predictor.get_product_list()
    selected_product_id = st.selectbox("Wybierz ID:", product_list)

    # Pobranie danych z naszego obiektu
    details = predictor.get_product_details(selected_product_id)

    st.markdown(f"""
    **Kategoria:** {details['category']}  \n
    **Aktualna cena:** {details['price']:.2f} \n
    **Cena rynkowa:** {details['competitor_price']:.2f}
    """)

    new_price = st.slider("Symulowana cena:",
                          min_value=details['price'] * 0.5,
                          max_value=details['price'] * 1.5,
                          value=details['price'])

with col2:
    st.subheader("📈 Wyniki Symulacji")

    # Wywołanie predykcji z oddzielnego pliku
    pred_demand = predictor.predict_demand(new_price, details['competitor_price'], details['category'], selected_product_id)
    estimated_revenue = pred_demand * new_price

    c1, c2 = st.columns(2)
    c1.metric("Przewidywany Popyt (tyg.)", f"{pred_demand:.2f} szt.")
    c2.metric("Przewidywany Przychód", f"{estimated_revenue:.2f}")

# --- 3. AGENT GEMINI ---
st.divider()
st.subheader("Analityk")

if api_key:
    if st.button("Poproś o analizę strategiczną"):
        llm = ChatGoogleGenerativeAI(
            model= "gemini-2.5-flash-lite",
            google_api_key=api_key
        )

        template = """
        Jesteś ekspertem pricingowym. Oceniasz symulację dla produktu: {category}.

        Sytuacja rynkowa:
        - Cena konkurencji: {competitor_price:.2f}
        - Stara cena: {old_price:.2f}
        - Nowa cena symulowana: {new_price:.2f}

        Wynik modelu AI:
        - Przewidywany popyt: {demand:.2f}
        - Przewidywany przychód: {revenue:.2f}

        Oceń zwięźle (max 4 punkty):
        1. Opłacalność ruchu.
        2. Ryzyko.
        3. Rekomendacja.
        4. Jaką cenę byś zasugerował(-a)?
        """

        prompt = PromptTemplate.from_template(template)
        formatted_prompt = prompt.format(
            category=details['category'],
            competitor_price=details['competitor_price'],
            old_price=details['price'],
            new_price=new_price,
            demand=pred_demand,
            revenue=estimated_revenue
        )

        with st.spinner("Analityk myśli..."):
            res = llm.invoke(formatted_prompt)
            st.markdown(res.content)
else:
    st.warning("Wprowadź klucz API, aby aktywować Agenta.")

# --- 4. WYKRES ---
st.divider()
prices = np.linspace(details['price'] * 0.5, details['price'] * 1.5, 50)
demands = [predictor.predict_demand(p, details['competitor_price'], details['category'], selected_product_id) for p in prices]

fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(prices, demands, label='Krzywa Popytu')
ax.scatter([new_price], [pred_demand], color='red', zorder=5, label='Twój Wybór')
ax.set_title(f"Elastyczność cenowa: {details['category']}")
ax.set_xlabel("Cena")
ax.legend()
st.pyplot(fig)