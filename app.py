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
        print("Model loaded.")
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
    st.success("System ready!")
except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.stop()


with st.sidebar:
    st.subheader("📊 Metrics")
    if hasattr(predictor, 'metrics') and predictor.metrics:
        # R2 Score
        r2 = predictor.metrics.get('r2', 0)
        st.metric("R² Score", f"{r2:.2%}")

        # MAE
        st.metric("MAE", f"{predictor.metrics.get('mae', 0):.2f}")

        # RMSE
        st.metric("RMSE", f"{predictor.metrics.get('rmse', 0):.2f}")

    else:
        st.warning("No metrics saved. Retrain model.")

# --- 2. INTERFEJS UŻYTKOWNIKA ---
st.divider()
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📦 Product Selection")
    product_list = predictor.get_product_list()
    selected_product_id = st.selectbox("Choose ID:", product_list)

    # Pobranie danych z naszego obiektu
    details = predictor.get_product_details(selected_product_id)

    st.markdown(f"""
    **Category:** {details['category']}  \n
    **Actual price:** {details['price']:.2f} \n
    **Market price:** {details['competitor_price']:.2f}
    """)

    new_price = st.slider("Simulated price:",
                          min_value=details['price'] * 0.5,
                          max_value=details['price'] * 1.5,
                          value=details['price'])

with col2:
    st.subheader("📈 Simulation Results")

    # Wywołanie predykcji z oddzielnego pliku
    pred_demand = predictor.predict_demand(new_price, details['competitor_price'], details['category'], selected_product_id)
    estimated_revenue = pred_demand * new_price

    c1, c2 = st.columns(2)
    c1.metric("Forecasted demand", f"{pred_demand:.2f} szt.")
    c2.metric("Forecasted revenue", f"{estimated_revenue:.2f}")

# --- 3. AGENT GEMINI ---
st.divider()
st.subheader("Analyst's Strategic Review")

if api_key:
    if st.button("Ask for analyst's opinion"):
        llm = ChatGoogleGenerativeAI(
            model= "gemini-2.5-flash-lite",
            google_api_key=api_key
        )

        template = """
            You are a pricing expert. You are evaluating the simulation for product: {category}.
            
            Market situation:
            - Competitor price: {competitor_price: .2f}
            - Old price: {old_price: .2f}
            - New simulated price: {new_price: .2f}
            
            AI model result:
            - Projected demand: {demand: .2f}
            - Projected revenue: {revenue: .2f}
            
            Succinctly rate (max 4 points):
            1. Traffic profitability.
            2. Risk.
            3. Recommendation.
            4. What price would you suggest?
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

        with st.spinner("Analyst thinks..."):
            res = llm.invoke(formatted_prompt)
            st.markdown(res.content)
else:
    st.warning("Wprowadź klucz API, aby aktywować Agenta.")

# --- 4. WYKRES ---
st.divider()
prices = np.linspace(details['price'] * 0.5, details['price'] * 1.5, 50)
demands = [predictor.predict_demand(p, details['competitor_price'], details['category'], selected_product_id) for p in prices]

fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(prices, demands, label='Demand Curve')
ax.scatter([new_price], [pred_demand], color='red', zorder=5, label='Your choice')
ax.set_title(f"Price elasticity: {details['category']}")
ax.set_xlabel("Price")
ax.legend()
st.pyplot(fig)