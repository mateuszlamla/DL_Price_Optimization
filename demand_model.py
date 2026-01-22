# demand_model.py
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.preprocessing import StandardScaler


class DemandPredictor:
    def __init__(self, data_path='data/'):
        self.data_path = data_path
        self.model = None
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.df = pd.DataFrame()
        self.features = ['price_normalized', 'price_ratio']
        self.product_base_prices = {}

    def load_data(self):
        """Wczytuje, czyści i agreguje dane."""
        try:
            orders = pd.read_csv(f'{self.data_path}olist_orders_dataset.csv')
            items = pd.read_csv(f'{self.data_path}olist_order_items_dataset.csv')
            products = pd.read_csv(f'{self.data_path}olist_products_dataset.csv')
        except FileNotFoundError:
            raise FileNotFoundError(f"Nie znaleziono plików CSV w folderze '{self.data_path}'.")

        # Merge danych
        df = pd.merge(orders, items, on='order_id', how='inner')
        df = pd.merge(df, products, on='product_id', how='inner')
        df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
        df = df[df['order_status'] == 'delivered']

        # Agregacja tygodniowa
        top_products = df['product_id'].value_counts().head(50).index.tolist()
        df_top = df[df['product_id'].isin(top_products)].copy()

        grouper = pd.Grouper(key='order_purchase_timestamp', freq='W')

        weekly_data = df_top.groupby(['product_id', 'product_category_name', grouper]).agg({
            'price': 'mean',
            'order_id': 'count'
        }).rename(columns={'order_id': 'demand'}).reset_index()

        weekly_data['demand'] = weekly_data['demand'].fillna(0)
        weekly_data['price'] = weekly_data.groupby('product_id')['price'].ffill()

        mean_prices = weekly_data.groupby('product_category_name')['price'].mean()
        self.product_base_prices = mean_prices.to_dict()

        weekly_data['base_price'] = weekly_data['product_category_name'].map(self.product_base_prices)
        weekly_data['price_normalized'] = weekly_data['price'] / weekly_data['base_price']

        cat_price = weekly_data.groupby(['product_category_name', 'order_purchase_timestamp'])['price'].mean().reset_index()
        cat_price.rename(columns={'price': 'competitor_price'}, inplace=True)

        final_df = pd.merge(weekly_data, cat_price, on=['product_category_name', 'order_purchase_timestamp'], how='left')
        final_df['price_ratio'] = final_df['price'] / final_df['competitor_price']

        self.df = final_df.dropna().reset_index(drop=True)
        return self.df

    def train(self, epochs=50):
        """Trenuje sieć neuronową."""
        if self.df.empty:
            raise ValueError("Brak danych. Najpierw uruchom load_data().")

        X = self.df[self.features].values
        Y = self.df[['demand']].values

        # Fit scalerów
        X_scaled = self.scaler_x.fit_transform(X)
        Y_scaled = self.scaler_y.fit_transform(Y)

        # Budowa modelu
        self.model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(X_scaled.shape[1],),
                         kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            layers.Dense(1, activation='linear')
        ])

        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.model.fit(X_scaled, Y_scaled, epochs=epochs, batch_size=32, verbose=0, validation_split=0.2)
        print("Trening zakończony.")

    def predict_demand(self, price, competitor_price, category_name=None):
        """Zwraca przewidywany popyt dla pojedynczego przypadku."""
        if self.model is None:
            raise ValueError("Model nie jest wytrenowany.")

        base_price = self.product_base_prices.get(category_name, price)
        if base_price == 0: base_price = 1  # Zabezpieczenie przed dzieleniem przez 0

        # Wyliczamy cechy tak samo jak w treningu
        price_normalized = price / base_price
        price_ratio = price / competitor_price

        # Input features: ['price_normalized', 'price_ratio']
        input_features = np.array([[price_normalized, price_ratio]])

        # Transformacja i predykcja
        input_scaled = self.scaler_x.transform(input_features)
        pred_scaled = self.model.predict(input_scaled, verbose=0)
        pred_demand = self.scaler_y.inverse_transform(pred_scaled)[0][0]

        return max(0, pred_demand)

    def get_product_details(self, product_category):
        """Pomocnicza funkcja do pobierania ostatnich danych o produkcie."""
        product_row = self.df[self.df['product_category_name'] == product_category].iloc[-1]
        return {
            'category': product_row['product_category_name'],
            'price': float(product_row['price']),
            'competitor_price': float(product_row['competitor_price']),
            'current_demand': float(product_row['demand'])
        }

    def get_product_list(self):
        return self.df['product_category_name'].unique()