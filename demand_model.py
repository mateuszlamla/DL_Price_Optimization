# demand_model.py
import pandas as pd
import numpy as np
import os
import joblib
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks, models
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


class DemandPredictor:
    def __init__(self, data_path='data/'):
        self.data_path = data_path
        self.model = None
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.cat_encoder = LabelEncoder()
        self.df = pd.DataFrame()

        # Lista cech (Features)
        self.num_features = [
            'price_normalized',
            'price_ratio',
            'price_vs_avg_4w',
            'price_delta',  # NOWOŚĆ: Zmiana ceny względem zeszłego tygodnia
            'month_sin',
            'month_cos',
            'is_black_friday',
            'freight_value',
            'product_photos_qty',
            'product_weight_g',
            'category_popularity',
            'demand_lag_1',
            'demand_lag_2',  # NOWOŚĆ: Popyt 2 tygodnie temu
            'demand_delta',  # NOWOŚĆ: (lag_1 - lag_2) czyli trend wzrostowy/spadkowy
            'demand_roll_mean_4w'
        ]
        self.product_base_prices = {}
        self.product_recent_stats = {}
        self.metrics = {}
        self.num_categories = 0

    def load_data(self):
        try:
            orders = pd.read_csv(f'{self.data_path}olist_orders_dataset.csv')
            items = pd.read_csv(f'{self.data_path}olist_order_items_dataset.csv')
            products = pd.read_csv(f'{self.data_path}olist_products_dataset.csv')
        except FileNotFoundError:
            raise FileNotFoundError(f"Nie znaleziono plików CSV w folderze '{self.data_path}'.")

        # Merge
        df = pd.merge(orders, items, on='order_id', how='inner')
        df = pd.merge(df, products, on='product_id', how='inner')
        df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
        df = df[df['order_status'] == 'delivered']

        # Zwiększamy zakres produktów do 600
        top_products = df['product_id'].value_counts().head(600).index.tolist()
        df_top = df[df['product_id'].isin(top_products)].copy()

        # Agregacja
        grouper = pd.Grouper(key='order_purchase_timestamp', freq='W')
        weekly_data = df_top.groupby(['product_id', 'product_category_name', grouper]).agg({
            'price': 'mean',
            'freight_value': 'mean',
            'product_photos_qty': 'first',
            'product_weight_g': 'first',
            'order_id': 'count'
        }).rename(columns={'order_id': 'demand'}).reset_index()

        # Sortowanie i ciągłość
        weekly_data = weekly_data.sort_values(['product_id', 'order_purchase_timestamp'])

        # Uzupełnianie
        weekly_data['demand'] = weekly_data['demand'].fillna(0)
        weekly_data['price'] = weekly_data.groupby('product_id')['price'].ffill().bfill()
        weekly_data['freight_value'] = weekly_data.groupby('product_id')['freight_value'].ffill().bfill()
        weekly_data['product_photos_qty'] = weekly_data['product_photos_qty'].fillna(1)
        weekly_data['product_weight_g'] = weekly_data['product_weight_g'].fillna(500)

        # --- FEATURE ENGINEERING ---

        # 1. Cechy czasowe
        weekly_data['month'] = weekly_data['order_purchase_timestamp'].dt.month
        weekly_data['day'] = weekly_data['order_purchase_timestamp'].dt.day
        weekly_data['is_black_friday'] = ((weekly_data['month'] == 11) & (weekly_data['day'] > 23)).astype(int)

        weekly_data['month_sin'] = np.sin(2 * np.pi * weekly_data['month'] / 12)
        weekly_data['month_cos'] = np.cos(2 * np.pi * weekly_data['month'] / 12)

        # 2. Advanced Lag Features (Momentum)
        # Lag 1 (sprzedaż tydzień temu)
        weekly_data['demand_lag_1'] = weekly_data.groupby('product_id')['demand'].shift(1)
        # Lag 2 (sprzedaż 2 tygodnie temu)
        weekly_data['demand_lag_2'] = weekly_data.groupby('product_id')['demand'].shift(2)

        # Delta Popytu: Czy sprzedaż rośnie? (Lag1 - Lag2)
        weekly_data['demand_delta'] = weekly_data['demand_lag_1'] - weekly_data['demand_lag_2']

        # Delta Ceny: Czy cena wzrosła względem poprzedniego tygodnia?
        # Shift ceny, żeby policzyć różnicę
        weekly_data['price_lag_1'] = weekly_data.groupby('product_id')['price'].shift(1)
        weekly_data['price_delta'] = weekly_data['price'] - weekly_data['price_lag_1']
        # Uzupełniamy brak pierwszej delty zerem
        weekly_data['price_delta'] = weekly_data['price_delta'].fillna(0)

        # Średnia krocząca
        weekly_data['demand_roll_mean_4w'] = weekly_data.groupby('product_id')['demand'].transform(
            lambda x: x.shift(1).rolling(4, min_periods=1).mean()
        )

        weekly_data = weekly_data.dropna(subset=['demand_lag_1', 'demand_lag_2', 'demand_roll_mean_4w'])

        # 3. Ceny i Konkurencja
        mean_prices = weekly_data.groupby('product_category_name')['price'].mean()
        self.product_base_prices = mean_prices.to_dict()
        weekly_data['base_price'] = weekly_data['product_category_name'].map(self.product_base_prices)
        weekly_data['price_normalized'] = weekly_data['price'] / weekly_data['base_price']

        cat_pop = weekly_data.groupby('product_category_name')['demand'].mean().to_dict()
        weekly_data['category_popularity'] = weekly_data['product_category_name'].map(cat_pop)

        cat_price = weekly_data.groupby(['product_category_name', 'order_purchase_timestamp'])[
            'price'].mean().reset_index()
        cat_price.rename(columns={'price': 'competitor_price'}, inplace=True)
        final_df = pd.merge(weekly_data, cat_price, on=['product_category_name', 'order_purchase_timestamp'],
                            how='left')
        final_df['price_ratio'] = final_df['price'] / final_df['competitor_price']

        final_df['rolling_price'] = final_df.groupby('product_id')['price'].transform(
            lambda x: x.rolling(4, min_periods=1).mean())
        final_df['price_vs_avg_4w'] = final_df['price'] / final_df['rolling_price']

        self.df = final_df.dropna().reset_index(drop=True)

        self.df['category_encoded'] = self.cat_encoder.fit_transform(self.df['product_category_name'])
        self.num_categories = len(self.cat_encoder.classes_)

        # Zapisujemy stan do symulacji
        last_entries = self.df.sort_values('order_purchase_timestamp').groupby('product_id').last()

        # Przygotowanie danych "na przyszłość"
        # demand (obecny) stanie się demand_lag_1
        # demand_lag_1 (obecny) stanie się demand_lag_2
        # cena (obecna) stanie się price_lag_1
        last_entries['next_demand_lag_1'] = last_entries['demand']
        last_entries['next_demand_lag_2'] = last_entries['demand_lag_1']
        last_entries['last_known_price'] = last_entries['price']

        self.product_recent_stats = last_entries[[
            'price', 'rolling_price', 'freight_value',
            'product_photos_qty', 'product_weight_g',
            'category_popularity', 'category_encoded',
            'next_demand_lag_1', 'next_demand_lag_2', 'demand_roll_mean_4w',
            'last_known_price'
        ]].to_dict('index')

        return self.df

    def train(self, epochs=250):
        if self.df.empty:
            raise ValueError("Brak danych.")

        X_num = self.df[self.num_features].values
        X_cat = self.df['category_encoded'].values
        Y = self.df[['demand']].values
        Y_log = np.log1p(Y)

        X_num_scaled = self.scaler_x.fit_transform(X_num)
        Y_scaled = self.scaler_y.fit_transform(Y_log)

        indices = np.arange(len(X_num))
        X_num_train, X_num_test, X_cat_train, X_cat_test, y_train, y_test, _, _ = train_test_split(
            X_num_scaled, X_cat, Y_scaled, indices, test_size=0.2, random_state=42
        )

        # --- MODEL ARCHITECTURE ---
        input_num = keras.Input(shape=(X_num_scaled.shape[1],), name='numeric_input')
        x_num = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(input_num)
        x_num = layers.BatchNormalization()(x_num)
        x_num = layers.Dropout(0.3)(x_num)

        input_cat = keras.Input(shape=(1,), name='category_input')
        embedding_size = min(50, int(self.num_categories / 2) + 1)
        x_cat = layers.Embedding(input_dim=self.num_categories, output_dim=embedding_size)(input_cat)
        x_cat = layers.Flatten()(x_cat)

        combined = layers.Concatenate()([x_num, x_cat])

        # Szersza i głębsza sieć
        z = layers.Dense(256, activation='relu')(combined)
        z = layers.Dropout(0.2)(z)
        z = layers.Dense(128, activation='relu')(z)
        z = layers.Dense(64, activation='relu')(z)
        output = layers.Dense(1, activation='linear')(z)

        self.model = keras.Model(inputs=[input_num, input_cat], outputs=output)

        # ZMIANA: Huber Loss zamiast MSE (odporność na outliery)
        loss_fn = keras.losses.Huber(delta=1.0)

        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                           loss=loss_fn, metrics=['mae'])

        # ZMIANA: ReduceLROnPlateau - zmniejszamy learning rate gdy przestajemy się uczyć
        lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        early_stop = callbacks.EarlyStopping(patience=15, monitor='val_loss', restore_best_weights=True)

        self.model.fit([X_num_train, X_cat_train], y_train,
                       epochs=epochs, batch_size=32, verbose=0,
                       validation_data=([X_num_test, X_cat_test], y_test),
                       callbacks=[early_stop, lr_scheduler])

        # Ewaluacja
        y_pred_scaled = self.model.predict([X_num_test, X_cat_test], verbose=0)
        y_pred_real = np.expm1(self.scaler_y.inverse_transform(y_pred_scaled))
        y_test_real = np.expm1(self.scaler_y.inverse_transform(y_test))
        y_pred_real = np.maximum(0, y_pred_real)

        self.metrics = {
            'mae': mean_absolute_error(y_test_real, y_pred_real),
            'mse': mean_squared_error(y_test_real, y_pred_real),
            'rmse': np.sqrt(mean_squared_error(y_test_real, y_pred_real)),
            'r2': r2_score(y_test_real, y_pred_real)
        }
        print(f"Metryki: {self.metrics}")

    def save_model(self, folder='model_store/'):
        if not os.path.exists(folder): os.makedirs(folder)
        if self.model: self.model.save(os.path.join(folder, 'my_model.keras'))

        artifacts = {
            'scaler_x': self.scaler_x,
            'scaler_y': self.scaler_y,
            'cat_encoder': self.cat_encoder,
            'base_prices': self.product_base_prices,
            'df': self.df,
            'metrics': self.metrics,
            'recent_stats': self.product_recent_stats,
            'num_categories': self.num_categories
        }
        joblib.dump(artifacts, os.path.join(folder, 'artifacts.pkl'))
        print(f"Model zapisany.")

    def load_saved_model(self, folder='model_store/'):
        if not os.path.exists(os.path.join(folder, 'my_model.keras')): return False

        self.model = keras.models.load_model(folder + 'my_model.keras')
        artifacts = joblib.load(folder + 'artifacts.pkl')

        self.scaler_x = artifacts['scaler_x']
        self.scaler_y = artifacts['scaler_y']
        self.cat_encoder = artifacts['cat_encoder']
        self.product_base_prices = artifacts['base_prices']
        self.df = artifacts['df']
        self.metrics = artifacts.get('metrics', {})
        self.product_recent_stats = artifacts.get('recent_stats', {})
        self.num_categories = artifacts.get('num_categories', 10)
        return True

    def predict_demand(self, price, competitor_price, category_name=None, product_id=None):
        if self.model is None: raise ValueError("Model niegotowy.")

        stats = self.product_recent_stats.get(product_id, {})
        if not stats:
            stats = self.df[self.df['product_category_name'] == category_name].mean(numeric_only=True).to_dict()
            try:
                cat_encoded = self.cat_encoder.transform([category_name])[0]
            except:
                cat_encoded = 0
                # Domyślne wartości dla nowych produktów
            demand_lag_1 = 0
            demand_lag_2 = 0
            last_known_price = price
        else:
            cat_encoded = stats['category_encoded']
            # Przesuwamy historię o 1 tydzień do przodu (bo przewidujemy przyszłość)
            demand_lag_1 = stats.get('next_demand_lag_1', 0)
            demand_lag_2 = stats.get('next_demand_lag_2', 0)
            last_known_price = stats.get('last_known_price', price)

        freight = stats.get('freight_value', 20.0)
        photos = stats.get('product_photos_qty', 1.0)
        weight = stats.get('product_weight_g', 500.0)
        cat_pop = stats.get('category_popularity', 1.0)
        rolling_price = stats.get('rolling_price', price)
        demand_roll_mean = stats.get('demand_roll_mean_4w', 1.0)

        # Obliczamy delty na bieżąco
        demand_delta = demand_lag_1 - demand_lag_2
        price_delta = price - last_known_price  # Różnica między nową symulowaną ceną a starą

        current_month = 6
        is_black_friday = 0
        month_sin = np.sin(2 * np.pi * current_month / 12)
        month_cos = np.cos(2 * np.pi * current_month / 12)

        base_price = self.product_base_prices.get(category_name, price)
        if base_price == 0: base_price = 1

        input_features = np.array([[
            price / base_price,
            price / competitor_price,
            price / rolling_price,
            price_delta,  # Nowa cecha
            month_sin,
            month_cos,
            is_black_friday,
            freight,
            photos,
            weight,
            cat_pop,
            demand_lag_1,
            demand_lag_2,  # Nowa cecha
            demand_delta,  # Nowa cecha
            demand_roll_mean
        ]])

        input_num_scaled = self.scaler_x.transform(input_features)
        input_cat = np.array([cat_encoded])

        pred_scaled = self.model.predict([input_num_scaled, input_cat], verbose=0)
        pred_demand = np.expm1(self.scaler_y.inverse_transform(pred_scaled)[0][0])
        return max(0, pred_demand)

    def get_product_details(self, product_id):
        product_rows = self.df[self.df['product_id'] == product_id]
        if product_rows.empty: return {'category': 'unknown', 'price': 0, 'competitor_price': 0, 'current_demand': 0}
        product_row = product_rows.iloc[-1]
        return {
            'category': product_row['product_category_name'],
            'price': float(product_row['price']),
            'competitor_price': float(product_row['competitor_price']),
            'current_demand': float(product_row['demand'])
        }

    def get_product_list(self):
        return self.df['product_id'].unique()