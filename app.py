import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats
import plotly.graph_objects as go  
import plotly.express as px
from io import StringIO


class IndustrialHeatTransferPredictor:
    def __init__(self, sequence_length=10, target_col='heat_flux'):
        self.sequence_length = sequence_length
        self.target_col = target_col
        self.scaler_X = RobustScaler()  # More robust to outliers
        self.scaler_y = RobustScaler()
        self.model = None
        self.feature_columns = None
        self.safety_bounds = {
            'max_temperature': 150,  # °C
            'max_pressure': 150,  # kPa
            'max_flow_rate': 5,  # m³/s
            'max_heat_flux': 1000  # W/m²
        }

    def engineer_features(self, df):
        """Advanced feature engineering for industrial heat transfer"""
        # Calculate time-based features
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek

        # Temperature differentials
        df['temp_differential'] = df['inlet_temperature'] - df['outlet_temperature']
        df['temp_rate_change'] = df['temp_differential'].diff()

        # Thermal effectiveness
        df['thermal_effectiveness'] = df['heat_transfer_rate'] / (df['flow_rate'] * 4186 * df['inlet_temperature'])

        # Nusselt number (simplified)
        df['nusselt_number'] = 0.023 * (df['reynolds_number'] ** 0.8) * (4.12 ** 0.4)

        # Pressure drop features
        df['pressure_gradient'] = df['pressure'].diff()

        # Energy efficiency metrics
        df['energy_efficiency_ratio'] = df['heat_transfer_rate'] / (df['flow_rate'] * df['pressure'])

        # Moving averages and standard deviations
        windows = [3, 5, 10]
        for window in windows:
            df[f'temp_ma_{window}'] = df['inlet_temperature'].rolling(window=window).mean()
            df[f'flow_ma_{window}'] = df['flow_rate'].rolling(window=window).mean()
            df[f'heat_flux_ma_{window}'] = df['heat_flux'].rolling(window=window).mean()
            df[f'temp_std_{window}'] = df['inlet_temperature'].rolling(window=window).std()

        # Fouling factor estimation (simplified)
        df['fouling_factor'] = 1 / (df['heat_transfer_rate'] / (df['temp_differential'] * df['surface_area']))

        # Fill NaN values created by differential and rolling calculations
        df = df.fillna(method='bfill')

        return df

    def add_safety_checks(self, df):
        """Add safety monitoring features"""
        df['temperature_warning'] = df['inlet_temperature'] > self.safety_bounds['max_temperature'] * 0.8
        df['pressure_warning'] = df['pressure'] > self.safety_bounds['max_pressure'] * 0.8
        df['flow_warning'] = df['flow_rate'] > self.safety_bounds['max_flow_rate'] * 0.8
        df['heat_flux_warning'] = df['heat_flux'] > self.safety_bounds['max_heat_flux'] * 0.8
        return df

    def detect_anomalies(self, df, columns, threshold=3):
        """Detect anomalies using Z-score method"""
        anomalies = {}
        for col in columns:
            z_scores = stats.zscore(df[col])
            anomalies[col] = abs(z_scores) > threshold
        return anomalies

    def load_and_preprocess_data(self, file):
        """Load and preprocess the heat transfer dataset with feature engineering"""
        # Read CSV file
        if type(file) == str:  # If file is a file path
            df = pd.read_csv(file)
        else:  # If file is an uploaded file
            content = file.getvalue().decode('utf-8')
            df = pd.read_csv(StringIO(content))

        # Engineer features
        df = self.engineer_features(df)

        # Add safety monitoring
        df = self.add_safety_checks(df)

        # Detect anomalies
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        anomalies = self.detect_anomalies(df, numerical_columns)

        # Remove timestamp and categorical columns for modeling
        modeling_columns = df.select_dtypes(include=[np.number]).columns
        self.feature_columns = [col for col in modeling_columns if col != self.target_col]

        X = df[self.feature_columns]
        y = df[[self.target_col]]

        # Scale the data
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)

        return X_scaled, y_scaled, df, anomalies

    def create_sequences(self, X, y):
        """Create sequences for time series prediction"""
        X_seq, y_seq = [], []
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length])
        return np.array(X_seq), np.array(y_seq)

    def build_model(self, input_shape):
        """Build advanced LSTM model architecture"""
        model = Sequential([
            Bidirectional(LSTM(256, return_sequences=True, kernel_regularizer=l2(0.01)),
                          input_shape=input_shape),
            Dropout(0.4),
            Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(0.01))),
            Dropout(0.4),
            Bidirectional(LSTM(64, kernel_regularizer=l2(0.01))),
            Dropout(0.3),
            Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
            Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
            Dense(1)
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='huber',  # More robust to outliers
            metrics=['mae', 'mse']
        )

        self.model = model
        return model

    def train_model(self, X, y, epochs=50, validation_split=0.2):
        """Train the model with the given data"""
        # Create sequences
        X_seq, y_seq = self.create_sequences(X, y)

        # Split data
        split_idx = int(len(X_seq) * (1 - validation_split))
        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]

        # Build model
        self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))

        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
        ]

        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )

        return history, X_seq, y_seq

    def predict(self, X_seq):
        """Make predictions using the trained model"""
        y_pred_scaled = self.model.predict(X_seq)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        return y_pred

    def evaluate_model(self, y_true, y_pred):
        """Evaluate model performance"""
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }

    def generate_industrial_report(self, df, predictions, actual_values):
        """Generate comprehensive industrial report"""
        report = {
            'model_performance': self.evaluate_model(actual_values, predictions),
            'safety_metrics': {
                'temperature_warnings': df['temperature_warning'].sum(),
                'pressure_warnings': df['pressure_warning'].sum(),
                'flow_warnings': df['flow_warning'].sum(),
                'heat_flux_warnings': df['heat_flux_warning'].sum()
            },
            'efficiency_metrics': {
                'avg_thermal_effectiveness': df['thermal_effectiveness'].mean(),
                'avg_energy_efficiency': df['energy_efficiency_ratio'].mean(),
                'fouling_factor_trend': self.calculate_trend(df['fouling_factor'])
            }
        }
        return report

    def calculate_trend(self, series):
        """Calculate simple linear trend of a time series"""
        x = np.arange(len(series))
        y = series.values
        coefficients = np.polyfit(x, y, 1)
        slope = coefficients[0]

        if slope > 0.01:
            return "Increasing (Potential Fouling Issue)"
        elif slope < -0.01:
            return "Decreasing (Improving Efficiency)"
        else:
            return "Stable"


# Streamlit UI
st.set_page_config(page_title="Industrial Heat Transfer Predictor", layout="wide")

st.title("Industrial Heat Transfer Prediction System")
st.markdown("""
This application uses advanced AI to predict heat flux in industrial systems.
Upload your sensor data to get predictions and system insights.
""")

# Sidebar
st.sidebar.header("Configuration")
sequence_length = st.sidebar.slider("Sequence Length", 5, 20, 10)
target_column = st.sidebar.selectbox("Target Column", ["heat_flux", "heat_transfer_rate", "outlet_temperature"])
epochs = st.sidebar.slider("Training Epochs", 10, 100, 50)
validation_split = st.sidebar.slider("Validation Split", 0.1, 0.3, 0.2)

# Initialize predictor
predictor = IndustrialHeatTransferPredictor(sequence_length=sequence_length, target_col=target_column)

# File uploader
st.subheader("1. Upload Time Series Data")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Display data sample
    with st.expander("Data Preview"):
        raw_data = pd.read_csv(uploaded_file)
        st.write(raw_data.head())

    # Process data
    with st.spinner("Processing data and engineering features..."):
        try:
            # Reset file pointer
            uploaded_file.seek(0)
            X_scaled, y_scaled, df, anomalies = predictor.load_and_preprocess_data(uploaded_file)
            st.success("Data preprocessing completed!")

            # Display engineered features
            with st.expander("Engineered Features Preview"):
                st.write(df.head())

            # Training section
            st.subheader("2. Train Model")
            train_button = st.button("Train Model")

            if train_button:
                with st.spinner("Training model... This may take a few minutes."):
                    history, X_seq, y_seq = predictor.train_model(X_scaled, y_scaled, epochs=epochs,
                                                                  validation_split=validation_split)
                    st.success("Model training completed!")

                    # Plot training history
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(history.history['loss'], label='Training Loss')
                    ax.plot(history.history['val_loss'], label='Validation Loss')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Loss')
                    ax.legend()
                    st.pyplot(fig)

                    # Predictions
                    st.subheader("3. Model Predictions")
                    with st.spinner("Generating predictions..."):
                        y_pred = predictor.predict(X_seq)
                        y_true = predictor.scaler_y.inverse_transform(y_seq)

                        # Plot predictions vs actual
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            y=y_true.flatten(),
                            mode='lines',
                            name='Actual',
                            line=dict(color='blue')
                        ))
                        fig.add_trace(go.Scatter(
                            y=y_pred.flatten(),
                            mode='lines',
                            name='Predicted',
                            line=dict(color='red')
                        ))
                        fig.update_layout(
                            title=f'Actual vs Predicted {target_column}',
                            xaxis_title='Time Steps',
                            yaxis_title=target_column,
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Performance metrics
                        metrics = predictor.evaluate_model(y_true, y_pred)

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("MAE", f"{metrics['mae']:.4f}")
                        with col2:
                            st.metric("RMSE", f"{metrics['rmse']:.4f}")
                        with col3:
                            st.metric("MAPE (%)", f"{metrics['mape']:.2f}")

                    # Industrial report
                    st.subheader("4. Industrial System Analysis")
                    with st.spinner("Generating system insights..."):
                        report = predictor.generate_industrial_report(df, y_pred, y_true)

                        # Safety metrics
                        st.write("### Safety Metrics")
                        safety_data = report['safety_metrics']
                        safety_fig = px.bar(
                            x=list(safety_data.keys()),
                            y=list(safety_data.values()),
                            labels={'x': 'Warning Type', 'y': 'Count'},
                            color=list(safety_data.values()),
                            color_continuous_scale='Reds'
                        )
                        safety_fig.update_layout(height=400)
                        st.plotly_chart(safety_fig, use_container_width=True)

                        # Efficiency metrics
                        st.write("### Efficiency Metrics")
                        eff_cols = st.columns(3)
                        with eff_cols[0]:
                            st.metric("Avg Thermal Effectiveness",
                                      f"{report['efficiency_metrics']['avg_thermal_effectiveness']:.4f}")
                        with eff_cols[1]:
                            st.metric("Avg Energy Efficiency",
                                      f"{report['efficiency_metrics']['avg_energy_efficiency']:.4f}")
                        with eff_cols[2]:
                            st.metric("Fouling Factor Trend", report['efficiency_metrics']['fouling_factor_trend'])

                        # Anomaly detection
                        st.write("### Anomaly Detection")
                        anomaly_counts = {col: sum(anomalies[col]) for col in anomalies}
                        top_anomalies = dict(sorted(anomaly_counts.items(), key=lambda x: x[1], reverse=True)[:10])

                        anomaly_fig = px.bar(
                            x=list(top_anomalies.keys()),
                            y=list(top_anomalies.values()),
                            labels={'x': 'Feature', 'y': 'Anomaly Count'},
                            color=list(top_anomalies.values()),
                            color_continuous_scale='Viridis'
                        )
                        anomaly_fig.update_layout(height=400)
                        st.plotly_chart(anomaly_fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error processing the data: {str(e)}")
            st.write("Please ensure your CSV file has the following required columns:")
            st.write("- timestamp (datetime)")
            st.write("- inlet_temperature")
            st.write("- outlet_temperature")
            st.write("- flow_rate")
            st.write("- pressure")
            st.write("- heat_transfer_rate")
            st.write("- heat_flux")
            st.write("- surface_area")
            st.write("- reynolds_number")
else:
    # Demo mode
    st.info("👆 Please upload a CSV file to get started, or use the demo mode below.")

    if st.button("Run Demo with Sample Data"):
        st.markdown("### Demo Mode")
        st.warning("This is running with synthetic data for demonstration purposes.")

        # Generate synthetic data
        with st.spinner("Generating synthetic data..."):
            np.random.seed(42)
            n_samples = 500

            # Create timestamps
            timestamps = pd.date_range(start='2023-01-01', periods=n_samples, freq='H')

            # Create base variables with some realistic patterns
            inlet_temp = 80 + 20 * np.sin(np.linspace(0, 10, n_samples)) + np.random.normal(0, 3, n_samples)
            flow_rate = 2 + 0.5 * np.sin(np.linspace(0, 15, n_samples)) + np.random.normal(0, 0.2, n_samples)
            pressure = 100 + 10 * np.sin(np.linspace(0, 8, n_samples)) + np.random.normal(0, 5, n_samples)

            # Calculate dependent variables
            temp_diff = 20 + 5 * np.sin(np.linspace(0, 12, n_samples)) + np.random.normal(0, 2, n_samples)
            outlet_temp = inlet_temp - temp_diff
            surface_area = 10 + np.random.normal(0, 0.1, n_samples)

            # Reynolds number based on flow
            reynolds_number = 20000 + 5000 * flow_rate + np.random.normal(0, 1000, n_samples)

            # Heat transfer calculations
            heat_transfer_rate = 4186 * flow_rate * temp_diff + np.random.normal(0, 1000, n_samples)
            heat_flux = heat_transfer_rate / surface_area + np.random.normal(0, 50, n_samples)

            # Create synthetic dataframe
            synthetic_data = pd.DataFrame({
                'timestamp': timestamps,
                'inlet_temperature': inlet_temp,
                'outlet_temperature': outlet_temp,
                'flow_rate': flow_rate,
                'pressure': pressure,
                'surface_area': surface_area,
                'reynolds_number': reynolds_number,
                'heat_transfer_rate': heat_transfer_rate,
                'heat_flux': heat_flux
            })

            # Save to StringIO to mimic file upload
            csv_buffer = StringIO()
            synthetic_data.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)

            # Process the data
            X_scaled, y_scaled, df, anomalies = predictor.load_and_preprocess_data(csv_buffer)
            st.success("Demo data generated and processed!")

            # Display data
            with st.expander("Synthetic Data Preview"):
                st.write(synthetic_data.head())

            # Training
            with st.spinner("Training demo model..."):
                history, X_seq, y_seq = predictor.train_model(X_scaled, y_scaled, epochs=30, validation_split=0.2)
                st.success("Demo model training completed!")

                # Plot training history
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(history.history['loss'], label='Training Loss')
                ax.plot(history.history['val_loss'], label='Validation Loss')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.legend()
                st.pyplot(fig)

                # Predictions
                st.subheader("Demo Predictions")
                y_pred = predictor.predict(X_seq)
                y_true = predictor.scaler_y.inverse_transform(y_seq)

                # Plot predictions vs actual
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=y_true.flatten(),
                    mode='lines',
                    name='Actual',
                    line=dict(color='blue')
                ))
                fig.add_trace(go.Scatter(
                    y=y_pred.flatten(),
                    mode='lines',
                    name='Predicted',
                    line=dict(color='red')
                ))
                fig.update_layout(
                    title=f'Actual vs Predicted {target_column}',
                    xaxis_title='Time Steps',
                    yaxis_title=target_column,
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)

                # Performance metrics
                metrics = predictor.evaluate_model(y_true, y_pred)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("MAE", f"{metrics['mae']:.4f}")
                with col2:
                    st.metric("RMSE", f"{metrics['rmse']:.4f}")
                with col3:
                    st.metric("MAPE (%)", f"{metrics['mape']:.2f}")

                # Industrial report
                st.subheader("Demo System Analysis")
                report = predictor.generate_industrial_report(df, y_pred, y_true)

                # Safety metrics
                st.write("### Safety Metrics")
                safety_data = report['safety_metrics']
                safety_fig = px.bar(
                    x=list(safety_data.keys()),
                    y=list(safety_data.values()),
                    labels={'x': 'Warning Type', 'y': 'Count'},
                    color=list(safety_data.values()),
                    color_continuous_scale='Reds'
                )
                safety_fig.update_layout(height=400)
                st.plotly_chart(safety_fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("### Industrial Heat Transfer Prediction System")
st.markdown(
    "This application uses advanced machine learning to predict and analyze heat transfer in industrial systems. The model uses bidirectional LSTM networks to capture complex patterns in time series data.")