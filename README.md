# Industrial Heat Transfer Prediction System

This project is an application that uses machine learning to predict heat transfer in industrial systems. It leverages a LSTM-based neural network to analyze time-series data and provide predictions, safety monitoring, and system efficiency insights.

## Features

* **Data Processing:** Loads, preprocesses, and engineers features from uploaded CSV files.
* **LSTM-based Prediction:** Predicts target variables (e.g., heat flux) using a Bidirectional LSTM model.
* **Safety Monitoring:** Detects and reports potential safety warnings based on defined thresholds.
* **Anomaly Detection:** Identifies anomalies in the input data using Z-score analysis.
* **Performance Evaluation:** Evaluates the model's predictions using metrics like MAE, RMSE, and MAPE.
* **Industrial Reporting:** Generates a comprehensive report with safety metrics, efficiency metrics, and trend analysis.
* **Streamlit Interface:** Provides an interactive web interface for uploading data, training the model, and visualizing results.
* **Demo Mode:** Includes a demo mode with synthetic data generation for testing the application without uploading data.

## Dependencies

The project relies on the following Python libraries:

* `streamlit`: For creating the web application.
* `pandas`: For data manipulation and analysis.
* `numpy`: For numerical computations.
* `tensorflow`: For building and training the LSTM model.
* `matplotlib`: For generating basic plots.
* `seaborn`: For enhanced data visualizations.
* `scikit-learn`: For data scaling and time series splitting.
* `scipy`: For statistical functions (Z-score calculation).
* `plotly`: For creating interactive plots.

## Installation

1.  Clone the repository to your local machine.
2.  Navigate to the project directory.
3.  It is highly recommended to create a virtual environment to manage dependencies. If you are using venv, you can create a virtual environment using the following command:

    ```bash
    python -m venv venv
    ```

4.  Activate the virtual environment. On Windows, you can activate it using:

    ```bash
    venv\Scripts\activate
    ```

    On macOS and Linux, use:

    ```bash
    source venv/bin/activate
    ```

5.  Install the required dependencies using pip:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  Run the Streamlit application:

    ```bash
    streamlit run app.py
    ```

2.  The application will open in your web browser.
3.  Upload your CSV file containing the time-series data.
4.  Configure the parameters in the sidebar (sequence length, target column, training epochs, validation split).
5.  Click the "Train Model" button to train the LSTM model.
6.  The application will display the training history, predictions, performance metrics, and the industrial system analysis report.
7.  You can also use the "Run Demo with Sample Data" button to test the application with synthetic data.

## Data Format

The application expects a CSV file with the following columns:

* `timestamp`:  Datetime values representing the time of the measurements.
* `inlet_temperature`: Inlet temperature values.
* `outlet_temperature`: Outlet temperature values.
* `flow_rate`: Flow rate values.
* `pressure`: Pressure values.
* `heat_transfer_rate`: Heat transfer rate values.
* `heat_flux`: Heat flux values.
* `surface_area`: Surface area values.
* `reynolds_number`: Reynolds number values.

## Demo Data

The `data_generator.py` script is provided to generate synthetic data for demonstration purposes.  You can run this script to create a sample `heat_transfer_data.csv` file.  However, the Streamlit app also has a built-in demo mode.

## Important Considerations

* **Data Quality:** The accuracy of the predictions depends heavily on the quality and representativeness of the input data.
* **Computational Resources:** Training the LSTM model may require significant computational resources, especially for large datasets.
* **Parameter Tuning:** The model's performance can be further improved by tuning the hyperparameters (e.g., LSTM layers, dropout rate, learning rate).
* **Error Handling:** The application includes error handling for common issues, such as incorrect data format, but it's important to validate your data before uploading.