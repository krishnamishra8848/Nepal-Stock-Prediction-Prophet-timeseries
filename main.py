import streamlit as st
import pandas as pd
from prophet import Prophet
from datetime import datetime

# Add title and description
st.title("Nepal Stock Prediction")

# Link to download stock CSV data
st.write("### Download Stock Data")
st.write("You can download the stock data CSV file from [Nepal Stock Exchange Alpha](https://nepsealpha.com/nepse-data).")

# Step 1: Upload CSV file
uploaded_file = st.file_uploader("Upload your stock data CSV file", type=["csv"])

if uploaded_file is not None:
    # Step 2: Read CSV file
    df = pd.read_csv(uploaded_file)

    # Check if 'Date' and 'Close' columns are present
    if 'Date' not in df.columns or 'Close' not in df.columns:
        st.warning("The CSV file must contain 'Date' and 'Close' columns.")
    else:
        # Ensure the 'Date' column is in datetime format
        df['Date'] = pd.to_datetime(df['Date']).dt.date

        # Sort data by Date
        df = df.sort_values(by='Date')

        # Get the last 7 days of actual data
        actual_last_7_days = df.tail(7)

        # Ask for inputs: Buying price and purchase date
        buying_price = st.number_input("Enter the price at which you bought the share:", min_value=0.0)

        # Restrict buying date input to past dates
        today = datetime.today().date()
        purchase_date = st.date_input("Enter the date when you purchased the share:", max_value=today)

        # Step 3: Prophet data preparation
        df_prophet = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

        # Prophet model training
        model = Prophet()
        model.fit(df_prophet)

        # Predict future and past prices
        future = model.make_future_dataframe(periods=365, include_history=True)  # Include past predictions
        forecast = model.predict(future)

        # Convert 'ds' column in forecast to date only (without time)
        forecast['ds'] = forecast['ds'].dt.date

        # Step 6: Filter forecast for past 7 days
        forecast_last_7_days = forecast[forecast['ds'].isin(actual_last_7_days['Date'])]

        # Merge actual and predicted close prices for the last 7 days
        last_7_days_comparison = pd.merge(actual_last_7_days[['Date', 'Close']], 
                                          forecast_last_7_days[['ds', 'yhat']], 
                                          left_on='Date', right_on='ds').drop(columns=['ds'])

        # Rename columns for clarity
        last_7_days_comparison.rename(columns={'Close': 'Actual', 'yhat': 'Predicted'}, inplace=True)

        # Calculate the difference between actual and predicted close prices
        last_7_days_comparison['Difference'] = last_7_days_comparison['Actual'] - last_7_days_comparison['Predicted']

        # Display the comparison table
        st.write("### Last 7 Days - Actual vs Predicted Close Prices")
        st.dataframe(last_7_days_comparison.style.format({"Actual": "{:.2f}", "Predicted": "{:.2f}", "Difference": "{:.2f}"}))

        # Calculate and display the average prediction error
        avg_error = abs(last_7_days_comparison['Difference'].mean())
        st.write(f"On average, the predicted price was off by: **{avg_error:.2f}**")

        # Display model performance message
        if avg_error <= 15:
            st.info("Our model predicts that future prices may vary within this margin.")
        else:
            st.warning("Our model can't understand this stock well. We don't recommend relying on this prediction.")

        # Calculate the average positive and negative deviation
        positive_deviation = last_7_days_comparison[last_7_days_comparison['Difference'] > 0]['Difference'].mean()
        negative_deviation = last_7_days_comparison[last_7_days_comparison['Difference'] < 0]['Difference'].mean()

        # Display 'Recommend' button only if both inputs are provided
        if st.button("Recommend"):
            # Step 4: Train Prophet model
            model = Prophet()
            model.fit(df_prophet)

            # Step 5: Forecast future prices for 365 days
            future = model.make_future_dataframe(periods=365)
            forecast = model.predict(future)

            # Add both positive and negative deviations to forecasted prices
            forecast['Positive Deviation'] = forecast['yhat'] + positive_deviation
            forecast['Negative Deviation'] = forecast['yhat'] - abs(negative_deviation)

            # Convert 'ds' column in forecast to date only (without time)
            forecast['ds'] = forecast['ds'].dt.date

            # Step 6: Filter forecast for dates greater than both the purchase date and today
            future_filtered = forecast[(forecast['ds'] > pd.to_datetime(purchase_date).date()) & (forecast['ds'] > pd.to_datetime(today).date())]

            # Step 7: Create a list of price thresholds
            price_thresholds = [buying_price + i for i in range(10, 201, 10)]  # +10, +20, +30,..., +200

            # Step 8: Recommend dates for each threshold
            recommended_dates = []
            for threshold in price_thresholds:
                # Find the first date where either positive or negative adjusted price exceeds the threshold
                threshold_dates = future_filtered[(future_filtered['Positive Deviation'] > threshold) | (future_filtered['Negative Deviation'] > threshold)]
                if not threshold_dates.empty:
                    first_date = threshold_dates.iloc[0]  # Get the first date where the price exceeds the threshold
                    recommended_dates.append((first_date['ds'], first_date['yhat']))

            # Step 9: Display recommendation or no-profit message
            if recommended_dates:
                st.write("### Recommended Dates to Sell:")

                # Convert recommended dates to DataFrame for better display
                df_recommend = pd.DataFrame(recommended_dates, columns=['Date', 'Predicted Price'])

                # Format date and style the table for better UI
                df_recommend['Date'] = df_recommend['Date'].astype(str)  # Convert date to string format
                st.dataframe(df_recommend.style.format({"Predicted Price": "{:.2f}"}).set_properties(**{
                    'background-color': '#f0f0f0',  # Light gray background
                    'font-size': '14px',
                    'color': '#333',  # Dark text
                    'border-color': '#ccc',  # Light border
                    'text-align': 'center'
                }))
            else:
                st.warning("No future dates found where the price exceeds your target prices.")

# Disclaimer (in red color)
st.markdown("<p style='color:red;'>Disclaimer: The stock predictions provided by this tool are based on historical data and statistical modeling. Actual market prices may vary significantly. Use this tool for informational purposes only and consult with a financial expert before making any investment decisions.</p>", unsafe_allow_html=True)
