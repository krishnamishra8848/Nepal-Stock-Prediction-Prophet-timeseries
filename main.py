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
        no_of_kitta = st.number_input("Enter the number of shares (kitta):", min_value=1)

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

        # Ask the user to input the date they want to sell their shares
        selling_date = st.date_input("Enter the date you want to sell your share (must be within next 365 days):", min_value=today)

        # Step 7: Filter the forecast for the selling date
        forecast_selling_date = forecast[forecast['ds'] == selling_date]

        if not forecast_selling_date.empty:
            predicted_price = forecast_selling_date['yhat'].values[0]

            # Calculate profit or loss
            total_buying_price = buying_price * no_of_kitta
            total_selling_price = predicted_price * no_of_kitta
            profit_or_loss = total_selling_price - total_buying_price

            # Display the predicted price and profit/loss
            st.write(f"### Prediction for {selling_date}")

            # Display the predicted price in h1 style with color based on comparison
            if predicted_price > buying_price:
                st.markdown(f"<h1 style='color:green;'>Predicted price per share: {predicted_price:.2f}</h1>", unsafe_allow_html=True)
                st.success(f"Profit: **{profit_or_loss:.2f}**")
            else:
                st.markdown(f"<h1 style='color:red;'>Predicted price per share: {predicted_price:.2f}</h1>", unsafe_allow_html=True)

            # Step 8: Input the user's selling price demand
            selling_price_demand = st.number_input("Enter your desired selling price per share:", min_value=0.0)

            if selling_price_demand > 0:
                # Filter the forecast to find the dates when the demand price is met or exceeded within the next year
                demand_meeting_dates = forecast[(forecast['ds'] > today) & (forecast['ds'] <= today.replace(year=today.year + 1))]
                demand_meeting_dates = demand_meeting_dates[demand_meeting_dates['yhat'] >= selling_price_demand]

                if not demand_meeting_dates.empty:
                    # Display the dates when the selling price demand is met or exceeded
                    st.write("### Dates where your demand price is met or exceeded")
                    demand_meeting_dates_display = demand_meeting_dates[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Predicted Price'})
                    st.table(demand_meeting_dates_display.style.format({"Predicted Price": "{:.2f}"}))
                else:
                    st.warning("Within 1 year, your demand price is not met. Please consider adjusting your demand price.")
        else:
            st.warning("No prediction available for the selected selling date. Please choose a different date.")

# Disclaimer (in red color)
st.markdown("<p style='color:red;'>Disclaimer: The stock predictions provided by this tool are based on historical data and statistical modeling. Actual market prices may vary significantly. Use this tool for informational purposes only and consult with a financial expert before making any investment decisions.</p>", unsafe_allow_html=True)
