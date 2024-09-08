import streamlit as st
import pandas as pd
from prophet import Prophet
from datetime import datetime

# Add title and in-depth description
st.title("Nepal Stock Prediction")

st.write("""
### About the Model
Our tool leverages the power of **Prophet**, a robust time series forecasting model developed by Facebook. Prophet is particularly effective for stock price predictions due to its ability to handle seasonal trends, irregular spikes, and long-term patterns in data. Here’s how it works:

1. **Data Preprocessing**: We take historical stock data from the CSV file you upload, which includes key features like Date, Open, High, Low, Close, Percent Change, and Volume.
2. **Model Training**: The Prophet model is trained using the past stock prices, particularly focusing on the **Close** prices to learn historical trends.
3. **Forecasting Future Prices**: The model forecasts stock prices for the upcoming 365 days based on the trends learned from the historical data.
4. **Recommendations**: Based on the forecasted prices, we suggest the best dates to sell your shares, considering a stepwise increase (e.g., 10, 20, 30 units, and so on).
5. **Data Filters**: We filter out the forecast data to recommend only dates that are beyond your purchase date and today, and where the predicted price exceeds the specified thresholds.
""")

# Add link to download stock CSV data
st.write("### Download Stock Data")
st.write("You can download the stock data CSV file from [Nepal Stock Exchange Alpha](https://nepsealpha.com/nepse-data).")

# Step 1: Upload CSV File
uploaded_file = st.file_uploader("Upload your stock data CSV file", type=["csv"])

if uploaded_file is not None:
    # Step 2: Read CSV file (hide the display of raw data)
    df = pd.read_csv(uploaded_file)

    # Ask for inputs: Buying price and purchase date
    buying_price = st.number_input("Enter the price at which you bought the share:", min_value=0.0)

    # Restrict buying date input to past dates
    today = datetime.today().date()
    purchase_date = st.date_input("Enter the date when you purchased the share:", max_value=today)

    # Ensure the 'Date' column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Step 3: Prophet data preparation
    df_prophet = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

    # Display 'Recommend' button only if both inputs are provided
    if st.button("Recommend"):
        # Step 4: Train Prophet model
        model = Prophet()
        model.fit(df_prophet)

        # Step 5: Forecast future prices for 365 days
        future = model.make_future_dataframe(periods=365)
        forecast = model.predict(future)

        # Step 6: Filter forecast for dates greater than both the purchase date and today
        future_filtered = forecast[(forecast['ds'] > pd.to_datetime(purchase_date)) & (forecast['ds'] > pd.to_datetime(today))]

        # Step 7: Create a list of price thresholds
        price_thresholds = [buying_price + i for i in range(10, 201, 10)]  # +10, +20, +30,..., +200

        # Step 8: Recommend dates for each threshold
        recommended_dates = []
        for threshold in price_thresholds:
            # Find the first date where the predicted price exceeds the threshold
            threshold_dates = future_filtered[future_filtered['yhat'] > threshold]
            if not threshold_dates.empty:
                first_date = threshold_dates.iloc[0]  # Get the first date where the price exceeds the threshold
                recommended_dates.append((first_date['ds'], first_date['yhat']))

        # Step 9: Display recommendation or no-profit message
        if recommended_dates:
            st.write("### Recommended Dates to Sell:")

            # Convert recommended dates to DataFrame for better display
            df_recommend = pd.DataFrame(recommended_dates, columns=['Date', 'Predicted Price'])

            # Format date and style the table for better UI
            df_recommend['Date'] = df_recommend['Date'].dt.strftime('%Y-%m-%d')
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
