import streamlit as st
from datetime import datetime
!pip install yfinance
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

# Input fields
start_date = st.text_input("Enter the start date (YYYY-MM-DD):", "")
end_date = st.text_input("Enter the end date (YYYY-MM-DD):", "")
stock_symbol = st.text_input("Enter the stock symbol:", "").upper()
n_years = st.slider("Future no. of years to consider for prediction", 1, 4)
period = n_years * 365

# Check for input errors
if not (start_date and end_date and stock_symbol):
    st.error("Please fill in all the required fields.")
else:
    try:
        datetime.strptime(start_date, "%Y-%m-%d")
        datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        st.error("Invalid date format. Please use the format YYYY-MM-DD.")
        st.stop()

    # Load data function
    def load_data(ticker):
        try:
            data = yf.download(ticker, start_date, end_date)
            if data.empty:
                st.error("Invalid stock symbol. Please enter a valid stock symbol.")
                return None
            data.reset_index(inplace=True)
            return data
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None

    # Loading data
    with st.spinner("Loading data..."):
        data = load_data(stock_symbol)

    # Display appropriate messages based on data
    if data is not None:
        st.text("Loading data...done!")

        # Filter data based on start and end dates
        filtered_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]

        # Display raw data
        st.subheader('Raw data')
        st.write(filtered_data.tail())

        # Display candlestick chart
        st.subheader('Real-time Candlestick Chart')
        fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
        fig.add_trace(go.Candlestick(x=filtered_data['Date'],
                                     open=filtered_data['Open'],
                                     high=filtered_data['High'],
                                     low=filtered_data['Low'],
                                     close=filtered_data['Close'],
                                     name='Candlestick Chart'),
                      row=1, col=1)
        fig.update_layout(xaxis_rangeslider_visible=False)
        st.plotly_chart(fig)



        # Chart with 100MA & 200MA
        st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
        ma100 = filtered_data.Close.rolling(100).mean()
        ma200 = filtered_data.Close.rolling(200).mean()
        fig_ma = plt.figure(figsize=(12, 6))
        plt.plot(ma100, 'r', label='100MA')
        plt.plot(ma200, 'g', label='200MA')
        plt.plot(filtered_data.Close, 'b', label='Closing Price')
        plt.legend()
        st.pyplot(fig_ma)

        # Forecasting
        df_train = data[['Date', 'Close']]
        df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

        m = Prophet()
        m.fit(df_train)
        future = m.make_future_dataframe(periods=period)
        forecast = m.predict(future)

        # Display forecast data
        st.subheader('Forecast Data')
        st.write(forecast.tail())

        # Display forecast chart
        st.subheader('Forecast Chart')
        fig1 = plot_plotly(m, forecast)
        fig1.update_layout(width=700)  # Adjust the value as needed
        st.plotly_chart(fig1)

        # Display forecast components
        st.subheader('Forecast Components')
        fig2 = m.plot_components(forecast)
        st.write(fig2)
