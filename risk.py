#libraries
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import datetime
import yfinance as yf


#dataset loading
axis = pd.read_csv(r"C:\Users\viole\stockenv\axis.csv")
hdfc = pd.read_csv(r"C:\Users\viole\stockenv\hdfc.csv")
icici = pd.read_csv(r"C:\Users\viole\stockenv\icici.csv")
indus = pd.read_csv(r"C:\Users\viole\stockenv\indus.csv")
kotak = pd.read_csv(r"C:\Users\viole\stockenv\kotak.csv")
sbi = pd.read_csv(r"C:\Users\viole\stockenv\sbi.csv")

#ticker dicts
options = {
    'Axis Bank (AXISBANK.NS)': 'AXISBANK.NS',
    'IndusInd Bank (INDUSINDBK.NS)': 'INDUSINDBK.NS',
    'Kotak Bank (KOTAKBANK.NS)': 'KOTAKBANK.NS',
    'HDFC Bank (HDFCBANK.NS)': 'HDFCBANK.NS',
    'ICICI Bank (ICICIBANK.NS)': 'ICICIBANK.NS',
    'State Bank of India (SBIN.NS)': 'SBIN.NS'
}
tickdfs = {
    'AXISBANK.NS': axis,
    'INDUSINDBK.NS': indus,
    'KOTAKBANK.NS': kotak,
    'HDFCBANK.NS': hdfc,
    'ICICIBANK.NS': icici,
    'SBIN.NS': sbi
}
tickimg = {
    'AXISBANK.NS': "axists.png",
    'INDUSINDBK.NS': "industs.png",
    'KOTAKBANK.NS': "kotakts.png",
    'HDFCBANK.NS': "hdfcts.png",
    'ICICIBANK.NS': "icicits.png",
    'SBIN.NS': "sbits.png"
}


#driver functions
def forecast(ticker):
    st.markdown("<h3>Stock Price Forecasting</h3>", unsafe_allow_html=True)
    TODAY = datetime.datetime.today().date()
    Y1 = TODAY - datetime.timedelta(days=365)

    data = yf.download(ticker, start=Y1, end=TODAY)
    
    data['Previous_Close'] = data['Close'].shift(1)
    data = data.dropna()
    
    X = data[['Previous_Close']].values
    y = data['Close'].values
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y.reshape(-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, shuffle=False)
    model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    model.fit(X_train, y_train.ravel())
    future_days = 30
    future_dates = pd.date_range(start=data.index[-1] + pd.DateOffset(1), periods=future_days)
    future_predictions = []
    last_value = X_test[-1]

    for _ in range(future_days):
        next_pred = model.predict(last_value.reshape(1, -1))
        future_predictions.append(next_pred[0])
        last_value = np.append(last_value[1:], next_pred)
    
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()
    data.index = pd.to_datetime(data.index)    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Original Data',line=dict(color='#f1dac4')))
    fig.add_trace(go.Scatter(x=future_dates, y=future_predictions, mode='lines', name='Future Predictions',line=dict(color='#a69cac')))
    fig.update_layout(
        title='Forecast',
        xaxis_title='Date',
        yaxis_title='Closing Price',
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label='1m', step='month', stepmode='backward'),
                    dict(count=6, label='6m', step='month', stepmode='backward'),
                    dict(count=1, label='YTD', step='year', stepmode='todate'),
                    dict(count=1, label='1y', step='year', stepmode='backward'),
                    dict(step='all')
                ])
            ),
            rangeslider=dict(visible=True),
            type='date'
        ),
        paper_bgcolor='#161b33',  # Using a hex code for light blue background
        plot_bgcolor='#161b33'
    )
    
    st.plotly_chart(fig)

def timeseries(ticker):
    st.markdown("<h3>Historical Time Series Analysis using LSTM</h3>", unsafe_allow_html=True)
    pt = tickimg[ticker]
    st.image(pt, caption=ticker, use_column_width=True)
    st.markdown("The following time series graph highlights the highs and lows of the bank during the duration of 2015-2021")
  
def marketrisk(stock_ticker, market_ticker='^NSEI', start_date='2021-01-01', end_date='2024-01-01'):
    st.markdown("<h3>Risk Assessment</h3>", unsafe_allow_html=True)
    # Download historical price data for the stock and market index
    stock_data = yf.download(stock_ticker, start=start_date, end=end_date)
    market_data = yf.download(market_ticker, start=start_date, end=end_date)

    # Calculate daily returns
    stock_data['Returns'] = stock_data['Adj Close'].pct_change()
    market_data['Returns'] = market_data['Adj Close'].pct_change()

    # Calculate annualized volatility
    volatility = stock_data['Returns'].std() * np.sqrt(252)

    # Drop NaN values and align data
    returns = pd.concat([stock_data['Returns'], market_data['Returns']], axis=1).dropna()
    returns.columns = ['Stock', 'Market']

    # Perform linear regression to calculate beta
    X = sm.add_constant(returns['Market'])
    model = sm.OLS(returns['Stock'], X).fit()
    beta = model.params[1]

    # Calculate average daily trading volume
    avg_daily_volume = stock_data['Volume'].mean()

    # Calculate standard deviation of daily returns
    stock_daily_volatility = stock_data['Returns'].std()

    # Display the values side by side with larger text
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Annualized Volatility", f"{volatility:.2%}", delta=None, delta_color="normal", help=None)
    col2.metric("Beta", f"{beta:.2f}", delta=None, delta_color="normal", help=None)
    col3.metric("Avg Daily Volume", f"{avg_daily_volume:.0f}", delta=None, delta_color="normal", help=None)
    col4.metric("Std Dev of Daily Returns", f"{stock_daily_volatility:.2%}", delta=None, delta_color="normal", help=None)

def predict(ticker):
    START = "2015-01-01"
    TODAY = datetime.date.today().strftime("%Y-%m-%d")
    
    # Download stock data
    data = yf.download(ticker, START, TODAY)
    
    # Prepare the data
    df_train = data[['Close']].copy()
    df_train.reset_index(inplace=True)
    df_train['Date'] = pd.to_datetime(df_train['Date'])
    df_train['Date_ordinal'] = df_train['Date'].map(pd.Timestamp.toordinal)

    if len(df_train) == 0:
        st.error('No valid data available for the selected stock. Please choose another stock or check your custom ticker.')
    else:
        X = df_train[['Date_ordinal']]
        y = df_train['Close']

        # Scaling the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit the model
        model = LinearRegression()
        model.fit(X_scaled, y)

        # Predict the next day
        next_day = (pd.to_datetime(TODAY) + datetime.timedelta(days=1)).to_pydatetime()
        next_day_ordinal = np.array([[next_day.toordinal()]])
        next_day_scaled = scaler.transform(next_day_ordinal)

        # Make prediction
        next_day_prediction = model.predict(next_day_scaled)

        # Create a DataFrame for the forecast
        forecast = pd.DataFrame({'Date': [next_day], 'Predicted Close': next_day_prediction})

        # Show and plot forecast
        st.subheader('Forecast data')
        st.write(forecast)


#streamlit app

st.markdown("""
    <style>
        .stApp {
            background-color: #161b33;
        }
        .stButton button {
            background-color: #f1dac4;
            color: #0d0c1d;
        }
    /* Customize the selectbox */
         .stSelectbox [data-baseweb="select"] {
        background-color: #161b33;  /* Light blue background */
        color: #ffffff;             /* Dark text color */
            /* Padding inside the selectbox */
        }
    
    /* Customize the dropdown options */
        .stSelectbox [data-baseweb="popover"] {
        background-color: #ffffff;  /* White background for options */
        color: #000000;             /* Black text for options */
        }
    </style>
""", unsafe_allow_html=True)
st.markdown(f"""
    <div style="text-align: center;">
        <h1>Bank Stock Forecasting and Risk Assessment</h1>
        <br>
        <p style="text-align: justify;">
            This application consists of multiple features related to stock prices and risk assessment:
        </p>
        <ul style="list-style-type: disc; text-align: justify;">
            <li>Stock price forecasting for banks for the next 30 days using <b>Support Vector Regressor</b>.</li>
            <li>Prediction of next day closing price using <b>Linear Regressor</b> and market risk values using <b>StatsModels</b>.</li>
            <li>Historical time series of closing prices using <b>LSTM</b>.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
#st.title('Bank Stock Forecasting and Risk Assessment')
#st.markdown("This application forecasts the stock price for banks for the next 30 days using Support Vector Regressor, predicts current closing price using Linear Regressor and market risk values using StatsModels, and presents a time series analysis of the closing price using LSTM")


selected_stock = st.selectbox('Select stock for prediction', list(options.keys()))
ticker = "ADANIENT.NS"#options[selected_stock]
if st.button('Forecast'):
    #closing price prediction
    predict(ticker)
    #price forecasting
    st.markdown(" ")
    st.markdown(" ")
    forecast(ticker)
    st.markdown("Above graph shows the forecasted closing price of the chosen stock upto one month from the current date.")
    #risk assessment
    st.markdown(" ")
    st.markdown(" ")
    marketrisk(ticker, 'NSE')
    #time series analysis
    st.markdown(" ")
    st.markdown(" ")
    timeseries(ticker)

