import streamlit as st
import pickle as pk
import pandas as pd
import pmdarima as pm
import matplotlib.pyplot as plt


def prepare(df):
    df.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis='columns', inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df.set_index('Date')
    df = df.dropna()
    return df


def arima(firm, regressor, day, period):
    model = regressor.predict(n_periods=day, return_conf_int=True)
    df = pd.DataFrame(model[0], index=period, columns=['Prediction'])
    lower, upper = pd.Series(model[1][:, 0], index=period), pd.Series(model[1][:, 1], index=period)
    plt.plot(firm['Close'])
    plt.plot(df)
    plt.fill_between(period, lower, upper, color='k', alpha=.15)


tcr, nestlr, ultrr = pk.load(open('tcs.pkl', 'rb')), pk.load(open('nestle.pkl', 'rb')), pk.load(open('ultra.pkl', 'rb'))
tcs, nestle, ultra = prepare(pd.read_csv('TCS.csv')), prepare(pd.read_csv('NESTLE.csv')), prepare(pd.read_csv('ULTRA.csv'))

st.write('Indraneel Dey')
st.write('Indian Institute of Technology, Madras')
st.title('Stock Price Forecasting')
st.write('Select the company whose closing stock price you wish to forecast')
company = st.selectbox('Select the company', ['TCS', 'Nestle', 'Ultratech'])
st.write('Enter the number of days after 12 August 2022 for which you wish to forecast')
days = st.number_input('Days', min_value=10)
forecast_range = pd.date_range(start='2022-08-13', periods=days, freq='D')
if st.button('Forecast'):
    if company == 'TCS':
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(arima(tcs, tcr, days, forecast_range))
    if company == 'Nestle':
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(arima(nestle, nestlr, days, forecast_range))
    if company == 'Ultratech':
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(arima(ultra, ultrr, days, forecast_range))
