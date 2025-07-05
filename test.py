# app.py - Hybrid Layout Version
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta

# Import สำหรับ Machine Learning
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# --- ส่วนของการตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Stock Analysis & Prediction", layout="wide")
st.title("Stock Analysis & Prediction Dashboard 📈")
st.write("แอปพลิเคชันวิเคราะห์และทำนายแนวโน้มราคาหุ้น (เพื่อการศึกษาเท่านั้น)")

# --- ส่วนรับ Input (ย้ายมาไว้ที่หน้าหลัก) ---
st.divider()
st.header("1. เลือกหุ้นและช่วงเวลา")

col1, col2 = st.columns(2)
with col1:
    stock_symbol = st.text_input("กรอกชื่อย่อหุ้น (Ticker)", "NVDA").upper()
with col2:
    default_start_date = date.today() - timedelta(days=3*365)
    start_date = st.date_input("วันเริ่มต้นข้อมูล", default_start_date)

end_date = date.today()

# --- ฟังก์ชันหลักในการดึงข้อมูล (Cache เพื่อความเร็ว) ---
@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    return data

# โหลดข้อมูล
try:
    data = load_data(stock_symbol, start_date, end_date)
    if data.empty:
        st.error(f"ไม่สามารถดึงข้อมูลของหุ้น '{stock_symbol}' ได้ กรุณาตรวจสอบชื่อ Ticker และช่วงเวลา")
    else:
        st.success(f"โหลดข้อมูลหุ้น {stock_symbol} ตั้งแต่วันที่ {start_date} ถึง {end_date} สำเร็จ")
        
        # --- สร้าง Tabs สำหรับการแสดงผล (โค้ดส่วนนี้กลับมาเหมือนเดิม) ---
        tab1, tab2, tab3 = st.tabs(["📊 ข้อมูลและกราฟราคา", "🔬 วิเคราะห์ทางเทคนิค (Analysis)", "🚀 ทำนายอนาคต (Prediction)"])

        # ==================== Tab 1: ข้อมูลและกราฟราคา ====================
        with tab1:
            st.header(f"ข้อมูลราคาหุ้น {stock_symbol}")
            st.dataframe(data.tail(10))

            st.header("กราฟราคาปิด (Close Price)")
            fig_price = plt.figure(figsize=(12, 6))
            plt.plot(data['Close'])
            plt.title(f'Price History for {stock_symbol}', fontsize=16)
            plt.xlabel('Date')
            plt.ylabel('Price (USD)')
            plt.grid(True)
            st.pyplot(fig_price)

        # ==================== Tab 2: วิเคราะห์ทางเทคนิค ====================
        with tab2:
            st.header("Technical Analysis")
            
            # คำนวณ Indicators
            data['SMA50'] = data['Close'].rolling(window=50).mean()
            data['SMA200'] = data['Close'].rolling(window=200).mean()
            delta = data['Close'].diff(1)
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))

            latest_data = data.dropna().iloc[-1]

            st.subheader("สรุปสัญญาณล่าสุด")
            metric_col1, metric_col2 = st.columns(2)
            
            with metric_col1: # RSI
                rsi_value = latest_data['RSI']
                if rsi_value < 30:
                    st.metric(label="RSI", value=f"{rsi_value:.2f}", delta="Oversold (สัญญาณซื้อ)", delta_color="inverse")
                elif rsi_value > 70:
                    st.metric(label="RSI", value=f"{rsi_value:.2f}", delta="Overbought (สัญญาณขาย)", delta_color="inverse")
                else:
                    st.metric(label="RSI", value=f"{rsi_value:.2f}", delta="Neutral")
            
            with metric_col2: # SMA
                sma50_value = latest_data['SMA50']
                sma200_value = latest_data['SMA200']
                if sma50_value > sma200_value:
                    st.metric(label="SMA Trend", value="ขาขึ้น", delta="Golden Cross")
                else:
                    st.metric(label="SMA Trend", value="ขาลง", delta="Dead Cross", delta_color="inverse")

            st.subheader("กราฟราคาและ Indicators")
            fig_sma = plt.figure(figsize=(12, 6))
            plt.plot(data['Close'], label='Close Price')
            plt.plot(data['SMA50'], label='SMA 50 Days')
            plt.plot(data['SMA200'], label='SMA 200 Days')
            plt.title('Price with SMA', fontsize=16)
            plt.legend()
            st.pyplot(fig_sma)

            fig_rsi = plt.figure(figsize=(12, 4))
            plt.plot(data['RSI'], label='RSI')
            plt.axhline(70, color='red', linestyle='--')
            plt.axhline(30, color='green', linestyle='--')
            plt.title('Relative Strength Index (RSI)', fontsize=16)
            plt.legend()
            st.pyplot(fig_rsi)

        # ==================== Tab 3: ทำนายอนาคต ====================
        with tab3:
            st.header("Price Prediction using LSTM")

            if st.button("🚀 เริ่มการทำนายราคา (อาจใช้เวลาหลายนาที)"):
                with st.spinner("กำลังเตรียมข้อมูลและสอนโมเดล LSTM..."):
                    # เตรียมข้อมูลและสร้างโมเดล (โค้ดเหมือนเดิม)
                    close_data = data.filter(['Close'])
                    dataset = close_data.values
                    training_data_len = int(np.ceil(len(dataset) * .9))

                    scaler = MinMaxScaler(feature_range=(0,1))
                    scaled_data = scaler.fit_transform(dataset)

                    train_data = scaled_data[0:training_data_len, :]
                    x_train, y_train = [], []
                    for i in range(60, len(train_data)):
                        x_train.append(train_data[i-60:i, 0])
                        y_train.append(train_data[i, 0])
                    x_train, y_train = np.array(x_train), np.array(y_train)
                    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
                    
                    model = Sequential()
                    model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
                    model.add(LSTM(64, return_sequences=False))
                    model.add(Dense(25))
                    model.add(Dense(1))
                    model.compile(optimizer='adam', loss='mean_squared_error')
                    model.fit(x_train, y_train, batch_size=1, epochs=1)
                    st.success("สอนโมเดลสำเร็จ!")

                with st.spinner("กำลังทำนายราคา..."):
                    # ทำนายราคา (โค้ดเหมือนเดิม)
                    test_data = scaled_data[training_data_len - 60:, :]
                    x_test = []
                    for i in range(60, len(test_data)):
                        x_test.append(test_data[i-60:i, 0])
                    x_test = np.array(x_test)
                    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))
                    
                    predictions = model.predict(x_test)
                    predictions = scaler.inverse_transform(predictions)

                # แสดงผล (โค้ดเหมือนเดิม)
                st.subheader("ผลการทำนายเทียบกับราคาจริง")
                train = close_data[:training_data_len]
                valid = close_data[training_data_len:]
                valid['Predictions'] = predictions
                
                fig_pred = plt.figure(figsize=(16,8))
                plt.title('Model Prediction vs Actual Price', fontsize=18)
                plt.xlabel('Date', fontsize=18)
                plt.ylabel('Close Price USD ($)', fontsize=18)
                plt.plot(train['Close'], label='Train History')
                plt.plot(valid['Close'], label='Actual Price')
                plt.plot(valid['Predictions'], label='Predicted Price')
                plt.legend(loc='lower right')
                st.pyplot(fig_pred)
                
                # ทำนาย 1 วันข้างหน้า
                last_60_days = close_data[-60:].values
                last_60_days_scaled = scaler.transform(last_60_days)
                X_pred = []
                X_pred.append(last_60_days_scaled)
                X_pred = np.array(X_pred)
                X_pred = np.reshape(X_pred, (X_pred.shape[0], X_pred.shape[1], 1))
                pred_price_scaled = model.predict(X_pred)
                pred_price = scaler.inverse_transform(pred_price_scaled)
                
                st.subheader(f"ราคาที่ทำนายสำหรับวันทำการถัดไป:")
                st.success(f"## ${pred_price[0][0]:.2f}")

except Exception as e:
    st.error(f"เกิดข้อผิดพลาดในการทำงาน: {e}")