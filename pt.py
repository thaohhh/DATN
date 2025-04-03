import pandas as pd
import streamlit as st
from vnstock import stock_historical_data, listing_companies
from datetime import datetime, timedelta
import joblib

@st.cache_data
def get_stock_list():
    df = listing_companies()
    if df is not None and 'ticker' in df.columns:
        return df[['ticker', 'organName', 'sector', 'industry']]
    return pd.DataFrame()

@st.cache_data
def load_vnstock_data(symbol, start_date):
    df = None  # Khởi tạo biến trước để tránh lỗi UnboundLocalError
    try:
        df = stock_historical_data(
            symbol,
            start_date=start_date,
            end_date=datetime.today().strftime('%Y-%m-%d'),
            resolution='1D'
        )
        if df is not None and not df.empty:
            df['date'] = pd.to_datetime(df['time'])
            df.set_index('date', inplace=True)
            return df
    except Exception as e:
        st.warning(f"Lỗi khi tải dữ liệu {symbol}: {e}")
    return None  # Đảm bảo trả về None nếu có lỗi

def calculate_macd(data):
    short_ema = data['close'].ewm(span=12, adjust=False).mean()
    long_ema = data['close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = short_ema - long_ema
    data['Signal Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    return data

def calculate_bollinger_bands(data, window=20, num_std=2):
    rolling_mean = data['close'].rolling(window=window).mean()
    rolling_std = data['close'].rolling(window=window).std()
    data['Upper Band'] = rolling_mean + (rolling_std * num_std)
    data['Lower Band'] = rolling_mean - (rolling_std * num_std)
    return data

def calculate_sma(data, period):
    data[f'SMA {period}'] = data['close'].rolling(period).mean()
    return data

def load_model(asset):
    model_path = f'model/{asset}_model.pkl'
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.warning(f'Không tìm thấy mô hình cho {asset}. Vui lòng huấn luyện trước.')
        return None

def predict_next_day(asset, data):
    model = load_model(asset)
    if model is None or data.empty:
        return None
    latest_data = data.iloc[-1]
    features = latest_data[['open', 'high', 'low', 'volume']].values.reshape(1, -1)
    prediction = model.predict(features)[0]
    return prediction

def main():
    st.sidebar.title("Options")

    stock_df = get_stock_list()

    if stock_df.empty:
        st.error("Không thể tải danh sách công ty.")
        return

    st.subheader("Danh sách công ty")
    st.dataframe(stock_df)

    st.sidebar.subheader('Chọn mã cổ phiếu')
    asset = st.sidebar.selectbox("Chọn mã cổ phiếu", stock_df['ticker'].tolist(), index=0)

    start_date = st.sidebar.date_input("Chọn ngày bắt đầu", datetime(2024, 1, 1))
    start_date = start_date.strftime('%Y-%m-%d')

    if asset:
        st.title(f"{asset} - Biểu đồ chứng khoán")
        data = load_vnstock_data(asset, start_date)

        if data is not None and not data.empty:
            data = calculate_macd(data)

            if st.sidebar.checkbox('SMA'):
                period = st.sidebar.slider('SMA period', min_value=5, max_value=500, value=20, step=1)
                data = calculate_sma(data, period)

            if st.sidebar.checkbox('Bollinger Bands'):
                data = calculate_bollinger_bands(data)

            data2 = data[['close']].copy()

            sma_columns = [col for col in data.columns if col.startswith("SMA")]
            for col in sma_columns:
                data2[col] = data[col]

            if 'Upper Band' in data.columns and 'Lower Band' in data.columns:
                data2['Upper Band'] = data['Upper Band']
                data2['Lower Band'] = data['Lower Band']

            #st.subheader('Biểu đồ giá cổ phiếu')
            #st.line_chart(data2)

            if st.sidebar.checkbox('MACD'):
                data2['MACD'] = data['MACD']
                data2['Signal Line'] = data['Signal Line']

        # Hiển thị biểu đồ
            st.subheader('Biểu đồ chứng khoán')
            st.line_chart(data2)

            if st.sidebar.checkbox('Xem thống kê'):
                st.subheader('Thống kê dữ liệu')
                st.table(data2.describe())

            if st.sidebar.checkbox('Xem dữ liệu chi tiết'):
                st.subheader(f'Dữ liệu lịch sử {asset}')
                detailed_data = data[['open', 'high', 'low', 'close', 'volume']]
                st.write(detailed_data)

            selected_date = st.sidebar.date_input("Chọn ngày để dự báo ngày tiếp theo")
            selected_date_str = selected_date.strftime('%Y-%m-%d')

            if st.sidebar.button('Dự báo cho ngày tiếp theo của ngày đã chọn'):
                if selected_date_str in data.index.strftime('%Y-%m-%d').values:
                    selected_data = data.loc[selected_date_str]
                    selected_features = selected_data[['open', 'high', 'low', 'volume']].values.reshape(1, -1)
                    model = load_model(asset)
                    if model:
                        predicted_close = model.predict(selected_features)[0]
                        next_day = datetime.strptime(selected_date_str, '%Y-%m-%d') + timedelta(days=1)
                        next_day_str = next_day.strftime('%Y-%m-%d')
                        actual_close = data.loc[next_day_str, 'close'] if next_day_str in data.index.strftime('%Y-%m-%d').values else None

                        st.subheader('Kết quả dự báo')
                        st.write(f"Ngày được chọn: {selected_date_str}")
                        st.write(f"Giá đóng cửa dự báo cho ngày tiếp theo ({next_day_str}): {predicted_close:.2f}")

                        if actual_close is not None:
                            st.write(f"Giá đóng cửa thực tế: {actual_close:.2f}")
                        else:
                            st.warning("Chưa có dữ liệu thực tế cho ngày tiếp theo.")
                    else:
                        st.error("Không thể tải mô hình để dự báo.")
                else:
                    st.error("Ngày được chọn không tồn tại trong dữ liệu.")

        else:
            st.warning("Không có dữ liệu hoặc mã cổ phiếu không hợp lệ.")

if __name__ == '__main__':
    main()
