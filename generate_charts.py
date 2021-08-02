import pandas as pd
import plotly.graph_objects as go
import random
import os
import cv2
from uuid import uuid4
import matplotlib
import matplotlib.pyplot as plt
import mplfinance as mpf
matplotlib.use('Agg')

df = pd.read_csv('/tf/stuff/data/Binance_ADAUSDT_minute.csv')
df1 = pd.read_csv('/tf/stuff/data/Binance_BTCUSDT_minute.csv')
df2 = pd.read_csv('/tf/stuff/data/Binance_LINKUSDT_minute.csv')
df3 = pd.read_csv('/tf/stuff/data/Binance_ETHUSDT_minute.csv')
df4 = pd.read_csv('/tf/stuff/data/Binance_XRPUSDT_minute.csv')

df = pd.concat([df, df1, df2, df3, df4])

reduced = df[['unix', 'open', 'high', 'low', 'close']]
reduced['date'] = pd.to_datetime(reduced['unix'],unit='ms')
reduced.set_index(['date'], inplace=True)
df = reduced
name = 'h'
predict_length = 20
chart_range = 200
classes = [0.95, 0.96, 0.97, 0.98, 0.99, 1.0, 1.01, 1.02, 1.03, 1.04, 1.05]

scale_percent = 5
folder_name = 'chart_high_fidelity_200h_20_proper'

for c in classes:
    if not os.path.exists(f'/tf/stuff/data/{folder_name}/'):
        os.makedirs(f'/tf/stuff/data/{folder_name}/')

train = int(df.shape[0] * 0.8)

chart_data = []

produced = 0
while produced < 10000:

    if produced % 100 == 0:
        print(produced)

    count = df.shape[0]
    start = random.randint(0, train - predict_length)
    end   = start + chart_range
    last_price = df.iloc[end]['close']
    predict = df.iloc[end + predict_length]['close']
    delta = last_price / predict
    rounded_delta = round(last_price / predict, 2)

    if rounded_delta >= 1.05: rounded_delta = 1.05
    elif rounded_delta <= 0.95: rounded_delta = 0.95

    produced = produced + 1
    chart = reduced[start:end]

    path = f'/tf/stuff/data/{folder_name}'
    file = path+ f'/{uuid4()}.png'
    mpf.plot(chart, type='candle', style='charles', axisoff=True, savefig=dict(fname=file, bbox_inches="tight"), returnfig=True, closefig=True)
    plt.close('all')
    img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    width = int(img.shape[1] * scale_percent / 50)
    height = int(img.shape[0] * scale_percent / 50)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    cv2.imwrite(file ,resized)

    chart_data.append({"file_path": file, "start": start, "end": end, "predict": end + predict_length, "label": rounded_delta})


data_df = pd.DataFrame(chart_data)
# old_data = pd.read_csv('/tf/stuff/data/chart_high_fidelity_100h_20_proper.csv')

# data_df = pd.concat([old_data, data_df])
data_df.to_csv(f'/tf/stuff/data/{folder_name}.csv')