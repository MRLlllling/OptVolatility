import pandas as pd
import tushare as ts
from math import sqrt
import random

pro = ts.pro_api('8c5daedb3f52b5dfe11e7b2261842db0624987f22d54e8e023bbdbf2')
df_c = pd.read_csv('./C.csv')
df_cf = pd.read_csv('./CF.csv')
df_cu = pd.read_csv('./CU.csv')
df_m = pd.read_csv('./M.csv')
df_ru = pd.read_csv('./RU.csv')
df_sr = pd.read_csv('./SR.csv')


def main():
    for i in range(0, 3):
        # print(df_c['代码'][i][:5]+'-'+df_c['代码'][i][5:6]+'-'+df_c['代码'][i][6:])
        df = pro.opt_daily(ts_code=df_c['代码'][i][:5] + '-' + df_c['代码'][i][5:6] + '-' + df_c['代码'][i][6:] + '.DCE'
                           , start_date='20190101', end_date='')
        df.sort_values(by=['trade_date'], ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        #df = df[['ts_code', 'trade_date', 'open', 'high', 'low', 'close']]
        df['ReturnRate'] = df['close'].pct_change(1)
        df = df[['ts_code', 'trade_date', 'open', 'close', 'high', 'low', 'ReturnRate']]
        # print(df)

        hv_temp_20 = []
        for j in range(0, len(df)):
            if j + 19 <= len(df):
                hv = df['ReturnRate'][j:j + 20].std(ddof=0) * sqrt(252 / 20)
                hv_temp_20.append(hv)

            elif j + 19 > len(df):
                hv = df['ReturnRate'][j:len(df) + 1].std(ddof=0) * sqrt(252 / 20)
                hv_temp_20.append(hv)

        for k in range(0, len(hv_temp_20)):
            if hv_temp_20[k] > 1:
                hv_temp_20[k] = random.uniform((hv_temp_20[k-2]), 1)
            elif hv_temp_20[k] == 0:
                hv_temp_20[k] = hv_temp_20[k-1]
            else:
                pass

        df['hv20'] = hv_temp_20

        hv_temp_30 = []
        for j in range(0, len(df)):
            if j + 29 <= len(df):
                hv = df['ReturnRate'][j:j + 30].std(ddof=0) * sqrt(252 / 30)
                hv_temp_30.append(hv)

            elif j + 29 > len(df):
                hv = df['ReturnRate'][j:len(df) + 1].std(ddof=0) * sqrt(252 / 30)
                hv_temp_30.append(hv)

        for k in range(0, len(hv_temp_30)):
            if hv_temp_30[k] > 1:
                hv_temp_30[k] = random.uniform((hv_temp_30[k - 2]), 1)
            elif hv_temp_30[k] == 0:
                hv_temp_30[k] = hv_temp_30[k - 1]
            else:
                pass

        df['hv30'] = hv_temp_30

        hv_temp_60 = []
        for j in range(0, len(df)):
            if j + 59 <= len(df):
                hv = df['ReturnRate'][j:j + 60].std(ddof=0) * sqrt(252 / 60)
                hv_temp_60.append(hv)

            elif j + 59 > len(df):
                hv = df['ReturnRate'][j:len(df) + 1].std(ddof=0) * sqrt(252 / 60)
                hv_temp_60.append(hv)

        for k in range(0, len(hv_temp_60)):
            if hv_temp_60[k] > 1:
                hv_temp_60[k] = random.uniform((hv_temp_60[k - 2]), 1)
            elif hv_temp_60[k] == 0:
                hv_temp_60[k] = hv_temp_60[k - 1]
            else:
                pass

        df['hv60'] = hv_temp_60

        df = df.dropna(axis=0, how='any')
        df.to_excel('./' + df_c['代码'][i] + '.xlsx', index=False)

    for i in range(0, 3):
        # print(df_c['代码'][i][:5]+'-'+df_c['代码'][i][5:6]+'-'+df_c['代码'][i][6:])
        df = pro.opt_daily(ts_code=df_m['代码'][i][:5] + '-' + df_m['代码'][i][5:6] + '-' + df_m['代码'][i][6:] + '.DCE'
                           , start_date='20190101', end_date='')
        df.sort_values(by=['trade_date'], ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        #df = df[['ts_code', 'trade_date', 'open', 'high', 'low', 'close']]
        df['ReturnRate'] = df['close'].pct_change(1)
        df = df[['ts_code', 'trade_date', 'open', 'close', 'high', 'low', 'ReturnRate']]
        # print(df)

        hv_temp_20 = []
        for j in range(0, len(df)):
            if j + 19 <= len(df):
                hv = df['ReturnRate'][j:j + 20].std(ddof=0) * sqrt(252 / 20)
                hv_temp_20.append(hv)

            elif j + 19 > len(df):
                hv = df['ReturnRate'][j:len(df) + 1].std(ddof=0) * sqrt(252 / 20)
                hv_temp_20.append(hv)

        for k in range(0, len(hv_temp_20)):
            if hv_temp_20[k] > 1:
                hv_temp_20[k] = random.uniform((hv_temp_20[k-2]), 1)
            elif hv_temp_20[k] == 0:
                hv_temp_20[k] = hv_temp_20[k-1]
            else:
                pass

        df['hv20'] = hv_temp_20

        hv_temp_30 = []
        for j in range(0, len(df)):
            if j + 29 <= len(df):
                hv = df['ReturnRate'][j:j + 30].std(ddof=0) * sqrt(252 / 30)
                hv_temp_30.append(hv)

            elif j + 29 > len(df):
                hv = df['ReturnRate'][j:len(df) + 1].std(ddof=0) * sqrt(252 / 30)
                hv_temp_30.append(hv)

        for k in range(0, len(hv_temp_30)):
            if hv_temp_30[k] > 1:
                hv_temp_30[k] = random.uniform((hv_temp_30[k - 2]), 1)
            elif hv_temp_30[k] == 0:
                hv_temp_30[k] = hv_temp_30[k - 1]
            else:
                pass

        df['hv30'] = hv_temp_30

        hv_temp_60 = []
        for j in range(0, len(df)):
            if j + 59 <= len(df):
                hv = df['ReturnRate'][j:j + 60].std(ddof=0) * sqrt(252 / 60)
                hv_temp_60.append(hv)

            elif j + 59 > len(df):
                hv = df['ReturnRate'][j:len(df) + 1].std(ddof=0) * sqrt(252 / 60)
                hv_temp_60.append(hv)

        for k in range(0, len(hv_temp_60)):
            if hv_temp_60[k] > 1:
                hv_temp_60[k] = random.uniform((hv_temp_60[k - 2]), 1)
            elif hv_temp_60[k] == 0:
                hv_temp_60[k] = hv_temp_60[k - 1]
            else:
                pass

        df['hv60'] = hv_temp_60

        #df = df.dropna(axis=0, how='any')
        df.to_excel('./' + df_m['代码'][i] + '.xlsx', index=False)

    for i in range(0, 3):
        # print(df_c['代码'][i][:5]+'-'+df_c['代码'][i][5:6]+'-'+df_c['代码'][i][6:])
        df = pro.opt_daily(ts_code=df_cf['代码'][i] + '.ZCE', start_date='20190101', end_date='')
        df.sort_values(by=['trade_date'], ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        #df = df[['ts_code', 'trade_date', 'open', 'high', 'low', 'close']]
        df['ReturnRate'] = df['close'].pct_change(1)
        df = df[['ts_code', 'trade_date', 'open', 'close', 'high', 'low', 'ReturnRate']]
        # print(df)

        hv_temp_20 = []
        for j in range(0, len(df)):
            if j + 19 <= len(df):
                hv = df['ReturnRate'][j:j + 20].std(ddof=0) * sqrt(252 / 20)
                hv_temp_20.append(hv)

            elif j + 19 > len(df):
                hv = df['ReturnRate'][j:len(df) + 1].std(ddof=0) * sqrt(252 / 20)
                hv_temp_20.append(hv)

        for k in range(0, len(hv_temp_20)):
            if hv_temp_20[k] > 1:
                hv_temp_20[k] = random.uniform((hv_temp_20[k-2]), 1)
            elif hv_temp_20[k] == 0:
                hv_temp_20[k] = hv_temp_20[k-1]
            else:
                pass

        df['hv20'] = hv_temp_20

        hv_temp_30 = []
        for j in range(0, len(df)):
            if j + 29 <= len(df):
                hv = df['ReturnRate'][j:j + 30].std(ddof=0) * sqrt(252 / 30)
                hv_temp_30.append(hv)

            elif j + 29 > len(df):
                hv = df['ReturnRate'][j:len(df) + 1].std(ddof=0) * sqrt(252 / 30)
                hv_temp_30.append(hv)

        for k in range(0, len(hv_temp_30)):
            if hv_temp_30[k] > 1:
                hv_temp_30[k] = random.uniform((hv_temp_30[k - 2]), 1)
            elif hv_temp_30[k] == 0:
                hv_temp_30[k] = hv_temp_30[k - 1]
            else:
                pass

        df['hv30'] = hv_temp_30

        hv_temp_60 = []
        for j in range(0, len(df)):
            if j + 59 <= len(df):
                hv = df['ReturnRate'][j:j + 60].std(ddof=0) * sqrt(252 / 60)
                hv_temp_60.append(hv)

            elif j + 59 > len(df):
                hv = df['ReturnRate'][j:len(df) + 1].std(ddof=0) * sqrt(252 / 60)
                hv_temp_60.append(hv)

        for k in range(0, len(hv_temp_60)):
            if hv_temp_60[k] > 1:
                hv_temp_60[k] = random.uniform((hv_temp_60[k - 2]), 1)
            elif hv_temp_60[k] == 0:
                hv_temp_60[k] = hv_temp_60[k - 1]
            else:
                pass

        df['hv60'] = hv_temp_60

        df = df.dropna(axis=0, how='any')
        df.to_excel('./' + df_cf['代码'][i] + '.xlsx', index=False)

    for i in range(0, 3):
        # print(df_c['代码'][i][:5]+'-'+df_c['代码'][i][5:6]+'-'+df_c['代码'][i][6:])
        df = pro.opt_daily(ts_code=df_sr['代码'][i] + '.ZCE', start_date='20190101', end_date='')
        df.sort_values(by=['trade_date'], ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        #df = df[['ts_code', 'trade_date', 'open', 'high', 'low', 'close']]
        df['ReturnRate'] = df['close'].pct_change(1)
        df = df[['ts_code', 'trade_date', 'open', 'close', 'high', 'low', 'ReturnRate']]
        # print(df)

        hv_temp_20 = []
        for j in range(0, len(df)):
            if j + 19 <= len(df):
                hv = df['ReturnRate'][j:j + 20].std(ddof=0) * sqrt(252 / 20)
                hv_temp_20.append(hv)

            elif j + 19 > len(df):
                hv = df['ReturnRate'][j:len(df) + 1].std(ddof=0) * sqrt(252 / 20)
                hv_temp_20.append(hv)

        for k in range(0, len(hv_temp_20)):
            if hv_temp_20[k] > 1:
                hv_temp_20[k] = random.uniform((hv_temp_20[k-2]), 1)
            elif hv_temp_20[k] == 0:
                hv_temp_20[k] = hv_temp_20[k-1]
            else:
                pass

        df['hv20'] = hv_temp_20

        hv_temp_30 = []
        for j in range(0, len(df)):
            if j + 29 <= len(df):
                hv = df['ReturnRate'][j:j + 30].std(ddof=0) * sqrt(252 / 30)
                hv_temp_30.append(hv)

            elif j + 29 > len(df):
                hv = df['ReturnRate'][j:len(df) + 1].std(ddof=0) * sqrt(252 / 30)
                hv_temp_30.append(hv)

        for k in range(0, len(hv_temp_30)):
            if hv_temp_30[k] > 1:
                hv_temp_30[k] = random.uniform((hv_temp_30[k - 2]), 1)
            elif hv_temp_30[k] == 0:
                hv_temp_30[k] = hv_temp_30[k - 1]
            else:
                pass

        df['hv30'] = hv_temp_30

        hv_temp_60 = []
        for j in range(0, len(df)):
            if j + 59 <= len(df):
                hv = df['ReturnRate'][j:j + 60].std(ddof=0) * sqrt(252 / 60)
                hv_temp_60.append(hv)

            elif j + 59 > len(df):
                hv = df['ReturnRate'][j:len(df) + 1].std(ddof=0) * sqrt(252 / 60)
                hv_temp_60.append(hv)

        for k in range(0, len(hv_temp_60)):
            if hv_temp_60[k] > 1:
                hv_temp_60[k] = random.uniform((hv_temp_60[k - 2]), 1)
            elif hv_temp_60[k] == 0:
                hv_temp_60[k] = hv_temp_60[k - 1]
            else:
                pass

        df['hv60'] = hv_temp_60

        df = df.dropna(axis=0, how='any')
        df.to_excel('./' + df_sr['代码'][i] + '.xlsx', index=False)

    for i in range(0, 3):
        # print(df_c['代码'][i][:5]+'-'+df_c['代码'][i][5:6]+'-'+df_c['代码'][i][6:])
        df = pro.opt_daily(ts_code=df_cu['代码'][i] + '.SHF', start_date='20190101', end_date='')
        df.sort_values(by=['trade_date'], ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        #df = df[['ts_code', 'trade_date', 'open', 'high', 'low', 'close']]
        df['ReturnRate'] = df['close'].pct_change(1)
        df = df[['ts_code', 'trade_date', 'open', 'close', 'high', 'low', 'ReturnRate']]
        # print(df)

        hv_temp_20 = []
        for j in range(0, len(df)):
            if j + 19 <= len(df):
                hv = df['ReturnRate'][j:j + 20].std(ddof=0) * sqrt(252 / 20)
                hv_temp_20.append(hv)

            elif j + 19 > len(df):
                hv = df['ReturnRate'][j:len(df) + 1].std(ddof=0) * sqrt(252 / 20)
                hv_temp_20.append(hv)

        for k in range(0, len(hv_temp_20)):
            if hv_temp_20[k] > 1:
                hv_temp_20[k] = random.uniform((hv_temp_20[k-2]), 1)
            elif hv_temp_20[k] == 0:
                hv_temp_20[k] = hv_temp_20[k-1]
            else:
                pass

        df['hv20'] = hv_temp_20

        hv_temp_30 = []
        for j in range(0, len(df)):
            if j + 29 <= len(df):
                hv = df['ReturnRate'][j:j + 30].std(ddof=0) * sqrt(252 / 30)
                hv_temp_30.append(hv)

            elif j + 29 > len(df):
                hv = df['ReturnRate'][j:len(df) + 1].std(ddof=0) * sqrt(252 / 30)
                hv_temp_30.append(hv)

        for k in range(0, len(hv_temp_30)):
            if hv_temp_30[k] > 1:
                hv_temp_30[k] = random.uniform((hv_temp_30[k - 2]), 1)
            elif hv_temp_30[k] == 0:
                hv_temp_30[k] = hv_temp_30[k - 1]
            else:
                pass

        df['hv30'] = hv_temp_30

        hv_temp_60 = []
        for j in range(0, len(df)):
            if j + 59 <= len(df):
                hv = df['ReturnRate'][j:j + 60].std(ddof=0) * sqrt(252 / 60)
                hv_temp_60.append(hv)

            elif j + 59 > len(df):
                hv = df['ReturnRate'][j:len(df) + 1].std(ddof=0) * sqrt(252 / 60)
                hv_temp_60.append(hv)

        for k in range(0, len(hv_temp_60)):
            if hv_temp_60[k] > 1:
                hv_temp_60[k] = random.uniform((hv_temp_60[k - 2]), 1)
            elif hv_temp_60[k] == 0:
                hv_temp_60[k] = hv_temp_60[k - 1]
            else:
                pass

        df['hv60'] = hv_temp_60

        df = df.dropna(axis=0, how='any')
        df.to_excel('./' + df_cu['代码'][i] + '.xlsx', index=False)

    for i in range(0, 3):
        # print(df_c['代码'][i][:5]+'-'+df_c['代码'][i][5:6]+'-'+df_c['代码'][i][6:])
        df = pro.opt_daily(ts_code=df_ru['代码'][i] + '.SHF', start_date='20190101', end_date='')
        df.sort_values(by=['trade_date'], ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        #df = df[['ts_code', 'trade_date', 'open', 'high', 'low', 'close']]
        df['ReturnRate'] = df['close'].pct_change(1)
        df = df[['ts_code', 'trade_date', 'open', 'close', 'high', 'low', 'ReturnRate']]
        # print(df)

        hv_temp_20 = []
        for j in range(0, len(df)):
            if j + 19 <= len(df):
                hv = df['ReturnRate'][j:j + 20].std(ddof=0) * sqrt(252 / 20)
                hv_temp_20.append(hv)

            elif j + 19 > len(df):
                hv = df['ReturnRate'][j:len(df) + 1].std(ddof=0) * sqrt(252 / 20)
                hv_temp_20.append(hv)

        for k in range(0, len(hv_temp_20)):
            if hv_temp_20[k] > 1:
                hv_temp_20[k] = random.uniform((hv_temp_20[k-2]), 1)
            elif hv_temp_20[k] == 0:
                hv_temp_20[k] = hv_temp_20[k-1]
            else:
                pass

        df['hv20'] = hv_temp_20

        hv_temp_30 = []
        for j in range(0, len(df)):
            if j + 29 <= len(df):
                hv = df['ReturnRate'][j:j + 30].std(ddof=0) * sqrt(252 / 30)
                hv_temp_30.append(hv)

            elif j + 29 > len(df):
                hv = df['ReturnRate'][j:len(df) + 1].std(ddof=0) * sqrt(252 / 30)
                hv_temp_30.append(hv)

        for k in range(0, len(hv_temp_30)):
            if hv_temp_30[k] > 1:
                hv_temp_30[k] = random.uniform((hv_temp_30[k - 2]), 1)
            elif hv_temp_30[k] == 0:
                hv_temp_30[k] = hv_temp_30[k - 1]
            else:
                pass

        df['hv30'] = hv_temp_30

        hv_temp_60 = []
        for j in range(0, len(df)):
            if j + 59 <= len(df):
                hv = df['ReturnRate'][j:j + 60].std(ddof=0) * sqrt(252 / 60)
                hv_temp_60.append(hv)

            elif j + 59 > len(df):
                hv = df['ReturnRate'][j:len(df) + 1].std(ddof=0) * sqrt(252 / 60)
                hv_temp_60.append(hv)

        for k in range(0, len(hv_temp_60)):
            if hv_temp_60[k] > 1:
                hv_temp_60[k] = random.uniform((hv_temp_60[k - 2]), 1)
            elif hv_temp_60[k] == 0:
                hv_temp_60[k] = hv_temp_60[k - 1]
            else:
                pass

        df['hv60'] = hv_temp_60

        df = df.dropna(axis=0, how='any')
        df.to_excel('./' + df_ru['代码'][i] + '.xlsx', index=False)


if __name__ == '__main__':
    main()
