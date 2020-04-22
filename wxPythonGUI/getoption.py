import time
from requests import get
import pandas as pd

URL = "http://nufm.dfcfw.com/EM_Finance2014NumericApplication/JS.aspx?" \
      "cb=jQuery112409097934162577812_{time_stamp}&type=CT&token=4f1862fc3b5e77c150a2b985b12db0fd" \
      "&sty=FC2UCO&js=({{data:[(x)],recordsFiltered:(tot)}})&cmd=C.{market}OPTION&st=(Code)&sr=-1" \
      "&p=1&ps=10000&_={time_stamp}"
MARKETS = ['CZCE', 'SHFE', 'DCE']
HEADERS = ['代码', '名称', '最新价', '涨跌额', '涨跌幅(%)', '成交量', '成交额', '持仓量', '行权价', '剩余日',
           '日增', '昨结', '今开']


def main():
    data1 = []
    time_stamp = int(round(time.time() * 1000))
    for market in MARKETS:
        data = get(URL.format(market=market, time_stamp=time_stamp)).content.decode()
        data_list = data.split('["')[1].split('"]')[0].split('","')
        for i in data_list:
            data_item = dict(zip(HEADERS, i.split(',')[1:14]))
            data_item['最新价'] = None if data_item['最新价'] == '-' else float(data_item['最新价'])
            data_item['涨跌额'] = None if data_item['涨跌额'] == '-' else float(data_item['涨跌额'])
            data_item['涨跌幅(%)'] = None if data_item['涨跌幅(%)'] == '-' else float(data_item['涨跌幅(%)'])
            data_item['成交量'] = None if data_item['成交量'] == '-' else int(data_item['成交量'])
            data_item['成交额'] = None if data_item['成交额'] == '-' else float(data_item['成交额'])
            data_item['持仓量'] = None if data_item['持仓量'] == '-' else int(data_item['持仓量'])
            data_item['行权价'] = float(data_item['行权价'])
            data_item['剩余日'] = None if data_item['剩余日'] == '-' else int(data_item['剩余日'])
            data_item['日增'] = None if data_item['日增'] == '-' else int(data_item['日增'])
            data_item['昨结'] = float(data_item['昨结'])
            data_item['今开'] = None if data_item['今开'] == '-' else float(data_item['今开'])

            data1.append(data_item)

    df = pd.DataFrame(data1)

    df = df[~df['代码'].str.contains('P')]

    df['剩余日'].fillna(25, inplace=True)
    #df = df.dropna(axis=0, how='any')

    print(df)
    df.to_csv('./total.csv', index=False, encoding='utf_8_sig')

    df = pd.read_csv('./total.csv')

    df_c = df[df['名称'].str.contains('玉米')]
    df_c.reset_index(drop=True, inplace=True)
    df_c = df_c.sort_values(by=['成交量'], ascending=False)
    df_c.to_csv('./C.csv', index=False, encoding='utf_8_sig')

    df_sr = df[df['名称'].str.contains('白糖')]
    df_sr.reset_index(drop=True, inplace=True)
    df_sr = df_sr.sort_values(by=['成交量'], ascending=False)
    df_sr.to_csv('./SR.csv', index=False, encoding='utf_8_sig')

    df_cf = df[df['名称'].str.contains('棉花')]
    df_cf.reset_index(drop=True, inplace=True)
    df_cf = df_cf.sort_values(by=['成交量'], ascending=False)
    df_cf.to_csv('./CF.csv', index=False, encoding='utf_8_sig')

    df_ru = df[df['名称'].str.contains('橡胶')]
    df_ru.reset_index(drop=True, inplace=True)
    df_ru = df_ru.sort_values(by=['成交量'], ascending=False)
    df_ru.to_csv('./RU.csv', index=False, encoding='utf_8_sig')

    df_cu = df[df['名称'].str.contains('沪铜')]
    df_cu.reset_index(drop=True, inplace=True)
    df_cu = df_cu.sort_values(by=['成交量'], ascending=False)
    df_cu.to_csv('./CU.csv', index=False, encoding='utf_8_sig')

    df_m = df[df['名称'].str.contains('豆粕')]
    df_m.reset_index(drop=True, inplace=True)
    df_m = df_m.sort_values(by=['成交量'], ascending=False)
    df_m.to_csv('./M.csv', index=False, encoding='utf_8_sig')


if __name__ == '__main__':
    main()
