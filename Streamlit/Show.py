import streamlit as st
import os
import pandas as pd
from bokeh.models import HoverTool, ColumnDataSource, NumeralTickFormatter, DatetimeTickFormatter, Legend
from bokeh.plotting import figure
import numpy as np
from math import isnan
import plotly.graph_objs as go
import plotly.offline as py

# 忽略pandas dataframe.copy警告
pd.set_option('mode.chained_assignment', None)

dirs = 'd:/期货期权/处理数据/波动率/'
file = os.listdir(dirs)

dirs_iv = 'd:/期货期权/隐含波动率/相关数据/'
file_iv = os.listdir(dirs_iv)


def iv_surface(file_name, opt_name):
    df = pd.read_csv('d:/期货期权/隐含波动率/' + file_name)
    df = df.drop(columns=['ts_code', 'close'])

    df2 = pd.read_csv('d:/期货期权/隐含波动率/相关数据/' + file_name)
    df['10d'] = df2['10d']
    df['20d'] = df2['20d']
    df['30d'] = df2['30d']
    df['60d'] = df2['60d']

    df = df.sort_values(by=['exercise_price'], ascending=1)

    for i in range(0, len(df)):
        if str(isnan(df['10d'][i])) == 'True':
            df['10d'][i] = 0
        if str(isnan(df['20d'][i])) == 'True':
            df['20d'][i] = 0
        if str(isnan(df['30d'][i])) == 'True':
            df['30d'][i] = 0
        if str(isnan(df['60d'][i])) == 'True':
            df['60d'][i] = 0

    df2 = df.drop(columns='exercise_price')

    x = np.array(df.exercise_price)
    y = np.array([10, 20, 30, 60])

    x, y = np.meshgrid(x, y)

    z = np.array(df2)
    z = z.T

    fig = go.Figure(data=[go.Surface(x=y, y=x, z=z, colorscale='jet')])
    fig.update_layout(
        title_text=opt_name + '波动率曲面',
        autosize=False,
        height=800,
        width=1000,
        scene=dict(
            xaxis_title='maturity',
            yaxis_title='option price',
            zaxis_title='rate'
        )
    )

    st.plotly_chart(fig)


def hiv(file_name):
    hiv_data = pd.read_csv('d:/期货期权/hiv/' + file_name + '.csv')
    hiv_data['trade_date'] = pd.to_datetime(hiv_data['trade_date'], format='%Y%m%d')
    hiv_data['date_string'] = hiv_data['trade_date'].dt.strftime('%Y/%m/%d')

    src = ColumnDataSource(hiv_data)
    src.data.keys()

    p = figure(title=file_name + 'HV与IV近一月对比图', x_axis_label='Date', y_axis_label='Rate', x_axis_type='datetime', plot_width=800, plot_height=300)
    p1 = p.line(source=src, x='trade_date', y='HV20', line_width=2, color='blue', alpha=0.5)
    p2 = p.line(source=src, x='trade_date', y='IV20', line_width=2, color='red', alpha=0.5)

    legend = Legend(items=[('HV20', [p1]), ('IV20', [p2])])
    p.add_layout(legend, 'right')
    p.legend.click_policy = 'hide'

    p.add_tools(HoverTool(renderers=[p1], tooltips=[('Date', '@date_string'), ('HV20', '@HV20{%0.2F}')]))
    p.add_tools(HoverTool(renderers=[p2], tooltips=[('Date', '@date_string'), ('IV20', '@IV20{%0.2F}')]))

    p.xaxis.formatter = DatetimeTickFormatter(days=["%Y/%m/%d"], months=["%Y/%m/%d"], years=["%Y/%m/%d"])
    p.yaxis.formatter = NumeralTickFormatter(format='%F')

    st.bokeh_chart(p)


###################main########################
st.header('商品期货期权波动率分析展示界面')

data = pd.read_excel('d:/期货期权/处理数据/All.xlsx')
data['HV20_AVG'] = data['HV20_AVG'].apply(lambda x: '%.2f%%' % (x*100))
data['HV30_AVG'] = data['HV30_AVG'].apply(lambda x: '%.2f%%' % (x*100))
data['HV60_AVG'] = data['HV60_AVG'].apply(lambda x: '%.2f%%' % (x*100))
data['HV90_AVG'] = data['HV90_AVG'].apply(lambda x: '%.2f%%' % (x*100))

data_iv = pd.read_excel('d:/期货期权/隐含波动率/总表.xlsx')
data_iv['平值隐含波动率'] = data_iv['平值隐含波动率'].apply(lambda x: '%.2f%%' % (x*100))
data_iv['加权隐含波动率'] = data_iv['加权隐含波动率'].apply(lambda x: '%.2f%%' % (x*100))
data_iv['历史波动率(20)'] = data_iv['历史波动率(20)'].apply(lambda x: '%.2f%%' % (x*100))

future = st.sidebar.selectbox('请选择需要查看的期货:', ['概况', '金', '玉米', '棉花', '铜', '铁矿石', '豆粕', '甲醇', '菜粕', '橡胶', '白糖', 'PTA'])


if future == '概况':
    st.subheader('历史波动率')
    st.dataframe(data)
    st.subheader('隐含波动率')
    st.dataframe(data_iv)


elif future == '金':
    # if st.sidebar.checkbox('沪-金-AU', key='au'):
    au_show = data.iloc[0:1]
    au_iv_show = data_iv.iloc[0:1]
    if st.sidebar.checkbox('Historical Volatility', key='au'):
        st.subheader('沪金AU近年历史波动率数据')
        st.dataframe(au_show)

        # au K线图
        au_data = pd.read_excel('d:/期货期权/处理数据/波动率/' + file[0], index_col=0)
        au = au_data[['trade_date', 'HV10', 'HV20', 'HV30', 'HV60']]
        au['trade_date'] = pd.to_datetime(au['trade_date'], format='%Y%m%d')
        au['date_string'] = au['trade_date'].dt.strftime('%Y/%m/%d')

        src = ColumnDataSource(au)
        src.data.keys()

        p = figure(title=data['ts_code'][0] + ' HV', x_axis_label='Date', y_axis_label='Rate', x_axis_type='datetime', plot_width=800, plot_height=300)

        p1 = p.line(source=src, x='trade_date', y='HV10', line_width=2, color='blue', alpha=0.5)
        p2 = p.line(source=src, x='trade_date', y='HV20', line_width=2, color='red', alpha=0.5)
        p3 = p.line(source=src, x='trade_date', y='HV30', line_width=2, color='green', alpha=0.5)
        p4 = p.line(source=src, x='trade_date', y='HV60', line_width=2, color='yellow', alpha=0.5)

        legend = Legend(items=[('HV10', [p1]), ('HV20', [p2]), ('HV30', [p3]), ('HV60', [p4])])
        p.add_layout(legend, 'right')
        p.legend.click_policy = 'hide'

        p.add_tools(HoverTool(renderers=[p1], tooltips=[('Date', '@date_string'), ('HV10', '@HV10{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p2], tooltips=[('Date', '@date_string'), ('HV20', '@HV20{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p3], tooltips=[('Date', '@date_string'), ('HV30', '@HV30{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p4], tooltips=[('Date', '@date_string'), ('HV60', '@HV60{%0.2F}')]))

        p.xaxis.formatter = DatetimeTickFormatter(days=["%Y/%m/%d"], months=["%Y/%m/%d"], years=["%Y/%m/%d"])
        p.yaxis.formatter = NumeralTickFormatter(format='%F')

        st.bokeh_chart(p)

        # au 历史波动率锥图
        st.subheader('AU近年历史波动率锥')
        au_awl = pd.read_excel('d:/期货期权/处理数据/波动率/' + file[0], index_col=0)
        au_awl = pd.DataFrame({'HV_name': au_awl['HV'][:8], '10分位': au_awl['10分位'][:8], '25分位': au_awl['25分位'][:8], '50分位': au_awl['50分位'][:8], '75分位': au_awl['75分位'][:8], '90分位': au_awl['90分位'][:8]})
        au_awl = au_awl.set_index('HV_name')

        st.dataframe(au_awl)
        st.line_chart(au_awl)

    if st.sidebar.checkbox('Implied Volatility', key='au'):
        st.subheader('沪金AU近期隐含波动率数据')
        st.dataframe(au_iv_show)

        au_iv_info = pd.read_csv('d:/期货期权/隐含波动率/AU.SHF.csv', index_col=0)
        au_iv_data = pd.read_csv(dirs_iv + file_iv[0], index_col=0)

        au_iv_info['close'] = au_iv_info['close'].iloc[0:len(au_iv_data)]
        au_iv_info = au_iv_info.dropna(axis=0, how='any')
        au_iv_data['close'] = au_iv_info['close'].values

        src = ColumnDataSource(au_iv_data)
        src.data.keys()

        p = figure(title=data_iv['ts_code'][0] + ' IV', x_axis_label='Strike Price', y_axis_label='Rate', plot_width=800, plot_height=300)
        p1 = p.line(source=src, x='close', y='10d', line_width=2, color='blue', alpha=0.5)
        p2 = p.line(source=src, x='close', y='20d', line_width=2, color='red', alpha=0.5)
        p3 = p.line(source=src, x='close', y='30d', line_width=2, color='green', alpha=0.5)
        p4 = p.line(source=src, x='close', y='60d', line_width=2, color='yellow', alpha=0.5)

        legend = Legend(items=[('IV10', [p1]), ('IV20', [p2]), ('IV30', [p3]), ('IV60', [p4])])
        p.add_layout(legend, 'right')
        p.legend.click_policy = 'hide'

        p.add_tools(HoverTool(renderers=[p1], tooltips=[('Strike Price', '@close'), ('IV10', '@10d{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p2], tooltips=[('Strike Price', '@close'), ('IV20', '@20d{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p3], tooltips=[('Strike Price', '@close'), ('IV30', '@30d{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p4], tooltips=[('Strike Price', '@close'), ('IV60', '@60d{%0.2F}')]))

        p.yaxis.formatter = NumeralTickFormatter(format='%F')

        st.bokeh_chart(p)

        if st.sidebar.checkbox('IV Surface', key='au'):
            iv_surface('AU.SHF.csv', '沪金AU')

        if st.sidebar.checkbox('HV&IV', key='au'):
            hiv('AU.SHF')

    st.write('____________________________________________________________')


elif future == '玉米':
    c_show = data.iloc[1:2]
    c_iv_show = data_iv.iloc[1:2]
    if st.sidebar.checkbox('Historical Volatility', key='c'):
        st.subheader('大连玉米C近年历史波动率数据')
        st.dataframe(c_show)

        # c K线图
        c_data = pd.read_excel('d:/期货期权/处理数据/波动率/' + file[1], index_col=0)
        c = c_data[['trade_date', 'HV10', 'HV20', 'HV30', 'HV60']]
        c['trade_date'] = pd.to_datetime(c['trade_date'], format='%Y%m%d')
        c['date_string'] = c['trade_date'].dt.strftime('%Y/%m/%d')

        src = ColumnDataSource(c)
        src.data.keys()

        p = figure(title=data['ts_code'][1] + ' HV', x_axis_label='Date', y_axis_label='Rate', x_axis_type='datetime', plot_width=800, plot_height=300)

        p1 = p.line(source=src, x='trade_date', y='HV10', line_width=2, color='blue', alpha=0.5)
        p2 = p.line(source=src, x='trade_date', y='HV20', line_width=2, color='red', alpha=0.5)
        p3 = p.line(source=src, x='trade_date', y='HV30', line_width=2, color='green', alpha=0.5)
        p4 = p.line(source=src, x='trade_date', y='HV60', line_width=2, color='yellow', alpha=0.5)

        legend = Legend(items=[('HV10', [p1]), ('HV20', [p2]), ('HV30', [p3]), ('HV60', [p4])])
        p.add_layout(legend, 'right')
        p.legend.click_policy = 'hide'

        p.add_tools(HoverTool(renderers=[p1], tooltips=[('Date', '@date_string'), ('HV10', '@HV10{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p2], tooltips=[('Date', '@date_string'), ('HV20', '@HV20{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p3], tooltips=[('Date', '@date_string'), ('HV30', '@HV30{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p4], tooltips=[('Date', '@date_string'), ('HV60', '@HV60{%0.2F}')]))

        p.xaxis.formatter = DatetimeTickFormatter(days=["%Y/%m/%d"], months=["%Y/%m/%d"], years=["%Y/%m/%d"])
        p.yaxis.formatter = NumeralTickFormatter(format='%F')

        st.bokeh_chart(p)

        # c 历史波动率锥图
        st.subheader('大连玉米C近年历史波动率锥')
        c_awl = pd.read_excel('d:/期货期权/处理数据/波动率/' + file[1], index_col=0)
        c_awl = pd.DataFrame({'HV': c_awl['HV'][:8], '10分位': c_awl['10分位'][:8], '25分位': c_awl['25分位'][:8], '50分位': c_awl['50分位'][:8], '75分位': c_awl['75分位'][:8], '90分位': c_awl['90分位'][:8]})
        c_awl = c_awl.set_index('HV')

        st.dataframe(c_awl)

        st.line_chart(c_awl)

    if st.sidebar.checkbox('Implied Volatility', key='c'):
        st.subheader('大连玉米C近期隐含波动率数据')
        st.dataframe(c_iv_show)

        c_iv_info = pd.read_csv('d:/期货期权/隐含波动率/C.DCE.csv', index_col=0)
        c_iv_data = pd.read_csv(dirs_iv + file_iv[1], index_col=0)

        c_iv_info['close'] = c_iv_info['close'].iloc[0:len(c_iv_data)]
        c_iv_info = c_iv_info.dropna(axis=0, how='any')
        c_iv_data['close'] = c_iv_info['close'].values

        src = ColumnDataSource(c_iv_data)
        src.data.keys()

        p = figure(title=data_iv['ts_code'][1] + ' IV', x_axis_label='Strike Price', y_axis_label='Rate', plot_width=800, plot_height=300)
        p1 = p.line(source=src, x='close', y='10d', line_width=2, color='blue', alpha=0.5)
        p2 = p.line(source=src, x='close', y='20d', line_width=2, color='red', alpha=0.5)
        p3 = p.line(source=src, x='close', y='30d', line_width=2, color='green', alpha=0.5)
        p4 = p.line(source=src, x='close', y='60d', line_width=2, color='yellow', alpha=0.5)

        legend = Legend(items=[('IV10', [p1]), ('IV20', [p2]), ('IV30', [p3]), ('IV60', [p4])])
        p.add_layout(legend, 'right')
        p.legend.click_policy = 'hide'

        p.add_tools(HoverTool(renderers=[p1], tooltips=[('Strike Price', '@close'), ('IV10', '@10d{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p2], tooltips=[('Strike Price', '@close'), ('IV20', '@20d{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p3], tooltips=[('Strike Price', '@close'), ('IV30', '@30d{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p4], tooltips=[('Strike Price', '@close'), ('IV60', '@60d{%0.2F}')]))

        p.yaxis.formatter = NumeralTickFormatter(format='%F')

        st.bokeh_chart(p)

        if st.sidebar.checkbox('IV Surface', key='c'):
            iv_surface('C.DCE.csv', '大连玉米C')

        if st.sidebar.checkbox('HV&IV', key='c'):
            hiv('C.DCE')

    st.write('____________________________________________________________')


elif future == '棉花':
    cf_show = data.iloc[2:3]
    cf_iv_show = data_iv.iloc[2:3]
    if st.sidebar.checkbox('Historical Volatility', key='cf'):
        st.subheader('郑州棉花CF近年历史波动率数据')
        st.dataframe(cf_show)

        # c K线图
        cf_data = pd.read_excel('d:/期货期权/处理数据/波动率/' + file[2], index_col=0)
        cf = cf_data[['trade_date', 'HV10', 'HV20', 'HV30', 'HV60']]
        cf['trade_date'] = pd.to_datetime(cf['trade_date'], format='%Y%m%d')
        cf['date_string'] = cf['trade_date'].dt.strftime('%Y/%m/%d')

        src = ColumnDataSource(cf)
        src.data.keys()

        p = figure(title=data['ts_code'][2] + ' HV', x_axis_label='Date', y_axis_label='Rate', x_axis_type='datetime', plot_width=800, plot_height=300)

        p1 = p.line(source=src, x='trade_date', y='HV10', line_width=2, color='blue', alpha=0.5)
        p2 = p.line(source=src, x='trade_date', y='HV20', line_width=2, color='red', alpha=0.5)
        p3 = p.line(source=src, x='trade_date', y='HV30', line_width=2, color='green', alpha=0.5)
        p4 = p.line(source=src, x='trade_date', y='HV60', line_width=2, color='yellow', alpha=0.5)

        legend = Legend(items=[('HV10', [p1]), ('HV20', [p2]), ('HV30', [p3]), ('HV60', [p4])])
        p.add_layout(legend, 'right')
        p.legend.click_policy = 'hide'

        p.add_tools(HoverTool(renderers=[p1], tooltips=[('Date', '@date_string'), ('HV10', '@HV10{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p2], tooltips=[('Date', '@date_string'), ('HV20', '@HV20{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p3], tooltips=[('Date', '@date_string'), ('HV30', '@HV30{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p4], tooltips=[('Date', '@date_string'), ('HV60', '@HV60{%0.2F}')]))

        p.xaxis.formatter = DatetimeTickFormatter(days=["%Y/%m/%d"], months=["%Y/%m/%d"], years=["%Y/%m/%d"])
        p.yaxis.formatter = NumeralTickFormatter(format='%F')

        st.bokeh_chart(p)

        # c 历史波动率锥图
        st.subheader('郑州棉花近年历史波动率锥')
        cf_awl = pd.read_excel('d:/期货期权/处理数据/波动率/' + file[2], index_col=0)
        cf_awl = pd.DataFrame({'HV': cf_awl['HV'][:8], '10分位': cf_awl['10分位'][:8], '25分位': cf_awl['25分位'][:8], '50分位': cf_awl['50分位'][:8], '75分位': cf_awl['75分位'][:8], '90分位': cf_awl['90分位'][:8]})
        cf_awl = cf_awl.set_index('HV')

        st.dataframe(cf_awl)

        st.line_chart(cf_awl)

    if st.sidebar.checkbox('Implied Volatility', key='cf'):
        st.subheader('郑州棉花CF近期隐含波动率数据')
        st.dataframe(cf_iv_show)

        cf_iv_info = pd.read_csv('d:/期货期权/隐含波动率/CF.ZCE.csv', index_col=0)
        cf_iv_data = pd.read_csv(dirs_iv + file_iv[2], index_col=0)

        cf_iv_info['close'] = cf_iv_info['close'].iloc[0:len(cf_iv_data)]
        cf_iv_info = cf_iv_info.dropna(axis=0, how='any')
        cf_iv_data['close'] = cf_iv_info['close'].values

        src = ColumnDataSource(cf_iv_data)
        src.data.keys()

        p = figure(title=data_iv['ts_code'][2] + ' IV', x_axis_label='Strike Price', y_axis_label='Rate', plot_width=800, plot_height=300)
        p1 = p.line(source=src, x='close', y='10d', line_width=2, color='blue', alpha=0.5)
        p2 = p.line(source=src, x='close', y='20d', line_width=2, color='red', alpha=0.5)
        p3 = p.line(source=src, x='close', y='30d', line_width=2, color='green', alpha=0.5)
        p4 = p.line(source=src, x='close', y='60d', line_width=2, color='yellow', alpha=0.5)

        legend = Legend(items=[('IV10', [p1]), ('IV20', [p2]), ('IV30', [p3]), ('IV60', [p4])])
        p.add_layout(legend, 'right')
        p.legend.click_policy = 'hide'

        p.add_tools(HoverTool(renderers=[p1], tooltips=[('Strike Price', '@close'), ('IV10', '@10d{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p2], tooltips=[('Strike Price', '@close'), ('IV20', '@20d{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p3], tooltips=[('Strike Price', '@close'), ('IV30', '@30d{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p4], tooltips=[('Strike Price', '@close'), ('IV60', '@60d{%0.2F}')]))

        p.yaxis.formatter = NumeralTickFormatter(format='%F')

        st.bokeh_chart(p)

        if st.sidebar.checkbox('IV Surface', key='cf'):
            iv_surface('CF.ZCE.csv', '郑州棉花CF')

        if st.sidebar.checkbox('HV&IV', key='cf'):
            hiv('CF.ZCE')

    st.write('____________________________________________________________')


elif future == '铜':
    cu_show = data.iloc[3:4]
    cu_iv_show = data_iv.iloc[3:4]
    if st.sidebar.checkbox('Historical Volatility', key='cu'):
        st.subheader('沪铜CU近年历史波动率数据')
        st.dataframe(cu_show)

        # c K线图
        cu_data = pd.read_excel('d:/期货期权/处理数据/波动率/' + file[3], index_col=0)
        cu = cu_data[['trade_date', 'HV10', 'HV20', 'HV30', 'HV60']]
        cu['trade_date'] = pd.to_datetime(cu['trade_date'], format='%Y%m%d')
        cu['date_string'] = cu['trade_date'].dt.strftime('%Y/%m/%d')

        src = ColumnDataSource(cu)
        src.data.keys()

        p = figure(title=data['ts_code'][3] + ' HV', x_axis_label='Date', y_axis_label='Rate', x_axis_type='datetime', plot_width=800, plot_height=300)

        p1 = p.line(source=src, x='trade_date', y='HV10', line_width=2, color='blue', alpha=0.5)
        p2 = p.line(source=src, x='trade_date', y='HV20', line_width=2, color='red', alpha=0.5)
        p3 = p.line(source=src, x='trade_date', y='HV30', line_width=2, color='green', alpha=0.5)
        p4 = p.line(source=src, x='trade_date', y='HV60', line_width=2, color='yellow', alpha=0.5)

        legend = Legend(items=[('HV10', [p1]), ('HV20', [p2]), ('HV30', [p3]), ('HV60', [p4])])
        p.add_layout(legend, 'right')
        p.legend.click_policy = 'hide'

        p.add_tools(HoverTool(renderers=[p1], tooltips=[('Date', '@date_string'), ('HV10', '@HV10{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p2], tooltips=[('Date', '@date_string'), ('HV20', '@HV20{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p3], tooltips=[('Date', '@date_string'), ('HV30', '@HV30{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p4], tooltips=[('Date', '@date_string'), ('HV60', '@HV60{%0.2F}')]))

        p.xaxis.formatter = DatetimeTickFormatter(days=["%Y/%m/%d"], months=["%Y/%m/%d"], years=["%Y/%m/%d"])
        p.yaxis.formatter = NumeralTickFormatter(format='%F')

        st.bokeh_chart(p)

        # c 历史波动率锥图
        st.subheader('沪铜近年历史波动率锥')
        cu_awl = pd.read_excel('d:/期货期权/处理数据/波动率/' + file[3], index_col=0)
        cu_awl = pd.DataFrame({'HV': cu_awl['HV'][:8], '10分位': cu_awl['10分位'][:8], '25分位': cu_awl['25分位'][:8], '50分位': cu_awl['50分位'][:8], '75分位': cu_awl['75分位'][:8], '90分位': cu_awl['90分位'][:8]})
        cu_awl = cu_awl.set_index('HV')

        cu_awl['10分位'] = cu_awl['10分位'].apply(lambda x: '%.2f%%' % (x * 100))
        cu_awl['25分位'] = cu_awl['25分位'].apply(lambda x: '%.2f%%' % (x * 100))
        cu_awl['50分位'] = cu_awl['50分位'].apply(lambda x: '%.2f%%' % (x * 100))
        cu_awl['75分位'] = cu_awl['75分位'].apply(lambda x: '%.2f%%' % (x * 100))
        cu_awl['90分位'] = cu_awl['90分位'].apply(lambda x: '%.2f%%' % (x * 100))

        st.dataframe(cu_awl)

        st.line_chart(cu_awl)

    if st.sidebar.checkbox('Implied Volatility', key='cu'):
        st.subheader('沪铜CU近期隐含波动率数据')
        st.dataframe(cu_iv_show)

        cu_iv_info = pd.read_csv('d:/期货期权/隐含波动率/CU.SHF.csv', index_col=0)
        cu_iv_data = pd.read_csv(dirs_iv + file_iv[3], index_col=0)

        cu_iv_info['close'] = cu_iv_info['close'].iloc[0:len(cu_iv_data)]
        cu_iv_info = cu_iv_info.dropna(axis=0, how='any')
        cu_iv_data['close'] = cu_iv_info['close'].values

        src = ColumnDataSource(cu_iv_data)
        src.data.keys()

        p = figure(title=data_iv['ts_code'][3] + ' IV', x_axis_label='Strike Price', y_axis_label='Rate', plot_width=800, plot_height=300)
        p1 = p.line(source=src, x='close', y='10d', line_width=2, color='blue', alpha=0.5)
        p2 = p.line(source=src, x='close', y='20d', line_width=2, color='red', alpha=0.5)
        p3 = p.line(source=src, x='close', y='30d', line_width=2, color='green', alpha=0.5)
        p4 = p.line(source=src, x='close', y='60d', line_width=2, color='yellow', alpha=0.5)

        legend = Legend(items=[('IV10', [p1]), ('IV20', [p2]), ('IV30', [p3]), ('IV60', [p4])])
        p.add_layout(legend, 'right')
        p.legend.click_policy = 'hide'

        p.add_tools(HoverTool(renderers=[p1], tooltips=[('Strike Price', '@close'), ('IV10', '@10d{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p2], tooltips=[('Strike Price', '@close'), ('IV20', '@20d{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p3], tooltips=[('Strike Price', '@close'), ('IV30', '@30d{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p4], tooltips=[('Strike Price', '@close'), ('IV60', '@60d{%0.2F}')]))

        p.yaxis.formatter = NumeralTickFormatter(format='%F')

        st.bokeh_chart(p)

        if st.sidebar.checkbox('IV Surface', key='cu'):
            iv_surface('CU.SHF.csv', '沪铜CU')

        if st.sidebar.checkbox('HV&IV', key='cu'):
            hiv('CU.SHF')

    st.write('____________________________________________________________')


elif future == '铁矿石':
    i_show = data.iloc[4:5]
    i_iv_show = data_iv.iloc[4:5]
    if st.sidebar.checkbox('Historical Volatility', key='i'):
        st.subheader('大连铁矿石I近年历史波动率数据')
        st.dataframe(i_show)

        # c K线图
        i_data = pd.read_excel('d:/期货期权/处理数据/波动率/' + file[4], index_col=0)
        i = i_data[['trade_date', 'HV10', 'HV20', 'HV30', 'HV60']]
        i['trade_date'] = pd.to_datetime(i['trade_date'], format='%Y%m%d')
        i['date_string'] = i['trade_date'].dt.strftime('%Y/%m/%d')

        src = ColumnDataSource(i)
        src.data.keys()

        p = figure(title=data['ts_code'][4] + ' HV', x_axis_label='Date', y_axis_label='Rate', x_axis_type='datetime', plot_width=800, plot_height=300)

        p1 = p.line(source=src, x='trade_date', y='HV10', line_width=2, color='blue', alpha=0.5)
        p2 = p.line(source=src, x='trade_date', y='HV20', line_width=2, color='red', alpha=0.5)
        p3 = p.line(source=src, x='trade_date', y='HV30', line_width=2, color='green', alpha=0.5)
        p4 = p.line(source=src, x='trade_date', y='HV60', line_width=2, color='yellow', alpha=0.5)

        legend = Legend(items=[('HV10', [p1]), ('HV20', [p2]), ('HV30', [p3]), ('HV60', [p4])])
        p.add_layout(legend, 'right')
        p.legend.click_policy = 'hide'

        p.add_tools(HoverTool(renderers=[p1], tooltips=[('Date', '@date_string'), ('HV10', '@HV10{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p2], tooltips=[('Date', '@date_string'), ('HV20', '@HV20{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p3], tooltips=[('Date', '@date_string'), ('HV30', '@HV30{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p4], tooltips=[('Date', '@date_string'), ('HV60', '@HV60{%0.2F}')]))

        p.xaxis.formatter = DatetimeTickFormatter(days=["%Y/%m/%d"], months=["%Y/%m/%d"], years=["%Y/%m/%d"])
        p.yaxis.formatter = NumeralTickFormatter(format='%F')

        st.bokeh_chart(p)

        # c 历史波动率锥图
        st.subheader('大连铁矿石近年历史波动率锥')
        i_awl = pd.read_excel('d:/期货期权/处理数据/波动率/' + file[4], index_col=0)
        i_awl = pd.DataFrame({'HV': i_awl['HV'][:8], '10分位': i_awl['10分位'][:8], '25分位': i_awl['25分位'][:8], '50分位': i_awl['50分位'][:8], '75分位': i_awl['75分位'][:8], '90分位': i_awl['90分位'][:8]})
        i_awl = i_awl.set_index('HV')

        st.dataframe(i_awl)

        st.line_chart(i_awl)

    if st.sidebar.checkbox('Implied Volatility', key='i'):
        st.subheader('大连铁矿石I近期隐含波动率数据')
        st.dataframe(i_iv_show)

        i_iv_info = pd.read_csv('d:/期货期权/隐含波动率/I.DCE.csv', index_col=0)
        i_iv_data = pd.read_csv(dirs_iv + file_iv[4], index_col=0)

        i_iv_info['close'] = i_iv_info['close'].iloc[0:len(i_iv_data)]
        i_iv_info = i_iv_info.dropna(axis=0, how='any')
        i_iv_data['close'] = i_iv_info['close'].values

        src = ColumnDataSource(i_iv_data)
        src.data.keys()

        p = figure(title=data_iv['ts_code'][4] + ' IV', x_axis_label='Strike Price', y_axis_label='Rate', plot_width=800, plot_height=300)
        p1 = p.line(source=src, x='close', y='10d', line_width=2, color='blue', alpha=0.5)
        p2 = p.line(source=src, x='close', y='20d', line_width=2, color='red', alpha=0.5)
        p3 = p.line(source=src, x='close', y='30d', line_width=2, color='green', alpha=0.5)
        p4 = p.line(source=src, x='close', y='60d', line_width=2, color='yellow', alpha=0.5)

        legend = Legend(items=[('IV10', [p1]), ('IV20', [p2]), ('IV30', [p3]), ('IV60', [p4])])
        p.add_layout(legend, 'right')
        p.legend.click_policy = 'hide'

        p.add_tools(HoverTool(renderers=[p1], tooltips=[('Strike Price', '@close'), ('IV10', '@10d{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p2], tooltips=[('Strike Price', '@close'), ('IV20', '@20d{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p3], tooltips=[('Strike Price', '@close'), ('IV30', '@30d{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p4], tooltips=[('Strike Price', '@close'), ('IV60', '@60d{%0.2F}')]))

        p.yaxis.formatter = NumeralTickFormatter(format='%F')

        st.bokeh_chart(p)

        if st.sidebar.checkbox('IV Surface', key='i'):
            iv_surface('I.DCE.csv', '大连铁矿石I')

        if st.sidebar.checkbox('HV&IV', key='i'):
            hiv('I.DCE')

    st.write('____________________________________________________________')


elif future == '豆粕':
    m_show = data.iloc[5:6]
    m_iv_show = data_iv.iloc[5:6]
    if st.sidebar.checkbox('Historical Volatility', key='m'):
        st.subheader('大连豆粕M近年历史波动率数据')
        st.dataframe(m_show)

        # c K线图
        m_data = pd.read_excel('d:/期货期权/处理数据/波动率/' + file[5], index_col=0)
        m = m_data[['trade_date', 'HV10', 'HV20', 'HV30', 'HV60']]
        m['trade_date'] = pd.to_datetime(m['trade_date'], format='%Y%m%d')
        m['date_string'] = m['trade_date'].dt.strftime('%Y/%m/%d')

        src = ColumnDataSource(m)
        src.data.keys()

        p = figure(title=data['ts_code'][5] + ' HV', x_axis_label='Date', y_axis_label='Rate', x_axis_type='datetime', plot_width=800, plot_height=300)

        p1 = p.line(source=src, x='trade_date', y='HV10', line_width=2, color='blue', alpha=0.5)
        p2 = p.line(source=src, x='trade_date', y='HV20', line_width=2, color='red', alpha=0.5)
        p3 = p.line(source=src, x='trade_date', y='HV30', line_width=2, color='green', alpha=0.5)
        p4 = p.line(source=src, x='trade_date', y='HV60', line_width=2, color='yellow', alpha=0.5)

        legend = Legend(items=[('HV10', [p1]), ('HV20', [p2]), ('HV30', [p3]), ('HV60', [p4])])
        p.add_layout(legend, 'right')
        p.legend.click_policy = 'hide'

        p.add_tools(HoverTool(renderers=[p1], tooltips=[('Date', '@date_string'), ('HV10', '@HV10{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p2], tooltips=[('Date', '@date_string'), ('HV20', '@HV20{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p3], tooltips=[('Date', '@date_string'), ('HV30', '@HV30{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p4], tooltips=[('Date', '@date_string'), ('HV60', '@HV60{%0.2F}')]))

        p.xaxis.formatter = DatetimeTickFormatter(days=["%Y/%m/%d"], months=["%Y/%m/%d"], years=["%Y/%m/%d"])
        p.yaxis.formatter = NumeralTickFormatter(format='%F')

        st.bokeh_chart(p)

        # c 历史波动率锥图
        st.subheader('大连豆粕近年历史波动率锥')
        m_awl = pd.read_excel('d:/期货期权/处理数据/波动率/' + file[5], index_col=0)
        m_awl = pd.DataFrame({'HV': m_awl['HV'][:8], '10分位': m_awl['10分位'][:8], '25分位': m_awl['25分位'][:8], '50分位': m_awl['50分位'][:8], '75分位': m_awl['75分位'][:8], '90分位': m_awl['90分位'][:8]})
        m_awl = m_awl.set_index('HV')

        st.dataframe(m_awl)

        st.line_chart(m_awl)

    if st.sidebar.checkbox('Implied Volatility', key='m'):
        st.subheader('大连豆粕M近期隐含波动率数据')
        st.dataframe(m_iv_show)

        m_iv_info = pd.read_csv('d:/期货期权/隐含波动率/M.DCE.csv', index_col=0)
        m_iv_data = pd.read_csv(dirs_iv + file_iv[5], index_col=0)

        m_iv_info['close'] = m_iv_info['close'].iloc[0:len(m_iv_data)]
        m_iv_info = m_iv_info.dropna(axis=0, how='any')
        m_iv_data['close'] = m_iv_info['close'].values

        src = ColumnDataSource(m_iv_data)
        src.data.keys()

        p = figure(title=data_iv['ts_code'][5] + ' IV', x_axis_label='Strike Price', y_axis_label='Rate', plot_width=800, plot_height=300)
        p1 = p.line(source=src, x='close', y='10d', line_width=2, color='blue', alpha=0.5)
        p2 = p.line(source=src, x='close', y='20d', line_width=2, color='red', alpha=0.5)
        p3 = p.line(source=src, x='close', y='30d', line_width=2, color='green', alpha=0.5)
        p4 = p.line(source=src, x='close', y='60d', line_width=2, color='yellow', alpha=0.5)

        legend = Legend(items=[('IV10', [p1]), ('IV20', [p2]), ('IV30', [p3]), ('IV60', [p4])])
        p.add_layout(legend, 'right')
        p.legend.click_policy = 'hide'

        p.add_tools(HoverTool(renderers=[p1], tooltips=[('Strike Price', '@close'), ('IV10', '@10d{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p2], tooltips=[('Strike Price', '@close'), ('IV20', '@20d{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p3], tooltips=[('Strike Price', '@close'), ('IV30', '@30d{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p4], tooltips=[('Strike Price', '@close'), ('IV60', '@60d{%0.2F}')]))

        p.yaxis.formatter = NumeralTickFormatter(format='%F')

        st.bokeh_chart(p)

        if st.sidebar.checkbox('IV Surface', key='m'):
            iv_surface('M.DCE.csv', '大连豆粕M')

        if st.sidebar.checkbox('HV&IV', key='m'):
            hiv('M.DCE')

    st.write('____________________________________________________________')


elif future == '甲醇':
    ma_show = data.iloc[6:7]
    ma_iv_show = data_iv.iloc[6:7]
    if st.sidebar.checkbox('Historical Volatility', key='ma'):
        st.subheader('郑州甲醇MA近年历史波动率数据')
        st.dataframe(ma_show)

        # c K线图
        ma_data = pd.read_excel('d:/期货期权/处理数据/波动率/' + file[6], index_col=0)
        ma = ma_data[['trade_date', 'HV10', 'HV20', 'HV30', 'HV60']]
        ma['trade_date'] = pd.to_datetime(ma['trade_date'], format='%Y%m%d')
        ma['date_string'] = ma['trade_date'].dt.strftime('%Y/%m/%d')

        src = ColumnDataSource(ma)
        src.data.keys()

        p = figure(title=data['ts_code'][6] + ' HV', x_axis_label='Date', y_axis_label='Rate', x_axis_type='datetime', plot_width=800, plot_height=300)

        p1 = p.line(source=src, x='trade_date', y='HV10', line_width=2, color='blue', alpha=0.5)
        p2 = p.line(source=src, x='trade_date', y='HV20', line_width=2, color='red', alpha=0.5)
        p3 = p.line(source=src, x='trade_date', y='HV30', line_width=2, color='green', alpha=0.5)
        p4 = p.line(source=src, x='trade_date', y='HV60', line_width=2, color='yellow', alpha=0.5)

        legend = Legend(items=[('HV10', [p1]), ('HV20', [p2]), ('HV30', [p3]), ('HV60', [p4])])
        p.add_layout(legend, 'right')
        p.legend.click_policy = 'hide'

        p.add_tools(HoverTool(renderers=[p1], tooltips=[('Date', '@date_string'), ('HV10', '@HV10{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p2], tooltips=[('Date', '@date_string'), ('HV20', '@HV20{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p3], tooltips=[('Date', '@date_string'), ('HV30', '@HV30{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p4], tooltips=[('Date', '@date_string'), ('HV60', '@HV60{%0.2F}')]))

        p.xaxis.formatter = DatetimeTickFormatter(days=["%Y/%m/%d"], months=["%Y/%m/%d"], years=["%Y/%m/%d"])
        p.yaxis.formatter = NumeralTickFormatter(format='%F')

        st.bokeh_chart(p)

        # c 历史波动率锥图
        st.subheader('郑州甲醇近年历史波动率锥')
        ma_awl = pd.read_excel('d:/期货期权/处理数据/波动率/' + file[6], index_col=0)
        ma_awl = pd.DataFrame({'HV': ma_awl['HV'][:8], '10分位': ma_awl['10分位'][:8], '25分位': ma_awl['25分位'][:8], '50分位': ma_awl['50分位'][:8], '75分位': ma_awl['75分位'][:8], '90分位': ma_awl['90分位'][:8]})
        ma_awl = ma_awl.set_index('HV')

        st.dataframe(ma_awl)

        st.line_chart(ma_awl)

    if st.sidebar.checkbox('Implied Volatility', key='ma'):
        st.subheader('郑州甲醇MA近期隐含波动率数据')
        st.dataframe(ma_iv_show)

        ma_iv_info = pd.read_csv('d:/期货期权/隐含波动率/MA.ZCE.csv', index_col=0)
        ma_iv_data = pd.read_csv(dirs_iv + file_iv[6], index_col=0)

        ma_iv_info['close'] = ma_iv_info['close'].iloc[0:len(ma_iv_data)]
        ma_iv_info = ma_iv_info.dropna(axis=0, how='any')
        ma_iv_data['close'] = ma_iv_info['close'].values

        src = ColumnDataSource(ma_iv_data)
        src.data.keys()

        p = figure(title=data_iv['ts_code'][6] + ' IV', x_axis_label='Strike Price', y_axis_label='Rate', plot_width=800, plot_height=300)
        p1 = p.line(source=src, x='close', y='10d', line_width=2, color='blue', alpha=0.5)
        p2 = p.line(source=src, x='close', y='20d', line_width=2, color='red', alpha=0.5)
        p3 = p.line(source=src, x='close', y='30d', line_width=2, color='green', alpha=0.5)
        p4 = p.line(source=src, x='close', y='60d', line_width=2, color='yellow', alpha=0.5)

        legend = Legend(items=[('IV10', [p1]), ('IV20', [p2]), ('IV30', [p3]), ('IV60', [p4])])
        p.add_layout(legend, 'right')
        p.legend.click_policy = 'hide'

        p.add_tools(HoverTool(renderers=[p1], tooltips=[('Strike Price', '@close'), ('IV10', '@10d{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p2], tooltips=[('Strike Price', '@close'), ('IV20', '@20d{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p3], tooltips=[('Strike Price', '@close'), ('IV30', '@30d{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p4], tooltips=[('Strike Price', '@close'), ('IV60', '@60d{%0.2F}')]))

        p.yaxis.formatter = NumeralTickFormatter(format='%F')

        st.bokeh_chart(p)

        if st.sidebar.checkbox('IV Surface', key='ma'):
            iv_surface('MA.ZCE.csv', '郑州甲醇MA')

        if st.sidebar.checkbox('HV&IV', key='ma'):
            hiv('MA.ZCE')

    st.write('____________________________________________________________')


elif future == '菜粕':
    rm_show = data.iloc[7:8]
    rm_iv_show = data_iv.iloc[7:8]
    if st.sidebar.checkbox('Historical Volatility', key='rm'):
        st.subheader('郑州菜粕RM近年历史波动率数据')
        st.dataframe(rm_show)

        # c K线图
        rm_data = pd.read_excel('d:/期货期权/处理数据/波动率/' + file[7], index_col=0)
        rm = rm_data[['trade_date', 'HV10', 'HV20', 'HV30', 'HV60']]
        rm['trade_date'] = pd.to_datetime(rm['trade_date'], format='%Y%m%d')
        rm['date_string'] = rm['trade_date'].dt.strftime('%Y/%m/%d')

        src = ColumnDataSource(rm)
        src.data.keys()

        p = figure(title=data['ts_code'][7] + ' HV', x_axis_label='Date', y_axis_label='Rate', x_axis_type='datetime', plot_width=800, plot_height=300)

        p1 = p.line(source=src, x='trade_date', y='HV10', line_width=2, color='blue', alpha=0.5)
        p2 = p.line(source=src, x='trade_date', y='HV20', line_width=2, color='red', alpha=0.5)
        p3 = p.line(source=src, x='trade_date', y='HV30', line_width=2, color='green', alpha=0.5)
        p4 = p.line(source=src, x='trade_date', y='HV60', line_width=2, color='yellow', alpha=0.5)

        legend = Legend(items=[('HV10', [p1]), ('HV20', [p2]), ('HV30', [p3]), ('HV60', [p4])])
        p.add_layout(legend, 'right')
        p.legend.click_policy = 'hide'

        p.add_tools(HoverTool(renderers=[p1], tooltips=[('Date', '@date_string'), ('HV10', '@HV10{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p2], tooltips=[('Date', '@date_string'), ('HV20', '@HV20{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p3], tooltips=[('Date', '@date_string'), ('HV30', '@HV30{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p4], tooltips=[('Date', '@date_string'), ('HV60', '@HV60{%0.2F}')]))

        p.xaxis.formatter = DatetimeTickFormatter(days=["%Y/%m/%d"], months=["%Y/%m/%d"], years=["%Y/%m/%d"])
        p.yaxis.formatter = NumeralTickFormatter(format='%F')

        st.bokeh_chart(p)

        # c 历史波动率锥图
        st.subheader('郑州菜粕近年历史波动率锥')
        rm_awl = pd.read_excel('d:/期货期权/处理数据/波动率/' + file[7], index_col=0)
        rm_awl = pd.DataFrame({'HV': rm_awl['HV'][:8], '10分位': rm_awl['10分位'][:8], '25分位': rm_awl['25分位'][:8], '50分位': rm_awl['50分位'][:8], '75分位': rm_awl['75分位'][:8], '90分位': rm_awl['90分位'][:8]})
        rm_awl = rm_awl.set_index('HV')

        st.dataframe(rm_awl)

        st.line_chart(rm_awl)

    if st.sidebar.checkbox('Implied Volatility', key='rm'):
        st.subheader('郑州菜粕RM近期隐含波动率数据')
        st.dataframe(rm_iv_show)

        rm_iv_info = pd.read_csv('d:/期货期权/隐含波动率/RM.ZCE.csv', index_col=0)
        rm_iv_data = pd.read_csv(dirs_iv + file_iv[7], index_col=0)

        rm_iv_info['close'] = rm_iv_info['close'].iloc[0:len(rm_iv_data)]
        rm_iv_info = rm_iv_info.dropna(axis=0, how='any')
        rm_iv_data['close'] = rm_iv_info['close'].values

        src = ColumnDataSource(rm_iv_data)
        src.data.keys()

        p = figure(title=data_iv['ts_code'][7] + ' IV', x_axis_label='Strike Price', y_axis_label='Rate', plot_width=800, plot_height=300)
        p1 = p.line(source=src, x='close', y='10d', line_width=2, color='blue', alpha=0.5)
        p2 = p.line(source=src, x='close', y='20d', line_width=2, color='red', alpha=0.5)
        p3 = p.line(source=src, x='close', y='30d', line_width=2, color='green', alpha=0.5)
        p4 = p.line(source=src, x='close', y='60d', line_width=2, color='yellow', alpha=0.5)

        legend = Legend(items=[('IV10', [p1]), ('IV20', [p2]), ('IV30', [p3]), ('IV60', [p4])])
        p.add_layout(legend, 'right')
        p.legend.click_policy = 'hide'

        p.add_tools(HoverTool(renderers=[p1], tooltips=[('Strike Price', '@close'), ('IV10', '@10d{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p2], tooltips=[('Strike Price', '@close'), ('IV20', '@20d{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p3], tooltips=[('Strike Price', '@close'), ('IV30', '@30d{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p4], tooltips=[('Strike Price', '@close'), ('IV60', '@60d{%0.2F}')]))

        p.yaxis.formatter = NumeralTickFormatter(format='%F')

        st.bokeh_chart(p)

        if st.sidebar.checkbox('IV Surface', key='rm'):
            iv_surface('RM.ZCE.csv', '郑州菜粕RM')

        if st.sidebar.checkbox('HV&IV', key='rm'):
            hiv('RM.ZCE')

    st.write('____________________________________________________________')


elif future == '橡胶':
    ru_show = data.iloc[8:9]
    ru_iv_show = data_iv.iloc[8:9]
    if st.sidebar.checkbox('Historical Volatility', key='ru'):
        st.subheader('沪橡胶RU近年历史波动率数据')
        st.dataframe(ru_show)

        # c K线图
        ru_data = pd.read_excel('d:/期货期权/处理数据/波动率/' + file[8], index_col=0)
        ru = ru_data[['trade_date', 'HV10', 'HV20', 'HV30', 'HV60']]
        ru['trade_date'] = pd.to_datetime(ru['trade_date'], format='%Y%m%d')
        ru['date_string'] = ru['trade_date'].dt.strftime('%Y/%m/%d')

        src = ColumnDataSource(ru)
        src.data.keys()

        p = figure(title=data['ts_code'][8] + ' HV', x_axis_label='Date', y_axis_label='Rate', x_axis_type='datetime', plot_width=800, plot_height=300)

        p1 = p.line(source=src, x='trade_date', y='HV10', line_width=2, color='blue', alpha=0.5)
        p2 = p.line(source=src, x='trade_date', y='HV20', line_width=2, color='red', alpha=0.5)
        p3 = p.line(source=src, x='trade_date', y='HV30', line_width=2, color='green', alpha=0.5)
        p4 = p.line(source=src, x='trade_date', y='HV60', line_width=2, color='yellow', alpha=0.5)

        legend = Legend(items=[('HV10', [p1]), ('HV20', [p2]), ('HV30', [p3]), ('HV60', [p4])])
        p.add_layout(legend, 'right')
        p.legend.click_policy = 'hide'

        p.add_tools(HoverTool(renderers=[p1], tooltips=[('Date', '@date_string'), ('HV10', '@HV10{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p2], tooltips=[('Date', '@date_string'), ('HV20', '@HV20{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p3], tooltips=[('Date', '@date_string'), ('HV30', '@HV30{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p4], tooltips=[('Date', '@date_string'), ('HV60', '@HV60{%0.2F}')]))

        p.xaxis.formatter = DatetimeTickFormatter(days=["%Y/%m/%d"], months=["%Y/%m/%d"], years=["%Y/%m/%d"])
        p.yaxis.formatter = NumeralTickFormatter(format='%F')

        st.bokeh_chart(p)

        # c 历史波动率锥图
        st.subheader('沪橡胶近年历史波动率锥')
        ru_awl = pd.read_excel('d:/期货期权/处理数据/波动率/' + file[8], index_col=0)
        ru_awl = pd.DataFrame({'HV': ru_awl['HV'][:8], '10分位': ru_awl['10分位'][:8], '25分位': ru_awl['25分位'][:8], '50分位': ru_awl['50分位'][:8], '75分位': ru_awl['75分位'][:8], '90分位': ru_awl['90分位'][:8]})
        ru_awl = ru_awl.set_index('HV')

        st.dataframe(ru_awl)

        st.line_chart(ru_awl)

    if st.sidebar.checkbox('Implied Volatility', key='ru'):
        st.subheader('沪橡胶RU近期隐含波动率数据')
        st.dataframe(ru_iv_show)

        ru_iv_info = pd.read_csv('d:/期货期权/隐含波动率/RU.SHF.csv', index_col=0)
        ru_iv_data = pd.read_csv(dirs_iv + file_iv[8], index_col=0)

        ru_iv_info['close'] = ru_iv_info['close'].iloc[0:len(ru_iv_data)]
        ru_iv_info = ru_iv_info.dropna(axis=0, how='any')
        ru_iv_data['close'] = ru_iv_info['close'].values

        src = ColumnDataSource(ru_iv_data)
        src.data.keys()

        p = figure(title=data_iv['ts_code'][8] + ' IV', x_axis_label='Strike Price', y_axis_label='Rate', plot_width=800, plot_height=300)
        p1 = p.line(source=src, x='close', y='10d', line_width=2, color='blue', alpha=0.5)
        p2 = p.line(source=src, x='close', y='20d', line_width=2, color='red', alpha=0.5)
        p3 = p.line(source=src, x='close', y='30d', line_width=2, color='green', alpha=0.5)
        p4 = p.line(source=src, x='close', y='60d', line_width=2, color='yellow', alpha=0.5)

        legend = Legend(items=[('IV10', [p1]), ('IV20', [p2]), ('IV30', [p3]), ('IV60', [p4])])
        p.add_layout(legend, 'right')
        p.legend.click_policy = 'hide'

        p.add_tools(HoverTool(renderers=[p1], tooltips=[('Strike Price', '@close'), ('IV10', '@10d{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p2], tooltips=[('Strike Price', '@close'), ('IV20', '@20d{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p3], tooltips=[('Strike Price', '@close'), ('IV30', '@30d{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p4], tooltips=[('Strike Price', '@close'), ('IV60', '@60d{%0.2F}')]))

        p.yaxis.formatter = NumeralTickFormatter(format='%F')

        st.bokeh_chart(p)

        if st.sidebar.checkbox('IV Surface', key='ru'):
            iv_surface('RU.SHF.csv', '沪橡胶RU')

        if st.sidebar.checkbox('HV&IV', key='ru'):
            hiv('RU.SHF')

    st.write('____________________________________________________________')


elif future == '白糖':
    sr_show = data.iloc[9:10]
    sr_iv_show = data_iv.iloc[9:10]
    if st.sidebar.checkbox('Historical Volatility', key='sr'):
        st.subheader('郑州白糖SR近年历史波动率数据')
        st.dataframe(sr_show)

        # c K线图
        sr_data = pd.read_excel('d:/期货期权/处理数据/波动率/' + file[9], index_col=0)
        sr = sr_data[['trade_date', 'HV10', 'HV20', 'HV30', 'HV60']]
        sr['trade_date'] = pd.to_datetime(sr['trade_date'], format='%Y%m%d')
        sr['date_string'] = sr['trade_date'].dt.strftime('%Y/%m/%d')

        src = ColumnDataSource(sr)
        src.data.keys()

        p = figure(title=data['ts_code'][9] + ' HV', x_axis_label='Date', y_axis_label='Rate', x_axis_type='datetime', plot_width=800, plot_height=300)

        p1 = p.line(source=src, x='trade_date', y='HV10', line_width=2, color='blue', alpha=0.5)
        p2 = p.line(source=src, x='trade_date', y='HV20', line_width=2, color='red', alpha=0.5)
        p3 = p.line(source=src, x='trade_date', y='HV30', line_width=2, color='green', alpha=0.5)
        p4 = p.line(source=src, x='trade_date', y='HV60', line_width=2, color='yellow', alpha=0.5)

        legend = Legend(items=[('HV10', [p1]), ('HV20', [p2]), ('HV30', [p3]), ('HV60', [p4])])
        p.add_layout(legend, 'right')
        p.legend.click_policy = 'hide'

        p.add_tools(HoverTool(renderers=[p1], tooltips=[('Date', '@date_string'), ('HV10', '@HV10{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p2], tooltips=[('Date', '@date_string'), ('HV20', '@HV20{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p3], tooltips=[('Date', '@date_string'), ('HV30', '@HV30{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p4], tooltips=[('Date', '@date_string'), ('HV60', '@HV60{%0.2F}')]))

        p.xaxis.formatter = DatetimeTickFormatter(days=["%Y/%m/%d"], months=["%Y/%m/%d"], years=["%Y/%m/%d"])
        p.yaxis.formatter = NumeralTickFormatter(format='%F')

        st.bokeh_chart(p)

        # c 历史波动率锥图
        st.subheader('郑州白糖近年历史波动率锥')
        sr_awl = pd.read_excel('d:/期货期权/处理数据/波动率/' + file[9], index_col=0)
        sr_awl = pd.DataFrame({'HV': sr_awl['HV'][:8], '10分位': sr_awl['10分位'][:8], '25分位': sr_awl['25分位'][:8], '50分位': sr_awl['50分位'][:8], '75分位': sr_awl['75分位'][:8], '90分位': sr_awl['90分位'][:8]})
        sr_awl = sr_awl.set_index('HV')

        st.dataframe(sr_awl)

        st.line_chart(sr_awl)

    if st.sidebar.checkbox('Implied Volatility', key='sr'):
        st.subheader('郑州白糖SR近期隐含波动率数据')
        st.dataframe(sr_iv_show)

        sr_iv_info = pd.read_csv('d:/期货期权/隐含波动率/SR.ZCE.csv', index_col=0)
        sr_iv_data = pd.read_csv(dirs_iv + file_iv[9], index_col=0)

        sr_iv_info['close'] = sr_iv_info['close'].iloc[0:len(sr_iv_data)]
        sr_iv_info = sr_iv_info.dropna(axis=0, how='any')
        sr_iv_data['close'] = sr_iv_info['close'].values

        src = ColumnDataSource(sr_iv_data)
        src.data.keys()

        p = figure(title=data_iv['ts_code'][9] + ' IV', x_axis_label='Strike Price', y_axis_label='Rate', plot_width=800, plot_height=300)
        p1 = p.line(source=src, x='close', y='10d', line_width=2, color='blue', alpha=0.5)
        p2 = p.line(source=src, x='close', y='20d', line_width=2, color='red', alpha=0.5)
        p3 = p.line(source=src, x='close', y='30d', line_width=2, color='green', alpha=0.5)
        p4 = p.line(source=src, x='close', y='60d', line_width=2, color='yellow', alpha=0.5)

        legend = Legend(items=[('IV10', [p1]), ('IV20', [p2]), ('IV30', [p3]), ('IV60', [p4])])
        p.add_layout(legend, 'right')
        p.legend.click_policy = 'hide'

        p.add_tools(HoverTool(renderers=[p1], tooltips=[('Strike Price', '@close'), ('IV10', '@10d{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p2], tooltips=[('Strike Price', '@close'), ('IV20', '@20d{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p3], tooltips=[('Strike Price', '@close'), ('IV30', '@30d{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p4], tooltips=[('Strike Price', '@close'), ('IV60', '@60d{%0.2F}')]))

        p.yaxis.formatter = NumeralTickFormatter(format='%F')

        st.bokeh_chart(p)

        if st.sidebar.checkbox('IV Surface', key='sr'):
            iv_surface('SR.ZCE.csv', '郑州白糖SR')

        if st.sidebar.checkbox('HV&IV', key='sr'):
            hiv('SR.ZCE')

    st.write('____________________________________________________________')


elif future == 'PTA':
    ta_show = data.iloc[10:11]
    ta_iv_show = data_iv.iloc[10:11]
    if st.sidebar.checkbox('Historical Volatility', key='ta'):
        st.subheader('郑州PTA近年历史波动率数据')
        st.dataframe(ta_show)

        # c K线图
        ta_data = pd.read_excel('d:/期货期权/处理数据/波动率/' + file[10], index_col=0)
        ta = ta_data[['trade_date', 'HV10', 'HV20', 'HV30', 'HV60']]
        ta['trade_date'] = pd.to_datetime(ta['trade_date'], format='%Y%m%d')
        ta['date_string'] = ta['trade_date'].dt.strftime('%Y/%m/%d')

        src = ColumnDataSource(ta)
        src.data.keys()

        p = figure(title=data['ts_code'][10] + ' HV', x_axis_label='Date', y_axis_label='Rate', x_axis_type='datetime', plot_width=800, plot_height=300)

        p1 = p.line(source=src, x='trade_date', y='HV10', line_width=2, color='blue', alpha=0.5)
        p2 = p.line(source=src, x='trade_date', y='HV20', line_width=2, color='red', alpha=0.5)
        p3 = p.line(source=src, x='trade_date', y='HV30', line_width=2, color='green', alpha=0.5)
        p4 = p.line(source=src, x='trade_date', y='HV60', line_width=2, color='yellow', alpha=0.5)

        legend = Legend(items=[('HV10', [p1]), ('HV20', [p2]), ('HV30', [p3]), ('HV60', [p4])])
        p.add_layout(legend, 'right')
        p.legend.click_policy = 'hide'

        p.add_tools(HoverTool(renderers=[p1], tooltips=[('Date', '@date_string'), ('HV10', '@HV10{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p2], tooltips=[('Date', '@date_string'), ('HV20', '@HV20{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p3], tooltips=[('Date', '@date_string'), ('HV30', '@HV30{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p4], tooltips=[('Date', '@date_string'), ('HV60', '@HV60{%0.2F}')]))

        p.xaxis.formatter = DatetimeTickFormatter(days=["%Y/%m/%d"], months=["%Y/%m/%d"], years=["%Y/%m/%d"])
        p.yaxis.formatter = NumeralTickFormatter(format='%F')

        st.bokeh_chart(p)

        # c 历史波动率锥图
        st.subheader('郑州PTA近年历史波动率锥')
        ta_awl = pd.read_excel('d:/期货期权/处理数据/波动率/' + file[10], index_col=0)
        ta_awl = pd.DataFrame({'HV': ta_awl['HV'][:8], '10分位': ta_awl['10分位'][:8], '25分位': ta_awl['25分位'][:8], '50分位': ta_awl['50分位'][:8], '75分位': ta_awl['75分位'][:8], '90分位': ta_awl['90分位'][:8]})
        ta_awl = ta_awl.set_index('HV')

        st.dataframe(ta_awl)

        st.line_chart(ta_awl)

    if st.sidebar.checkbox('Implied Volatility', key='ta'):
        st.subheader('郑州PTA近期隐含波动率数据')
        st.dataframe(ta_iv_show)

        ta_iv_info = pd.read_csv('d:/期货期权/隐含波动率/TA.ZCE.csv', index_col=0)
        ta_iv_data = pd.read_csv(dirs_iv + file_iv[10], index_col=0)

        ta_iv_info['close'] = ta_iv_info['close'].iloc[0:len(ta_iv_data)]
        ta_iv_info = ta_iv_info.dropna(axis=0, how='any')
        ta_iv_data['close'] = ta_iv_info['close'].values

        src = ColumnDataSource(ta_iv_data)
        src.data.keys()

        p = figure(title=data_iv['ts_code'][10] + ' IV', x_axis_label='Strike Price', y_axis_label='Rate', plot_width=800, plot_height=300)
        p1 = p.line(source=src, x='close', y='10d', line_width=2, color='blue', alpha=0.5)
        p2 = p.line(source=src, x='close', y='20d', line_width=2, color='red', alpha=0.5)
        p3 = p.line(source=src, x='close', y='30d', line_width=2, color='green', alpha=0.5)
        p4 = p.line(source=src, x='close', y='60d', line_width=2, color='yellow', alpha=0.5)

        legend = Legend(items=[('IV10', [p1]), ('IV20', [p2]), ('IV30', [p3]), ('IV60', [p4])])
        p.add_layout(legend, 'right')
        p.legend.click_policy = 'hide'

        p.add_tools(HoverTool(renderers=[p1], tooltips=[('Strike Price', '@close'), ('IV10', '@10d{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p2], tooltips=[('Strike Price', '@close'), ('IV20', '@20d{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p3], tooltips=[('Strike Price', '@close'), ('IV30', '@30d{%0.2F}')]))
        p.add_tools(HoverTool(renderers=[p4], tooltips=[('Strike Price', '@close'), ('IV60', '@60d{%0.2F}')]))

        p.yaxis.formatter = NumeralTickFormatter(format='%F')

        st.bokeh_chart(p)

        if st.sidebar.checkbox('IV Surface', key='ta'):
            iv_surface('TA.ZCE.csv', '郑州PTA')

        if st.sidebar.checkbox('HV&IV', key='ta'):
            hiv('TA.ZCE')

    st.write('____________________________________________________________')
