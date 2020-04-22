import wx
import wx.adv
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.backends import backend_wxagg
import matplotlib.gridspec as gridspec
matplotlib.use("WXAgg")
import pandas as pd
import mpl_finance as mpf


df = pd.read_csv('./C.csv')
# df_sr


class Frame(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, parent=None, title=u'量化', size=(1000, 600),
                          style=wx.DEFAULT_FRAME_STYLE ^ wx.MAXIMIZE_BOX)

        self.DisPanel = wx.Panel(self)

        self.ParaPanel = wx.Panel(self, -1)
        paraInput_Box = wx.StaticBox(self.ParaPanel, -1, u'参数输入')
        paraInput_Sizer = wx.StaticBoxSizer(paraInput_Box, wx.VERTICAL)
        Opt_Name_ComboBox = list(df['代码'][0:3])
        global optName_CMBO
        optName_CMBO = wx.ComboBox(self.ParaPanel, -1, df['代码'][0], choices=Opt_Name_ComboBox,
                                   style=wx.CB_READONLY | wx.CB_DROPDOWN)

        optName_CMBO.Bind(wx.EVT_COMBOBOX, self.OnCombo)

        optCode_Text = wx.StaticText(self.ParaPanel, -1, u'玉米')

        paraInput_Sizer.Add(optCode_Text, proportion=0, flag=wx.EXPAND | wx.ALL, border=2)
        paraInput_Sizer.Add(optName_CMBO, 0, wx.EXPAND | wx.ALL | wx.CENTER, 2)

        vboxnetA = wx.BoxSizer(wx.VERTICAL)
        vboxnetA.Add(paraInput_Sizer, proportion=0, flag=wx.EXPAND | wx.BOTTOM, border=2)

        self.ParaPanel.SetSizer(vboxnetA)

        self.FlexGridSizer = wx.FlexGridSizer(rows=3, cols=1, vgap=3, hgap=3)

        self.FlexGridSizer.SetFlexibleDirection(wx.BOTH)

        self.HBoxPanel = wx.BoxSizer(wx.HORIZONTAL)
        self.HBoxPanel.Add(self.ParaPanel, proportion=1, border=2, flag=wx.EXPAND | wx.ALL)
        self.HBoxPanel.Add(self.DisPanel, proportion=4, border=2, flag=wx.EXPAND | wx.ALL)

        self.SetSizer(self.HBoxPanel)
        #self.label = wx.StaticText(self.DisPanel, label="Your choice:", style=wx.ALIGN_CENTRE)

    def OnCombo(self, event):
        #self.label.SetLabel("You selected" + optName_CMBO.GetValue() + " from Combobox")
        self.pic = Figure()
        gs = gridspec.GridSpec(4, 1, left=0.10, bottom=0.10, right=0.96, top=0.96, wspace=None, hspace=0.1,
                               height_ratios=[3.5, 1, 1, 1])
        #self.am = self.figure.add_subplot(gs[0, :])
        data_K = pd.read_excel('./' + optName_CMBO.GetValue() + '.xlsx')
        data_K = data_K[~data_K['open'].isin([0])]
        self.graph_KAV = self.pic.add_subplot(gs[0, :])
        self.graph_KAV.cla()
        mpf.candlestick2_ochl(self.graph_KAV, data_K.open, data_K.close, data_K.high, data_K.low, width=0.5, colorup='r', colordown='g')
        self.graph_KAV.set_xlim(0, len(data_K.index))
        self.graph_KAV.set_xticks(range(0, len(data_K.trade_date), 15))
        self.graph_KAV.grid(True, color='k')

        data = pd.read_excel('./' + optName_CMBO.GetValue() + '.xlsx')
        data['trade_date'] = pd.to_datetime(data['trade_date'], format='%Y%m%d')
        self.axes = self.pic.add_subplot(gs[1, :])
        self.axes.cla()
        self.axes.plot(data['trade_date'], data['hv20'], color='red', label='HV20')
        self.axes.plot(data['trade_date'], data['hv30'], color='blue', label='HV30')
        self.axes.plot(data['trade_date'], data['hv60'], color='green', label='HV60')
        self.axes.grid(True)
        self.axes.set_xlabel('Date')
        self.axes.set_ylabel('Rate')
        self.axes.legend(loc='best', shadow=True, fontsize='10')
        self.axes.tick_params(labelsize=5)

        data2 = pd.read_csv('d:/期货期权/隐含波动率/相关数据/C.DCE.csv', index_col=0)
        data2_1 = pd.read_csv('d:/期货期权/隐含波动率/C.DCE.csv', index_col=0)
        data2['close'] = data2_1['close'].iloc[0:len(data2)]
        data2_1 = data2_1.dropna(axis=0, how='any')
        data2['close'] = data2_1['close'].values

        self.axes2 = self.pic.add_subplot(gs[3, :])
        self.axes2.cla()
        self.axes2.plot(data2['close'], data2['20d'], color='red', label='IV20')
        self.axes2.plot(data2['close'], data2['30d'], color='green', label='IV30')
        self.axes2.plot(data2['close'], data2['60d'], color='blue', label='IV60')
        self.axes2.grid(True)
        self.axes2.set_xlabel('Close')
        self.axes2.set_ylabel('Rate')
        self.axes2.legend(loc='best', shadow=True, fontsize='10')
        #self.axes2.tick_params(labelsize=5)
        canvas = FigureCanvas(self.DisPanel, -1, self.pic)


class App(wx.App):
    def OnInit(self):
        self.frame = Frame()
        self.frame.Show()
        self.SetTopWindow(self.frame)
        return True


if __name__ == '__main__':
    app = App()
    app.MainLoop()
