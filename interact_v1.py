#导入工具包
# Streamlit > ~1.12
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx as get_report_ctx
import json
import time
from pyecharts.globals import ThemeType
import streamlit.components.v1 as components
from pyecharts import options as opts
from pyecharts.charts import *
from pyecharts.commons.utils import JsCode
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn import metrics  
from sklearn.ensemble import GradientBoostingRegressor
from streamlit_echarts import st_pyecharts
from pyecharts.charts import HeatMap

data_path = r'./beijing.csv'

@st.cache_data
def process_data(file_path):
    #导入数据集文件
    data = pd.read_csv(file_path)
    #删除No这列无关变量
    data.drop(['No'], axis = 1, inplace = True)
    # 缺失值处理
    # 查看数据集中的特征缺失情况  
    missing=data.isnull().sum().reset_index().rename(columns={0:'missNum'})
    # 计算缺失比例
    missing['missRate']=missing['missNum']/data.shape[0]
    # 填充缺失值为后一个非缺失值
    data['pm2.5'].fillna(data['pm2.5'].bfill(), inplace=True)
    
    return data

@st.cache_data
def encode_categorical(data):
    # 对cbwd这列离散型变量one_hot编码
    dtypes_list = data.dtypes.values
    columns_list = data.columns
    for i in range(len(columns_list)):
        if dtypes_list[i] == 'object':  # 编码为数字
            lb = LabelEncoder()
            lb.fit(data[columns_list[i]])
            data[columns_list[i]] = lb.transform(data[columns_list[i]])
    return data

@st.cache_data
def split_data(data):
    # 按照7:3的比列将数据随机划分为训练集和测试集
    feature_data1 = data.loc[:, 'year':'hour']
    feature_data2 = data.loc[:, 'DEWP':'Ir']
    feature_data = pd.concat([feature_data1, feature_data2], axis=1)
    target = data['pm2.5']
    train_X, test_X, train_Y, test_Y = train_test_split(feature_data, target, test_size=0.3, random_state=0)
    return train_X, test_X, train_Y, test_Y

@st.cache_data
def correlation_heatmap(train_X):
    # spearman单因素分析查看各变量之间的相关性
    corr = train_X.corr(method='spearman')
    return corr

@st.cache_data
def process_data_with_spearman(train_X, test_X, correlation):
    def Slect_Sperman(correlation):
        del_listi = []
        del_listj = []
        for i in correlation.columns:
            for j in correlation.index:
                if 1 > abs(correlation[i][j]) > 0.8:
                    del_listi.append(i)
                    if j not in del_listi:
                        del_listj.append(j)
        return del_listj

    # 提取spearman后的数据(即删除了TEMP和PRES列)
    drop_list1 = list()
    drop_list1 = Slect_Sperman(correlation)
    train_X = train_X.drop(drop_list1,axis=1) 
    test_X = test_X.drop(drop_list1,axis=1) 
    column = list(train_X)
    
    return train_X, test_X, column

@st.cache_data
def importance_eval(train_X, test_X, train_Y):
    # 对变量进行随机森林特征重要性评分
    model = RandomForestRegressor(random_state=0)
    model.fit(train_X, train_Y)
    # 重要性得分
    importances = pd.DataFrame({'feature':train_X.columns, 'importance':np.round(model.feature_importances_, 3)})
    importances = importances.sort_values('importance',ascending = False).set_index('feature')
    # 删除重要性得分小于0.01的特征变量(即ls和lr)
    names=list(train_X)
    value=model.feature_importances_
    jihe=[]
    for i in range(0,value.shape[0]):
        if value[i]<0.01:
            jihe.append(i)

    NAMES=pd.DataFrame(names)
    NAMES.drop(jihe,axis=0,inplace=True)
    jihe2=[]
    for i in NAMES.index:
        jihe2.append(NAMES.loc[i,0])
    train_X = train_X.loc[:, jihe2]
    test_X = test_X.loc[:, jihe2]
    return train_X, test_X, importances

@st.cache_data
def model_eval(train_X, test_X, train_Y, test_Y):
    # 建立Pm2.5浓度的梯度提升树模型
    Gbdt = GradientBoostingRegressor(n_estimators=500, max_depth=9, random_state=1)
    model = Gbdt
    model.fit(train_X, train_Y)
    y_predict = model.predict(test_X)
    MSE_value = mean_squared_error(test_Y, y_predict)
    # 平均绝对误差MAE
    MAE_value = mean_absolute_error(test_Y, y_predict)
    # 均方根误差RMSE
    RMSE_value = np.sqrt(metrics.mean_squared_error(test_Y, y_predict))

    R2_value = (sum((y_predict - np.mean(test_Y)) ** 2)) / (
                sum((test_Y - np.mean(test_Y)) ** 2))

    eval_dict = {'MSE': [MSE_value], 'MAE': [MAE_value], 'RMSE': [RMSE_value], 'R^2': [R2_value]}
    eval_df = pd.DataFrame(eval_dict)

    return y_predict, eval_df

@st.cache_data
def compare_chart(y_predict, test_Y):
    # 画出预测值和真实值的对比折线图(前200个样本)
    test_Y.reset_index(drop=True, inplace=True)
    test_huitu=test_Y.loc[0:99]
    y_predict_huitu=y_predict[0:100]
    pred_real = pd.DataFrame({
    '样本': np.arange(100),
    '预测值': y_predict_huitu,
    '真实值': test_huitu})
    return pred_real

@st.cache_data
def month_chart(data,feature):
    # 不同月份的露点均值
    dewp=[]
    x=[]
    for i in range(1,13):
        idd=list(data['month'][data['month']==i].index)
        DEWP_data=data.loc[idd,feature].mean()
        x.append(i)
        dewp.append(DEWP_data)

    chart_data = pd.DataFrame({
        '月份':x,
        '均值':dewp
    })
    return chart_data

# '数据集概览'
data = process_data(data_path)
data_enc = encode_categorical(data)
train_X, test_X, train_Y, test_Y = split_data(data_enc)

#'不同月份的露点分布'
chart_data_DEWP = month_chart(data,'DEWP')
#'不同月份的温度分布'
chart_data_TEMP = month_chart(data,'TEMP')
#'不同月份的大气压分布'
chart_data_PRES = month_chart(data,'PRES')
#'不同月份的PM2.5分布'
chart_data_pm25 = month_chart(data,'pm2.5')

# '相关性评分'
corr = correlation_heatmap(train_X)
train_X, test_X, column = process_data_with_spearman(train_X, test_X, corr)
# '重要性评分'
train_X, test_X, importances = importance_eval(train_X, test_X, train_Y)
# '评估结果'
y_predict, eval_df = model_eval(train_X, test_X, train_Y, test_Y)

# '预测值与真实值对比'
pred_real = compare_chart(y_predict, test_Y)

# 在侧边栏中创建目录
option = st.sidebar.radio(
    '实验步骤',
    ('数据集概览','描述性分析','参数优化', '模型评估')
)

# 获取本地化的月份名称
months_list = ['一月', '二月', '三月', '四月', '五月', '六月', '七月', '八月', '九月', '十月', '十一月', '十二月']

# 根据选择的目录项在主面板中显示相应的内容
if option == '数据集概览':
    st.title("数据集概览")
    st.dataframe(data)  # 使用streamlit展示数据框
elif option == '描述性分析':
    with st.container():
        c1 = (
            Line()
            .add_xaxis(xaxis_data=months_list)
            .add_yaxis("DEWP", chart_data_DEWP['均值'].values.tolist())
            .set_global_opts(
                title_opts=opts.TitleOpts(title="月均露点值"),
                xaxis_opts=opts.AxisOpts(type_="category"),
                yaxis_opts=opts.AxisOpts(type_="value",axislabel_opts=opts.LabelOpts(formatter="{value}")),
                tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross")
                )
            .set_series_opts(label_opts=opts.LabelOpts(formatter=JsCode("function(x){return x.data[1].toFixed(2);}")))
        )
        c2 = (
            Line()
            .add_xaxis(xaxis_data=months_list)
            .add_yaxis("PM2.5", chart_data_pm25['均值'].values.tolist())
            
            .set_global_opts(
                title_opts=opts.TitleOpts(title="月均PM2.5浓度"),
                xaxis_opts=opts.AxisOpts(type_="category"),
                yaxis_opts=opts.AxisOpts(type_="value",axislabel_opts=opts.LabelOpts(formatter="{value}")),
                tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross")
                )
            .set_series_opts(label_opts=opts.LabelOpts(formatter=JsCode("function(x){return x.data[1].toFixed(2);}")))
        )
        c3 = (
            Line()
            .add_xaxis(xaxis_data=months_list)
            .add_yaxis("PRES", chart_data_PRES['均值'].values.tolist())
            
            .set_global_opts(
                title_opts=opts.TitleOpts(title="月均大气压强"),
                xaxis_opts=opts.AxisOpts(type_="category"),
                yaxis_opts=opts.AxisOpts(type_="value",axislabel_opts=opts.LabelOpts(formatter="{value}")),
                tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross")
                )
            .set_series_opts(label_opts=opts.LabelOpts(formatter=JsCode("function(x){return Math.floor(x.data[1]);}")))
        )
        c4 = (
            Line()
            .add_xaxis(xaxis_data=months_list)
            .add_yaxis("TEMP", chart_data_TEMP['均值'].values.tolist())
            
            .set_global_opts(
                title_opts=opts.TitleOpts(title="月均温度"),
                xaxis_opts=opts.AxisOpts(type_="category"),
                yaxis_opts=opts.AxisOpts(type_="value",axislabel_opts=opts.LabelOpts(formatter="{value}")),
                tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross")
                )
            .set_series_opts(label_opts=opts.LabelOpts(formatter=JsCode("function(x){return x.data[1].toFixed(2);}")))
        )
        t = Timeline(init_opts=opts.InitOpts(theme=ThemeType.LIGHT,width='600px'))
        t.add_schema(play_interval=10000,is_auto_play=True)
        t.add(c1, "月均露点值")
        t.add(c2, "月均PM2.5浓度")
        t.add(c3, "月均大气压强")
        t.add(c4, "月均温度")
        st_pyecharts(t, width='600px', height='500px')

elif option == '参数优化':
    st.title('相关性评分')
    # 将corr DataFrame转换为适合热力图的格式
    value = [(i, j, corr.iloc[i, j]) for i in range(corr.shape[0]) for j in range(corr.shape[1])]

    c = (
        HeatMap()
        .add_xaxis(list(corr.columns))
        .add_yaxis(series_name = "相关系数", yaxis_data = list(corr.index), value = value, label_opts = opts.LabelOpts(position="middle")
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="各属性相关系数"),
            visualmap_opts=opts.VisualMapOpts(min_=-1, max_=1,pos_right="right", pos_bottom="center"),
        )
        .set_series_opts(label_opts=opts.LabelOpts(formatter=JsCode("function(params){return params.value[2].toFixed(2);}")))
    )
    st_pyecharts(c, width='600px', height='600px')
    
    st.title('重要性评分')
    st.bar_chart(importances)

elif option == '模型评估':
    st.title('评估结果')
    # 设置表格样式
    # 将DataFrame转换为HTML字符串，不添加索引，将第一行加粗
    eval_df_html = eval_df.to_html(index=False, header=True, bold_rows=True)
    # 使用st.markdown来显示表格
    st.markdown(eval_df_html, unsafe_allow_html=True)
    st.title('模型预测值与实际值对比')
    st.area_chart(pred_real.set_index('样本'), color=["#007aff", "#ff1111"])
