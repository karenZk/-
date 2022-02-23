# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 01:20:56 2022

@author: Lenovo
"""
# In[ 1]:
import os
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

from factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity
import numpy as np
import numpy.linalg as nlg
import sys
import subprocess

# In[ 1]:

'''
下载因子分析包
'''
subprocess.check_call([sys.executable, '-m', 'pip', 'install',

                       'factor_analyzer'])

# In[ 2]:

'''
修改指数ROE, 总营收, 净利润三页的index，使其为datastamp
'''


def y_date(row): return row[17:21]+'0331' if (
    row[21] == '一') else(row[17:21] + '0630' if (row[21] == '中') else (
        row[17:21]+'0930' if (row[21] == '三') else (row[17:21]+'1231')))


# In[ 3]:
def load_data(file_path):
    '''
    读取数据
    '''

    with open(file_path, "rb") as f:
        y = pd.read_excel(f, '汇总', skiprows=2, names=[
                          "date", "ROE", "总营收", "净利润"])
        x = pd.read_excel(f, '指标', skiprows=4, header=None, index_col=0)
        cols = pd.read_excel(f, '指标', nrows=2, header=1, index_col=0)
        x.columns = cols.columns
        y = y[y['净利润'] != 0]
        y = y[y['净利润'] != '--']
        y = y.replace('--', method='bfill')
        y = y.dropna()
        date = y['date']
        y.index = pd.to_datetime(date.apply(y_date), format="%Y%m%d")
        y = y.drop(columns=["date"])
        y = y.sort_index(ascending=False)

        y = y[y.index.isin(x.index)]
        x = x[x.index >= (y.index[-1])]

    return y, x


# In[ 4]


def cal_cumsum_return(df):
    '''
    修改当月值至累计同比，适用于产量、销量等非价格数据
    '''

    df = df.replace(np.nan, 0)
    array = df.to_numpy()
    array = np.flip(array)  # to flip the order
    array = np.cumsum(array)
    array = np.flip(array)
    array = np.diff(array, n=12)/array[12:]
    array = np.insert(array, -1, np.zeros(12))
    array = np.nan_to_num(array, neginf=0, posinf=0)
    return array[0:]

# In[ 5]


def price_change(df):
    '''
    修改当月值至累计平均值，适用于价格、指数等价格类数据
    '''

    df = df.replace(np.nan, 0)
    ori_a = df.to_numpy()
    array = np.zeros(len(df))
    for i in range(len(array)-12):
        mon = int(df.index[i].month)
        array[i] = np.sum(ori_a[i:i+mon])/mon
    '''
    array = np.diff(array, n=12)/array[12:]
    array = np.insert(array, -1, np.zeros(12))
    array = np.nan_to_num(array, neginf=0, posinf=0)
    '''
    return array[0:]


# In[ 6]

def y_corr_check(df_y, df_x, corr_check):
    df = df_x[df_x.index.isin(df_y.index)]
    corr_df = df.corrwith(df_y['净利润'].astype(float))
    to_drop = corr_df.index[abs(corr_df) < corr_check]
    x = df_x.drop(columns=to_drop)
    # print(drop)
    return x

# In[ 7]


def del_y_outlier(y, n):
    '''
    Parameters
    ----------
    y : Dataframe Series
        原始值
    n : 替换几个逸出值.
    '''
    np_y = y.to_numpy()
    ind = np.argpartition(abs(np_y), -n)[-2*n:]
    np_y[ind[-n:]] = np_y[ind[:n]]
    return pd.DataFrame(np_y, index=y.index, columns=[y.name])


# In[ 8]
# 皮尔森相关系数


def Pearson_check(df_x):
    '''
    皮尔森相关系数 检查
    '''
    df = df_x
    kmo = calculate_kmo(df)  # kmo值要大于0.7
    bartlett = calculate_bartlett_sphericity(df)  # bartlett球形度检验p值要小于0.05
    print("\n因子分析适用性检验:")
    print('kmo:{},bartlett:{}'.format(kmo[1], bartlett[1]))
    return kmo[1], bartlett[1]

# In[9]


class fac_drop():
    '''
    挑选因子候选者
    '''

    def __init__(self, df=None):
        self.df = df
        self.n = len(self.df.columns)
        self.fa = FactorAnalyzer(
            rotation=None, n_factors=self.n, method='principal')
        self.fa.fit(self.df)
        self.com = self.fa.get_communalities()
        self.std = self.fa.get_factor_variance()
        self.eig = pd.DataFrame(
            {'特征值': self.std[0], '方差贡献率': self.std[1],
             '方差累计贡献率': self.std[2]})

    def flex_n(self):
        '''
        用于fix_n = False.
        优化方式：
            特征值小于1
        返回值：
            公因子个数
            需要删除的因子候选者名称
            方差贡献率

        '''
        keep = self.eig[self.eig['方差累计贡献率'].le(0.85)]
        drop = self.eig[self.eig['特征值'].le(1)].index
        self.weight = keep['方差贡献率']
        n_fac = len(self.weight)
        return n_fac, drop, self.weight

    def fix_n(self):
        '''
        用于fix_n = True.
        优化方式：
            删除因子特征值小于1
        返回值：
            需要删除的因子候选者名称
        '''
        drop = self.eig[self.eig['特征值'].le(1)].index
        return drop

# In[10]


def fac_renew(df, n_fix):
    '''
    输入： 因子候选
    输出： 使用的因子候选，因子weight，因子个数 
    常数： 
        n_fix 因子个数

    设定输出因子为n_fix个，提取值小于1的候选者，直至kimo值>0.8 或无法进行优化删除。
    '''
    fa = FactorAnalyzer(rotation=None, n_factors=n_fix, method='principal')
    fa.fit(df)
    col_original = df.columns
    kmo, bartlett, = Pearson_check(df)
    to_drop = fac_drop(df).fix_n()
    col_drop = col_original[to_drop]
    kmo_old = 0
    while kmo < 0.8 or bartlett > 0.05:
        # print(col_drop)
        if kmo == kmo_old:
            print('无法优化')
            break
        df = df.drop(columns=col_drop)
        col_original = df.columns
        to_drop = fac_drop(df).fix_n()
        fa = FactorAnalyzer(rotation=None, method='principal')
        fa.fit(df)
        kmo, bartlett = Pearson_check(df)
        col_drop = col_original[to_drop]
        kmo_old = kmo

    weight = fa.fit(df).get_factor_variance()[2]
    return n_fix, df, weight


# In[11]
def define_factor(df, n_def):
    '''
    输入： 因子候选
    输出： 使用的因子候选，因子weight，因子个数 

    不断删除特征值小于1的候选者，直至kimo值>0.8 或无法进行优化删除。
    如果第一次输出的因子个数为<2,则修改为fac_renew方式

    '''
    col_original = df.columns
    kmo, bartlett, = Pearson_check(df)
    n, to_drop, weight = fac_drop(df).flex_n()
    if n < 2:
        return fac_renew(df, n_def)

    col_drop = col_original[to_drop]
    n_old = n-1
    while kmo < 0.8 or bartlett > 0.05:
        # print(col_drop)
        if n_old == n-1:
            print('无法优化')
            break
        df = df.drop(columns=col_drop)
        col_original = df.columns
        n_old = n
        kmo, bartlett = Pearson_check(df)
        n, to_drop, weight = fac_drop(df).flex_n()
        col_drop = col_original[to_drop]
    return n, df, weight

# In[12]


def factor_analysis(df, n_fac, df_index, w):
    '''

    Parameters
    ----------
    df : Dataframe
        挑选出的因子候选者
    n_fac : int
        公因子个数
    df_index : index
        季度值分析的index

    Returns
    -------
    fa_t_score : 公因子值
    factor_score :  因子得分用于计算公因子

    '''
    df = df[df.index.isin(df_index)]
    fa = FactorAnalyzer(rotation=None, n_factors=n_fac, method='principal')
    fa.fit(df)

    # 查看公因子提取度
    #print("\n公因子提取度:\n", fa.get_communalities())

    # 查看因子载荷
    # print("\n因子载荷矩阵:\n",fa.loadings_)
    # 使用最大方差法旋转因子载荷矩阵
    fa_rotate = FactorAnalyzer(
        rotation='varimax', n_factors=n_fac, method='principal')
    fa_rotate.fit(df)

    # 查看旋转后的因子载荷
    # print("\n旋转后的因子载荷矩阵:\n", fa_5_rotate.loadings_)

    # 因子得分（回归方法）（系数矩阵的逆乘以因子载荷矩阵）
    X1 = np.mat(df.corr())
    X1 = nlg.inv(X1)

    # B=(R-1)*A  15*5
    factor_score = np.dot(X1, fa_rotate.loadings_)
    factor_score = pd.DataFrame(factor_score)
    factor_score.index = (df.corr()).columns
    #print("\n因子得分：\n", factor_score)

    fa_t_score = np.dot(np.mat(df), np.mat(factor_score))
    #print("\n应试者的四因子得分：\n", pd.DataFrame(fa_t_score))

    fa_t_score_sum = np.dot(fa_t_score, w) / sum(w)
    # fa_t_score_sum2 = np.mat(np.ones((35,2)))
    # fa_t_score_sum2[:,1] = fa_t_score_sum[:,0]
    return fa_t_score, factor_score

# In[13]


def lm_fit(df_y, df_F, factor_score, x_predict,  path, model_new_index=[]):
    '''
    Parameters
    ----------
    df_y : Dataframe
        y值
    df_x : Dataframe
        公因子值
    factor_score : Dataframe
        公因子分数
    x_predict : Dataframe
        季度预测的原始x值
    model_new_index： List
        使用的F和y 滞后项

    Returns
    -------
    result : Dataframe
        拟合y结果
    y_pred： float
        y预测实际值

    y_score ：float
        y景气度打分，(y_预测-y_min)/(y_max-y_min)
    coef : np array
        拟合系数
    x_ranked：data_faram 
        x得到第二次拟合使用的因子吸收，用于挑选重要指标
    x_F_chosen： data_faram
        重要的指标
    model_new_index: []
        ‘净利润’第一次回归显著项的位置，用于之后的y拟合和预测

    '''
    y = np.array(df_y.to_numpy(), dtype=float)
    fa_t_score = np.dot(x_predict.to_numpy(), np.mat(factor_score))  # 计算季度预测值

    F_test = np.c_[np.ones((len(df_F), 1)), df_F]  # 加入常熟项
    # 由于y值滞后四次，所以删除x_test的最后四行
    F_test = np.delete(F_test, list(range(-4, 0)), 0)
    F_test = np.c_[F_test, y[1:-3], y[2:-2], y[3:-1], y[4:]]  # 加入y值滞后值
    x_ranked = []
    x_F_chosen = []

    if df_y.name == '净利润':
        model1 = sm.OLS(y[:-4], F_test).fit()  # 线性回归
        coef = model1.params  # 获得系数
        pvalues = model1.pvalues  # 获得p-values
        # print(model1.summary())

        n = df_F.shape[1]
        model_new_index = np.where(pvalues <= .1)[0]
        if df_y.name == '净利润' and np.any(pvalues[1:n+1] <= .1):
            '''
            寻找对净利润影响最大的指标
            '''
            F_index = np.where(pvalues[1:n+1] <= .1)[0]  # pvalue < 0.1 的F
            x_ranked = factor_score.apply(
                lambda s: pd.Series(s.abs().nlargest(3).index))[F_index]
            x_F_chosen = pd.DataFrame(
                factor_score[F_index])
            x_F_chosen.columns = ['F' + str(e) for e in F_index]
            x_ranked.columns = x_F_chosen.columns
        elif df_y.name == '净利润':
            print('无显著')
            return

    model2 = sm.OLS(y[:-4], F_test[:, model_new_index]).fit()
    coef = model2.params  # 获得系数
    
    # print(model2.summary())
    y_fitted = model2.predict()  # 预测值

    result = pd.DataFrame(F_test, index=df_y.index[:-4])
    result['预测值'] = y_fitted
    result['实际值'] = y[:-4]
    result['resid'] = model2.resid
    print('std:', np.std(model2.resid))  # 计算预测值和实际值的方差

    fac_pred = np.c_[1, fa_t_score, y[0], y[1], y[2], y[3]]
    y_pred = np.dot(np.asarray(fac_pred).flatten()[model_new_index], coef)

    fontP = font_manager.FontProperties()
    fontP.set_family('SimHei')
    fontP.set_size(14)
    fig = plt.figure()

    result.plot(y=['resid', '实际值', '预测值'], color=['b', 'r', 'g'],
                label=['Residual', 'Actual', 'Fitted'], linewidth=0.9, xlabel='')

    plt.axhline(y=0.0, color='0', linestyle='-', linewidth=0.5)
    plt.axhline(y=0.0 + np.std(model2.resid), color='0',
                linestyle='--', linewidth=0.5)
    plt.axhline(y=0.0 - np.std(model2.resid), color='0',
                linestyle='--', linewidth=0.5)

    plt.legend(bbox_to_anchor=(0.25, -0.005), loc="lower center",
               bbox_transform=fig.transFigure, ncol=3)

    plt.title(df_y.name, fontproperties=fontP)
    file_path = os.path.join(path, df_y.name)
    plt.savefig(file_path)
    y_score = (y_pred-y.min())/(y.max()-y.min())  # 计算景气度指标
    return result, y_pred, y_score, coef, x_ranked, x_F_chosen, model_new_index

# In[14]


def pre_mon(df_y, df_x, coef, model_index):
    '''
    Parameters
    ----------
    df_y : 原始季度数据
    df_x : 原始月度数据
    coef : 拟合系数

    Returns
    -------
    y_result : 景气度
    '''
    y_month = df_y.reindex(df_x.index, method='bfill')
    y = np.array(y_month.to_numpy(), dtype=float)

    x_pred = np.c_[np.ones((len(df_x), 1)), np.mat(df_x)]
    x_pred = np.delete(x_pred, [-4, -3, -2, -1], 0)
    x_pred = np.c_[x_pred, y[1:-3], y[2:-2], y[3:-1], y[4:]]
    y_pred = np.dot(x_pred[:, model_index], coef)
    y_score = [(y_pred[0, i]-y.min())/(y.max()-y.min())
               for i in range(y_pred.shape[1])]
    return np.asarray(y_pred).flatten(), y_score

# In[15]


class return_result():
    def __init__(self, **kwargs):
        prop_defaults = {
            'script_dir': '',
            'sector_path': '',
            'pred_path': '',
            'results': [],
            'y_vol_pred': 0.0000,
            'ori_x_full': pd.DataFrame(),
            'coefs': pd.DataFrame(),
            'y_mon_pred': pd.DataFrame(),
            'fac_score': pd.DataFrame(),
            'x_ranked_top': pd.DataFrame(),
            'results_list': ['净利润', '总营收', 'ROE'],
        }
        self.__dict__.update(prop_defaults)
        self.__dict__.update(kwargs)
        self.pred_path = self.rel_path[:-5] + '_预测.xlsx'

        self.path = os.path.join(
            self.script_dir, self.sector_path, self.pred_path)

    def output_xlsx(self):
        with pd.ExcelWriter(self.path) as writer:
            for i in range(3):
                self.results[i].to_excel(
                    writer, sheet_name=self.results_list[i])
            self.y_vol_pred.to_excel(writer, sheet_name='最新一季度预测')

            self.ori_x_full.to_excel(writer, sheet_name='模型', startrow=1)

            self.coefs.to_excel(writer, sheet_name='模型',
                                startrow=4 + len(self.ori_x_full.index))

            self.y_mon_pred.to_excel(writer, sheet_name='月度预测')

            self.fac_score.to_excel(writer, sheet_name='指标')

            self.x_ranked_top.to_excel(writer, sheet_name='模型',
                                       startrow=10 + 1 + len(self.fac_score.index))

            writer.sheets['模型'].write_row(0, 0,  'F~X')
            writer.sheets['模型'].write_row(
                3 + len(self.fac_score.index), 0,  ['y', '~', '1+', 'F+', '滞后y'])
            writer.sheets['模型'].write_row(
                10 + len(self.fac_score.index), 0,  ['重要指标'])

        return
