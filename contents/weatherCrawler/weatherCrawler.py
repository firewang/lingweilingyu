# -*- encoding: utf-8 -*-
# @Version : 1.0  
# @Time    : 2019/5/5 9:37
# @Author  :  fireWang
# @note    :  pandas.read_html抓取网页表格数据，re正则表达式清洗,pandas.to_excel保存数据

import calendar
import datetime
import os
import random
import re
import time
from pprint import pprint

import numpy as np
import pandas as pd
from dateutil.parser import parse


def get_month_period(month_begin=1, month_end=0):
    '''
    获得自然月份间隔时间段, 默认取前一个自然月
    :param month_begin: 几个月前的第一天
    :param month_end: 几个月前的结束第一天
    :return: e.g（2018,4,1 ，2018，5,1）
    '''
    now = datetime.datetime.now()
    day = datetime.datetime.strptime(datetime.datetime.strftime(now.replace(day=1), "%Y-%m-%d"), "%Y-%m-%d")

    def get_day(shijian, zhouqi):
        for i in range(zhouqi):
            last_month_last_day = shijian - datetime.timedelta(days=1)
            cc = calendar.monthrange(last_month_last_day.year, last_month_last_day.month)
            last_month_first_day = shijian - datetime.timedelta(days=cc[1])
            shijian = last_month_first_day
            i += 1
        return (last_month_first_day, last_month_last_day)

    begin = get_day(day, month_begin)[0]
    end = get_day(day, month_end + 1)[1] + datetime.timedelta(days=1)
    return begin, end


def get_weather_data(city='hangzhou', time_func_name=get_month_period, *args):
    begin, end = time_func_name(*args)
    print(begin, end)
    # 获得需要爬取的日期区间
    date_list = [date.strftime("%Y%m") for date in pd.date_range(begin, end, freq='M')]
    # 构建url
    url_list = ["http://www.tianqihoubao.com/lishi/{}/month/{}.html".format(city, date) for date in date_list]
    pprint(url_list)
    # 合并后的天气信息文件
    filepath = os.path.join(os.path.abspath(os.getcwd()), 'data',
                            "weather-{}-{}-{}.xlsx".format(city, date_list[0], date_list[-1]))
    if os.path.exists(filepath):
        weather_data = pd.read_excel(filepath)
    else:
        # 抓取天气信息
        weather_data = pd.DataFrame(columns=["日期", "天气状况", "气温", "风力风向"])
        for index, url in enumerate(url_list):
            weatherDataFilePath = os.path.join(os.path.abspath(os.getcwd()), 'data',
                                               "weather-{}-{}.xlsx".format(city, date_list[index]))
            # print(weatherDataFilePath)
            try:
                weather_df = pd.read_excel(weatherDataFilePath, header=0)
                # 不完整月份的天气数据补充
                current_date = datetime.datetime.strptime(date_list[index], '%Y%m')
                if weather_df.shape[0] < calendar.monthrange(current_date.year, current_date.month)[1]:
                    weather_df = pd.DataFrame(pd.read_html(url, encoding='GBK', header=0)[0])
                    weather_df.to_excel(weatherDataFilePath, index=None)
            except Exception:
                weather_df = pd.DataFrame(pd.read_html(url, encoding='GBK', header=0)[0])
                weather_df.to_excel(weatherDataFilePath, index=None)
                # 随机等待 [1-10]秒 发送请求
                time.sleep(random.randint(1, 10))

            weather_data = pd.concat([weather_data, weather_df], ignore_index=True)

        weather_data.to_excel(filepath, index=None)
    return weather_data, filepath


def clean_fengli(x):
    '''正则表达式清洗风力数据的格式'''
    pattern1 = re.compile('(\d+)(\W+)(\d+)')  # 1-2, 1~2
    pattern2 = re.compile('(\d*)(\W+)(\d+)')  # <2  <=2
    pattern3 = re.compile('(\d+)')  # 2
    if re.match(pattern1, x):
        return np.mean((int(re.match(pattern1, x).groups()[0]), int(re.match(pattern1, x).groups()[2])))
    elif re.match(pattern2, x):
        return int(re.match(pattern2, x).group()[1]) - 0.5
    else:
        return int(re.match(pattern3, x).group(0))


def clean_weather_data(df, filepath, remove=True):
    '''使用正则表达式清洗天气数据'''
    ptianqi = re.compile('\w+')
    pwendu = re.compile('\d+')
    pfengli = re.compile('(\w+)\s+(\d*\W+\d+)')
    df['主天气状况'] = df.loc[:, '天气状况'].apply(lambda x: ptianqi.findall(x)[0])
    df['次天气状况'] = df.loc[:, '天气状况'].apply(lambda x: ptianqi.findall(x)[1])
    df['主风向'] = df.loc[:, '风力风向'].apply(lambda x: pfengli.findall(x)[0][0])
    df['主风力'] = df.loc[:, '风力风向'].apply(lambda x: pfengli.findall(x)[0][1])
    df['主风力'] = df.loc[:, '主风力'].apply(lambda x: clean_fengli(x))
    df['次风向'] = df.loc[:, '风力风向'].apply(lambda x: pfengli.findall(x)[1][0])
    df['次风力'] = df.loc[:, '风力风向'].apply(lambda x: pfengli.findall(x)[1][1])
    df['次风力'] = df.loc[:, '次风力'].apply(lambda x: clean_fengli(x))
    df['最高温度'] = df.loc[:, '气温'].apply(lambda x: pwendu.findall(x)[0])
    df['最低温度'] = df.loc[:, '气温'].apply(lambda x: pwendu.findall(x)[1])
    df["日期"] = df["日期"].apply(lambda x: parse("-".join(re.match('(\d+)\w*(\d{2})\w*(\d{2,})', x).groups())))
    if remove:
        os.remove(filepath)
    df.drop(columns=["天气状况", "气温", "风力风向"], inplace=True)
    # 存储所有清洗好的天气数据
    df.to_excel(filepath.replace('weather-', 'weatherCleaned-'), index=False)
    return df  # [日期 主天气状况 次天气状况 主风向 主风力 次风向 次风力 最高温度 最低温度]


if __name__ == '__main__':
    weather_data, filepath = get_weather_data('hangzhou', get_month_period, 3)
    clean_weather_data(weather_data, filepath, remove=True)
