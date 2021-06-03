# coding: utf-8
import pandas as pd

hdt = pd.read_csv('DeloitteWeekly_v3_LJS_04182021.csv')
yolo_hdt = hdt[hdt['County']=='Yolo']
yolo_hdt
pd.PeriodIndex(yolo_hdt['ResultDate'], freq='W')
yolo_hdt['week'] = pd.PeriodIndex(yolo_hdt['ResultDate'], freq='W')
yolo_hdt.query('Result=="Detected"').groupby('week').sum()
yolo_hdt.query('Result=="Detected"').groupby('week').count()
yolo_hdt.query('Result=="Detected"').groupby('week').count()['Row']
pos = yolo_hdt.query('Result=="Detected"').groupby('week').count()['Row']
neg = yolo_hdt.query('Result=="Not Detected"').groupby('week').count()['Row']
pos
neg
pos.join(neg)
pd.DateFrame({'pos result':pos, 'neg result':neg})
pd.DataFrame({'pos result':pos, 'neg result':neg})
pd.DataFrame({'pos result':pos, 'neg result':neg})['2020-12-1':'2021-2-1']
pd.DataFrame({'pos result':pos, 'neg result':neg})['2020-12-1':'2021-2-1'].to_csv('dec_jan_hdt.csv')
cdph = pd.read_csv('CDPH_testing_data_4_24.xlsx')
import pandas as pd
cdph = pd.read_csv('CDPH_testing_data_4_24.xlsx')
cdph = pd.read_excel('CDPH_testing_data_4_24.xlsx')
cdph
cdph['lab_result_date]
yolo = cdph['county'].str.lower() == 'yolo'
yolo = cdph[cdph['county'].str.lower() == 'yolo']
yolo
yolo.query('lab_result_date == "2020-12-15"')
yolo['week'] = pd.PeriodIndex(yolo['lab_result_date'], freq='W')
yolo.groupby('week').sum()
yolo.groupby('week').sum()['2020-12-1':'2021-1-1']
yolo.groupby('week').sum()['2020-12-1':'2021-1-1'].to_csv('dec_yolo_cdph.csv')
yolo.groupby('week').sum()['2020-12-1':'2021-2-1'].to_csv('dec_yolo_cdph.csv')
