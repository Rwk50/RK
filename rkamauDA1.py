# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 00:04:09 2019

@author: rkamau
"""
#Question 1
import pandas as pd
import numpy as np
WeatherData = pd.read_csv("Paris_weather_data_2017.csv",encoding='windows-1252',na_values='-')
print(WeatherData.isna().sum())

#Question 2
#removing the two columns with many nans
WeatherData = WeatherData.drop(WeatherData.columns[[18, 20]], axis = 1)
# interpolating the WeatherData
WeatherData=WeatherData.interpolate()
print(WeatherData.isna().sum())
import matplotlib.pyplot as plt
import seaborn as sns
#Correlation coeffients matrix
corr = WeatherData.corr()
print(corr)

#Heatmap
f, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 600, as_cmap=True), square=True, ax=ax,annot=True)

#Question 3
Energy=pd.read_excel("Electricity consumption for France.xls")
#Removing empty rows
Energy.dropna()

#Question 4
#Setting data type to datetime
WeatherData['Date'] = pd.to_datetime(WeatherData['Date'])
Energy['Date'] = pd.to_datetime(Energy['Date'],dayfirst=True)
#Synchronising data 
WeatherEnergyData = pd.merge(WeatherData,Energy,on='Date')
#Making a scatter plot 
plt.figure(figsize=(10,10))
plt.scatter(WeatherEnergyData['avg Temp.\xa0(°C)'],WeatherEnergyData['Energie journalière (MWh)'])
plt.xlabel('Mean temperature')
plt.ylabel('Energy consumption')
plt.title('Energy consumption versus mean temperature')

#Question 5
import numpy.polynomial.polynomial as poly
#Setting data type to datetime
WeatherData['Date']=pd.to_datetime(WeatherData['Date'])
Energy['Date']=pd.to_datetime(Energy['Date'],dayfirst=True)
#Synchronising the two data sets based on dates 
WeatherEnergyData=pd.merge(WeatherData,Energy,on='Date')
#Plotting 
plt.figure(figsize=(20,20))
plt.scatter(WeatherEnergyData['avg Temp.\xa0(°C)'],WeatherEnergyData['Energie journalière (MWh)'])
plt.xlabel('Mean temperature')
plt.ylabel('Energy consumption')
plt.title('Energy consumption versus mean temperature')
#Getting the corresponding coeffiecients of the polynomial
Polycoefficients = poly.polyfit(WeatherEnergyData['avg Temp.\xa0(°C)'],WeatherEnergyData['Energie journalière (MWh)'],2)
x=np.linspace(-3,35)
#Evaluating the polynomial at x values
y=poly.polyval(x,Polycoefficients)
#plotting the  quadratic fit
plt.plot(x,y,color='g')


# minimum energy consumption
min_energy_consump = min(y)
#avg temperature value for minimum energy consumption
min_avg_temperature = x[y.argmin()]
min_avg_temperature

plt.plot(min_avg_temperature, min_energy_consump,'ro',ms=15)
plt.show()