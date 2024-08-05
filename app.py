# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 08:48:01 2024

@author: annuk
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from pylab import rcParams
import statsmodels.api as sm

st.title("Stock Market Analysis and Prediction using Deep Learning")

company = st.selectbox("Choose a comapny: ",
                     ['Microsoft', 'Google', 'IBM','Amazon'])
 
st.header("Company name: "+ company)

if (company=='Microsoft'):
    dataset=pd.read_csv("C:/Users/annuk/Desktop/Stock-Market-Analysis-And-Forecasting-Using-Deep-Learning-master/Stock-Market-Analysis-And-Forecasting-Using-Deep-Learning-master/MSFT_2006-01-01_to_2018-01-01.csv", index_col='Date', parse_dates=['Date'])
elif (company=='Google'):
    dataset= pd.read_csv("C:/Users/annuk/Desktop/Stock-Market-Analysis-And-Forecasting-Using-Deep-Learning-master/Stock-Market-Analysis-And-Forecasting-Using-Deep-Learning-master/GOOGL_2006-01-01_to_2018-01-01.csv", index_col='Date', parse_dates=['Date'])
elif(company=='IBM'):
    dataset = pd.read_csv("C:/Users/annuk/Desktop/Stock-Market-Analysis-And-Forecasting-Using-Deep-Learning-master/Stock-Market-Analysis-And-Forecasting-Using-Deep-Learning-master/IBM_2006-01-01_to_2018-01-01.csv", index_col='Date', parse_dates=['Date'])
elif(company=='Amazon'):
    dataset = pd.read_csv("C:/Users/annuk/Desktop/Stock-Market-Analysis-And-Forecasting-Using-Deep-Learning-master/Stock-Market-Analysis-And-Forecasting-Using-Deep-Learning-master/AMZN_2006-01-01_to_2018-01-01.csv", index_col='Date', parse_dates=['Date'])

st.write("First few rows of the stock data:")
st.dataframe(dataset.head())

####################################################################################################

st.write("Descriptive statistics of the stock data:")
st.dataframe(dataset.describe())

####################################################################################################

fig = px.histogram(dataset, 
                   x='Close', 
                   marginal='box',
                   color_discrete_sequence=['purple'],
                   title='Distribution of Close')
fig.update_layout(bargap=0.1)

st.subheader("Distribution of the 'Close' prices:")
st.plotly_chart(fig)

fig = px.histogram(dataset, 
                   x='Open', 
                   marginal='box',
                   color_discrete_sequence=['green'],
                   title='Distribution of Open')
fig.update_layout(bargap=0.1)

st.subheader("Distribution of the 'Open' prices:")
st.plotly_chart(fig)

####################################################################################################

numeric_columns = dataset.select_dtypes(include=['float64', 'int64'])

correlation_matrix = numeric_columns.corr()

st.subheader("Correlation matrix of numeric columns:")
st.dataframe(correlation_matrix)

####################################################################################################

fig_scatter = px.scatter(dataset, 
                         x='Open', 
                         y='Close', 
                         color='Volume', 
                         opacity=0.8, 
                         title='Open vs. Close',
                         color_continuous_scale=px.colors.sequential.Magma)
fig_scatter.update_traces(marker_size=5)

st.subheader("Open vs. Close scatter plot:")
st.plotly_chart(fig_scatter)

####################################################################################################

st.subheader('Expanding window functions:')
dataset_mean = dataset['High'].expanding().mean()
dataset_std = dataset['High'].expanding().std()

plt.figure(figsize=(10, 6))
dataset['High'].plot(label='High')
dataset_mean.plot(label='Expanding Mean')
dataset_std.plot(label='Expanding Standard Deviation')
plt.legend()
plt.title(company)
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)

st.pyplot(plt)

####################################################################################################

# Time series decomposition for Google High values
rcParams['figure.figsize'] = 11, 9
decomposed_dataset_high = sm.tsa.seasonal_decompose(dataset["High"], period=360)  # The frequency is annual
figure = decomposed_dataset_high.plot()

st.subheader("Time series decomposition of High values:")
st.pyplot(figure)

####################################################################################################

if (company=='Microsoft'):
    msft_result=pd.read_excel("C:/Users/annuk/Desktop/Stock-Market-Analysis-And-Forecasting-Using-Deep-Learning-master/Stock-Market-Analysis-And-Forecasting-Using-Deep-Learning-master/msft_result.xlsx")
    
    import plotly.express as px
    import plotly.graph_objects as go
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=msft_result.index, y=msft_result[0],
                             mode='lines',
                             name='Train prediction'))
    fig.add_trace(go.Scatter(x=msft_result.index, y=msft_result[1],
                             mode='lines',
                             name='Test prediction'))
    fig.add_trace(go.Scatter(x=msft_result.index, y=msft_result[2],
                             mode='lines',
                             name='Actual Value'))
    
    fig.update_layout(
        xaxis=dict(
            showline=True,
            showgrid=True,
            showticklabels=False,
            linecolor='white',
            linewidth=2
        ),
        yaxis=dict(
            title_text='Close (USD)',
            titlefont=dict(
                family='Rockwell',
                size=12,
                color='white',
            ),
            showline=True,
            showgrid=True,
            showticklabels=True,
            linecolor='white',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Rockwell',
                size=12,
                color='white',
            ),
        ),
        showlegend=True,
        template='plotly_dark'
    )
    
    annotations = []
    annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                            xanchor='left', yanchor='bottom',
                            text='Microsoft Stock Prediction',
                            font=dict(family='Rockwell',
                                      size=26,
                                      color='white'),
                            showarrow=False))
    fig.update_layout(annotations=annotations)
    st.plotly_chart(fig)
    st.subheader("Evaluation metrics:")
    st.write('Train prediction score is: 0.58 RMSE')
    st.write('Test prediction score is:  3.52 RMSE')
#############################################################################################################

elif(company=='Google'):
    googl_result=pd.read_excel("C:/Users/annuk/Desktop/Stock-Market-Analysis-And-Forecasting-Using-Deep-Learning-master/Stock-Market-Analysis-And-Forecasting-Using-Deep-Learning-master/googl_result.xlsx")
    import plotly.express as px
    import plotly.graph_objects as go
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=googl_result.index, y=googl_result[0],
                             mode='lines',
                             name='Train prediction'))
    fig.add_trace(go.Scatter(x=googl_result.index, y=googl_result[1],
                             mode='lines',
                             name='Test prediction'))
    fig.add_trace(go.Scatter(x=googl_result.index, y=googl_result[2],
                             mode='lines',
                             name='Actual Value'))
    
    fig.update_layout(
        xaxis=dict(
            showline=True,
            showgrid=True,
            showticklabels=False,
            linecolor='white',
            linewidth=2
        ),
        yaxis=dict(
            title_text='Close (USD)',
            titlefont=dict(
                family='Rockwell',
                size=12,
                color='white',
            ),
            showline=True,
            showgrid=True,
            showticklabels=True,
            linecolor='white',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Rockwell',
                size=12,
                color='white',
            ),
        ),
        showlegend=True,
        template='plotly_dark'
    )
    
    annotations = []
    annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                            xanchor='left', yanchor='bottom',
                            text='Google Stock Prediction',
                            font=dict(family='Rockwell',
                                      size=26,
                                      color='white'),
                            showarrow=False))
    fig.update_layout(annotations=annotations)
    
    st.plotly_chart(fig)
    st.subheader("Evaluation metrics:")
    st.write('Train prediction score is: 6.38 RMSE')
    st.write('Test prediction score is:  66.85 RMSE')
    
#############################################################################################################

elif(company=='IBM'):
    ibm_result=pd.read_excel("C:/Users/annuk/Desktop/Stock-Market-Analysis-And-Forecasting-Using-Deep-Learning-master/Stock-Market-Analysis-And-Forecasting-Using-Deep-Learning-master/ibm_result.xlsx")
    
    import plotly.express as px
    import plotly.graph_objects as go
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ibm_result.index, y=ibm_result[0],
                             mode='lines',
                             name='Train prediction'))
    fig.add_trace(go.Scatter(x=ibm_result.index, y=ibm_result[1],
                             mode='lines',
                             name='Test prediction'))
    fig.add_trace(go.Scatter(x=ibm_result.index, y=ibm_result[2],
                             mode='lines',
                             name='Actual Value'))
    
    fig.update_layout(
        xaxis=dict(
            showline=True,
            showgrid=True,
            showticklabels=False,
            linecolor='red',
            linewidth=2
        ),
        yaxis=dict(
            title_text='Close (USD)',
            titlefont=dict(
                family='Rockwell',
                size=12,
                color='black',
            ),
            showline=True,
            showgrid=True,
            showticklabels=True,
            linecolor='blue',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Rockwell',
                size=12,
                color='white',
            ),
        ),
        showlegend=True,
        template='plotly_dark'
    )
    
    annotations = []
    annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                            xanchor='left', yanchor='bottom',
                            text='IBM Stock Prediction',
                            font=dict(family='Rockwell',
                                      size=26,
                                      color='white'),
                            showarrow=False))
    fig.update_layout(annotations=annotations)
    
    st.plotly_chart(fig)
    st.subheader("Evaluation metrics:")
    st.write('Train prediction score is: 2.03 RMSE')
    st.write('Test prediction score is:  1.95 RMSE')

#############################################################################################################

elif(company=='Amazon'):
    amzn_result=pd.read_excel("C:/Users/annuk/Desktop/Stock-Market-Analysis-And-Forecasting-Using-Deep-Learning-master/Stock-Market-Analysis-And-Forecasting-Using-Deep-Learning-master/amzn_result.xlsx")
    import plotly.express as px
    import plotly.graph_objects as go
    
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=amzn_result.index, y=amzn_result[0],
                             mode='lines',
                             name='Train prediction'))
    fig.add_trace(go.Scatter(x=amzn_result.index, y=amzn_result[1],
                             mode='lines',
                             name='Test prediction'))
    fig.add_trace(go.Scatter(x=amzn_result.index, y=amzn_result[2],
                             mode='lines',
                             name='Actual Value'))
    
    fig.update_layout(
        xaxis=dict(
            title_text='Prediction',
            showline=True,
            showgrid=True,
            showticklabels=False,
            linecolor='white',
            linewidth=2
        ),
        yaxis=dict(
            title_text='Close (USD)',
            titlefont=dict(
                family='Rockwell',
                size=12,
                color='white',
            ),
            showline=True,
            showgrid=True,
            showticklabels=True,
            linecolor='white',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Rockwell',
                size=12,
                color='white',
            ),
        ),
        showlegend=True,
        template='plotly_dark'
    )
    
    annotations = []
    annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                            xanchor='left', yanchor='bottom',
                            text='Amazon Stock Prediction',
                            font=dict(family='Rockwell',
                                      size=26,
                                      color='white'),
                            showarrow=False))
    fig.update_layout(annotations=annotations)
    
    st.plotly_chart(fig)
    st.subheader("Evaluation metrics:")
    st.write('Train prediction score is: 0.58 RMSE')
    st.write('Test prediction score is:  3.52 RMSE')
