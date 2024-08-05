# Stock-market-prediction
<b>Stock Market Analysis and Prediction</b>

**Introduction:**

A stock market or equity market is the aggregation of buyers and sellers of stocks or shares, which represent ownership claims on businesses. Generally, there are two ways for stock market prediction. Fundamental analysis relies on a company’s technique and fundamental information like market position, expenses, and annual growth rates. The second one is the technical analysis method, which concentrates on previous stock prices and values.
The task of stock prediction has always been a challenging problem for statistics experts. The main reason behind this prediction is buying stocks that are likely to increase in price and then selling stocks that are probably to fall. The evolution of data science, deep learning, and time series analysis the task of a stock buyer has become comprehensively easy. In the first part of our project, analysis of the data is done and in the second part, we will forecast the stock market.

**Datasets:**

Here we will use multiple stock market datasets such as the stock information of google, Microsoft, IBM and amazon between the years 2006 and 2018. There are a total of 3019 rows and 4 columns in each dataset.


**Stock Market Analysis:**

In the analysis part of the project, we find the following:

1. Factors effecting : The columns in the dataset are open, close, high, low and volume values of the stock for that day. These are the factors effecting the share price per day. 

2. Descriptive analysis: This includes the measures of central tendency like mean and median and measures of dispersion like quartiles and standard deviation.

3. Distribution of open and close: This is a graphical way of representing the frequencies of each open and close value

4. Correlation between close and open: Correlation implies the closeness or degree of dependency of one value on the other. Here, the correlation between open and close values is checked. 

5. Expanding window functions: These are used in calculations of time series data where there is continuous increment in the amount of data.

6. Time series decomposition: This method shows the trend of the stock over a particular period, seasonal changes for recurring patterns and resid or noise in the taken data.



**Prediction:**

In prediction, first we should preprocess the data so that there are no null values or improper  values which may give a wrong result. Then after the data is normalized to bring them to a common range which avoids unnecessary noise datapoints.
Next the data is divided into training data and testing data in the ratio 4:1. The training data is given as input to the Deep learning algorithm and is trained. Then the testing data is given as input and the model provides the output. These outputs and the actual data are compared to calculate the performance of the model. 

The algorithm used in the this project is a deep learning algorithm named GRU that is Gated Recurrent Unit. It is mainly used in time series analysis and anomaly detection.

In this method, the data is converted to sequences or patterns which are recognizable and then the prediction is done based on a single or multiple such patterns called steps. They are controlled by the layers used in the model like input layer, GRU layers and output layers.


**Libraries used:**

1. Pandas and numpy for data manipulation
2. matplotlib, seaborn,  plotly for visualization of data
3. pylab and statmodels for the time series decomposition
4. scikitlearn and tensorflow for normalization and scaling of data

**Outputs:**

![newplot (5)](https://github.com/user-attachments/assets/4dd06269-02b9-41e4-a192-848d66a0ddf8)
![newplot (1)](https://github.com/user-attachments/assets/1168f85a-f46f-4352-8777-e2bbb1c78e33)
![ibm](https://github.com/user-attachments/assets/a92809ac-3887-4b60-9c6c-d30da8bad2bd)
![download14](https://github.com/user-attachments/assets/b07dfd71-a040-4862-9357-0c8a85c2314d)
