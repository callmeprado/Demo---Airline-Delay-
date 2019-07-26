#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imports 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.offline as py
import plotly.figure_factory as ff
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
from plotly import tools
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot 
init_notebook_mode(connected = True)
cf.go_offline()

import seaborn as sns
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')

import datetime
import time
from time import strftime, gmtime

import os 
os.getcwd()
os.chdir('/Users/pradyutshukla/Desktop/Data Science material /Datasets/Airline Delays')


# In[2]:


df_flights = pd.read_csv('flights.csv', low_memory = False)


# In[3]:


df_flights.head()


# In[4]:


df_flights.loc[:,('YEAR','MONTH','DAY')].dtypes


# In[5]:


#Checking the count for each column 
df_flights.count()


# In[6]:


#Checking for each column type 
df_flights.dtypes


# In[7]:


# converting Input time value to datetime:

def conv_time(time_val):
    if pd.isnull(time_val):
        return np.nan
    else:
            # replace 24:00 o'clock with 00:00 o'clock:
        if time_val == 2400: time_val = 0
            # creating a 4 digit value out of input value:
        time_val = "{0:04d}".format(int(time_val))
            # creating a time datatype out of input value: 
        time_formatted = datetime.time(int(time_val[0:2]), int(time_val[2:4]))
    return time_formatted

## converting required columns to datetime time format and write it back into the dataframe: 
df_flights['ARRIVAL_TIME'] = df_flights['ARRIVAL_TIME'].apply(conv_time)
df_flights['DEPARTURE_TIME'] = df_flights['DEPARTURE_TIME'].apply(conv_time)
df_flights['SCHEDULED_DEPARTURE'] = df_flights['SCHEDULED_DEPARTURE'].apply(conv_time)
df_flights['WHEELS_OFF'] = df_flights['WHEELS_OFF'].apply(conv_time)
df_flights['WHEELS_ON'] = df_flights['WHEELS_ON'].apply(conv_time)
df_flights['SCHEDULED_ARRIVAL'] = df_flights['SCHEDULED_ARRIVAL'].apply(conv_time)


# In[8]:


df_flights.dtypes 


# In[9]:


# null value analysing function.
# gives some infos on columns types and number of null values:
def nullAnalysis(df):
    tab_info=pd.DataFrame(df.dtypes).T.rename(index={0:'column type'})

    tab_info=tab_info.append(pd.DataFrame(df.isnull().sum()).T.rename(index={0:'null values (nb)'}))
    tab_info=tab_info.append(pd.DataFrame(df.isnull().sum()/df.shape[0]*100)
                         .T.rename(index={0:'null values (%)'}))
    return tab_info

nullAnalysis(df_flights)


# In[10]:


""" The following itmes have a lot of missing(NaN) values: 
- CANCELLATION_REASON
- AIR_SYSTEM_DELAY
- SECURITY_DELAY
- AIRLINE_DELAY
- LATE_AIRCRAFT_DELAY
- WEATHER_DELAY 

Do the Null values affect our results? """


# In[11]:


#Analyzing selected columns where AIRLINE_DELAY isnot null
df_flights.loc[df_flights['AIRLINE_DELAY'].notnull(), ['AIRLINE_DELAY','AIR_SYSTEM_DELAY','SECURITY_DELAY',
                                                       'AIRLINE_DELAY','LATE_AIRCRAFT_DELAY','WEATHER_DELAY']].head()


# In[12]:


#It's ok to transform the NAN-data to the value "0.0" because 
#there was no impact on the flight by these data that causes a delay:

df_flights['AIRLINE_DELAY'] = df_flights['AIRLINE_DELAY'].fillna(0)
df_flights['AIR_SYSTEM_DELAY'] = df_flights['AIR_SYSTEM_DELAY'].fillna(0)
df_flights['SECURITY_DELAY'] = df_flights['SECURITY_DELAY'].fillna(0)
df_flights['LATE_AIRCRAFT_DELAY'] = df_flights['LATE_AIRCRAFT_DELAY'].fillna(0)
df_flights['WEATHER_DELAY'] = df_flights['WEATHER_DELAY'].fillna(0)


# In[13]:


nullAnalysis(df_flights)


# In[14]:


# % of Cancellation Reasons is still very high for the missing values 
# Can't ignore the remaining for the sake of model accuracy 
df_flights.loc[df_flights['CANCELLATION_REASON'].notnull(),['CANCELLATION_REASON']].head(15)


# In[15]:


"""
The reason for cancellation of flights splits into the following occurrences:

A - Airline/Carrier
B - Weather
C - National Air System
D - Security
Checking for the counts of each occurence... """


# In[16]:


# grouping by CANCELLATION_REASON to see the ration
df_flights['CANCELLATION_REASON'].value_counts()


# In[17]:


# converting categoric value to numeric
df_flights.loc[df_flights['CANCELLATION_REASON'] == 'A', 'CANCELLATION_REASON'] = 1
df_flights.loc[df_flights['CANCELLATION_REASON'] == 'B', 'CANCELLATION_REASON'] = 2
df_flights.loc[df_flights['CANCELLATION_REASON'] == 'C', 'CANCELLATION_REASON'] = 3
df_flights.loc[df_flights['CANCELLATION_REASON'] == 'D', 'CANCELLATION_REASON'] = 4

# -----------------------------------
# converting NaN data to numeric zero
df_flights['CANCELLATION_REASON'] = df_flights['CANCELLATION_REASON'].fillna(0)


# In[18]:


nullAnalysis(df_flights)


# In[19]:


"""ELAPSED_TIME = TAXI_OUT + AIR_TIME + TAXI_IN

    AIR_TIME = WHEELS_ON - WHEELS_OFF

    ARRIVAL_TIME = WHEELS_ON + TAXI_IN
    
  . . .the values to calculate these times are also NaN - data. That is probably 
    the reason for its initial NaN - data value. Will declare the data as outliers. """


# In[20]:


# drop the last 1% of missing data rows.
df_flights = df_flights.dropna(axis=0)


# In[21]:


#Analyzing Distribution after Cleansing, Conversion and Preprocessing
df_times = df_flights[
[
    'SCHEDULED_DEPARTURE','DEPARTURE_TIME','DEPARTURE_DELAY','TAXI_OUT','WHEELS_OFF',
    'SCHEDULED_TIME','ELAPSED_TIME','AIR_TIME','DISTANCE','WHEELS_ON','TAXI_IN',
    'SCHEDULED_ARRIVAL','ARRIVAL_TIME','ARRIVAL_DELAY','DIVERTED','CANCELLED','CANCELLATION_REASON',
    'AIR_SYSTEM_DELAY','SECURITY_DELAY','AIRLINE_DELAY','LATE_AIRCRAFT_DELAY','WEATHER_DELAY'
]]

pd.set_option('float_format', '{:f}'.format)

df_times.describe()


# In[22]:


df_airlines = pd.read_csv('airlines.csv')
df_airlines


# In[23]:


df_flights['AIRLINE'].value_counts()


# In[24]:


#MERGING AIRLINES WITH FLIGHTS DATASET 
df_flights = df_flights.merge(df_airlines, left_on='AIRLINE', right_on='IATA_CODE', how='inner')


# In[25]:


df_flights.head()


# In[26]:


# dropping old column and inserting new one
df_flights.insert(loc=5, column='AIRLINE', value=df_flights.AIRLINE_y)
df_flights.insert(loc=6, column='A_Code', value=df_flights.IATA_CODE)


# In[27]:


df_flights = df_flights.drop(['AIRLINE_y','IATA_CODE'], axis=1)
df_flights = df_flights.drop(['AIRLINE_x'], axis=1)
df_flights.head()


# In[28]:


df_airport = pd.read_csv('airports.csv')
df_airport.head()
df_airport['IATA_CODE'].nunique()


# In[29]:


#MERGING AIRPORT WITH FLIGHTS DATASET 
df_flights = pd.merge(df_flights,df_airport[['IATA_CODE','AIRPORT','CITY']], left_on='ORIGIN_AIRPORT', right_on = 'IATA_CODE')


# In[30]:


df_flights = df_flights.drop(['IATA_CODE'], axis=1)
df_flights = pd.merge(df_flights,df_airport[['IATA_CODE','AIRPORT','CITY']], left_on='DESTINATION_AIRPORT', right_on = 'IATA_CODE')
df_flights = df_flights.drop(['IATA_CODE'], axis=1)


# In[31]:


df_flights.head()


# In[265]:


dff = df_flights.CITY_x.value_counts()[:10]


# In[273]:


dff.iplot(kind = 'bar', xTitle = 'Top 10 cities', yTitle = 'Number of flights in a month')


# In[267]:


dff1 = df_flights.CITY_x.value_counts()

trace = go.Bar(
    x=dff1.index,
    y=dff1.values,
    marker=dict(
        color = dff1.values,
        colorscale='Jet',
        showscale=True
    )
)

data = [trace]
layout = go.Layout(
    title='Origin City Distribution', 
    yaxis = dict(title = '# of Flights')
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# In[125]:


state_count = []
store = {}
count = 1
for i in df_airport['STATE'].unique().tolist():
    store[i] = len(df_airport[df_airport['STATE'] == i])

for i, j in store.items():
    state_count.append(j)
#state_count


# In[108]:


data = dict(type = 'choropleth', locations = df_airport['STATE'].unique().tolist(), 
           locationmode = 'USA-states', 
           colorscale = 'Portland', 
           text = ['Philly', 'Texas'], 
           z = state_count, 
           colorbar = {'title': 'Airport Density'})
layout = dict(geo = {'scope':'usa'})
choromap = go.Figure(data = [data], layout = layout)
iplot(choromap)


# In[124]:


dff = df_flights.AIRLINE.value_counts()
dff.iplot(kind = 'bar', xTitle = 'Airliners', yTitle = 'Number of flights in a year')


# In[126]:


df_flights.head()


# In[213]:


df_JAN = df_flights[df_flights['MONTH']==1]  
JAN = len(df_JAN)
df_FEB = df_flights[df_flights['MONTH']==2]
FEB = len(df_FEB)
df_MAR = df_flights[df_flights['MONTH']==3]
MAR = len(df_MAR)
df_APR = df_flights[df_flights['MONTH']==4]
APR = len(df_APR)
df_MAY = df_flights[df_flights['MONTH']==5]
MAY = len(df_MAY)
df_JUN = df_flights[df_flights['MONTH']==6]
JUN = len(df_JUN)
df_JUL = df_flights[df_flights['MONTH']==7]
JUL = len(df_JUL)
df_AUG = df_flights[df_flights['MONTH']==8]
AUG = len(df_AUG)
df_SEP = df_flights[df_flights['MONTH']==9]
SEP = len(df_SEP)
df_OCT = df_flights[df_flights['MONTH']==10]
OCT = len(df_OCT)
df_NOV = df_flights[df_flights['MONTH']==11]
NOV = len(df_NOV)
df_DEC = df_flights[df_flights['MONTH']==12]
DEC = len(df_DEC)


# In[197]:


data = {'Months':['Jan', 'Feb', 'March', 'April', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct',
                  'Nov', 'Dec'], 'No_of_flights':[JAN, FEB, MAR, APR, MAY, JUN, JUL, AUG, SEP, OCT, NOV, DEC]}


# In[198]:


Monthly_flights = pd.DataFrame(data)


# In[199]:


Monthly_flights


# In[212]:


trace = go.Bar(
    x=Monthly_flights.Months,
    y=Monthly_flights.No_of_flights,
    marker=dict(
        color = Monthly_flights.No_of_flights,
        colorscale='Greens',
        showscale=True)
)

data = [trace]
layout = go.Layout(
    title='Yearly Flight patterns', 
    xaxis= dict(title = 'Months'), yaxis = dict(title ='Number of flights(monthly)')
                                                
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# In[295]:


dayOfWeek={1:'Monday', 2:'Tuesday', 3:'Wednesday', 4:'Thursday', 5:'Friday', 6:'Saturday', 7:'Sunday'}
df_DOW = df_flights.DAY_OF_WEEK.value_counts()


# In[296]:


df_DOW = df_DOW.to_frame().sort_index()


# In[312]:


trace = go.Bar(
    x=df_DOW.index,
    y=df_DOW.DAY_OF_WEEK,
    marker=dict(
        color = df_DOW.DAY_OF_WEEK,
        colorscale='Picnic',
        showscale=True)
)

data = [trace]
layout = go.Layout(
    title='Weekly Flight patterns', 
    xaxis= dict(title = 'Days'), yaxis = dict(title ='Number of flights(weekly)')
                                                
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# In[220]:


df_DOW


# In[226]:


df_Air = df_flights.groupby('AIRLINE').DEPARTURE_DELAY.mean().to_frame().sort_values(by='DEPARTURE_DELAY',
                                                    ascending=False).round(2)
trace1 = go.Bar(
    x=df_Air.index,
    y=df_Air.DEPARTURE_DELAY,
    name='Departure_delay',
    marker=dict(
        color = 'blue'
    )
)

df_Air = df_flights.groupby('AIRLINE').ARRIVAL_DELAY.mean().to_frame().sort_values(by='ARRIVAL_DELAY',
                                                    ascending=False).round(2)
trace2 = go.Bar(
    x=df_Air.index,
    y=df_Air.ARRIVAL_DELAY,
    name='Arrival_delay',
    marker=dict(
        color = 'navy'
    )
)

data = [trace1, trace2]
layout = go.Layout(xaxis=dict(tickangle=15), title='Mean Arrival & Departure Delay by Airlines',
    yaxis = dict(title = 'minutes'), 
                   barmode='stack')

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# In[229]:


df_flights['DEP_ARR_DIFF'] = df_flights['DEPARTURE_DELAY'] - df_flights['ARRIVAL_DELAY']
df_mean = df_flights.groupby('AIRLINE').DEP_ARR_DIFF.mean().to_frame().sort_values(by='DEP_ARR_DIFF',
                                                    ascending=False).round(2)

trace = go.Bar(
    x=df_mean.index,
    y=df_mean.DEP_ARR_DIFF,
    marker=dict(
        color = df_mean.DEP_ARR_DIFF,
        colorscale='Rainbow',
        showscale=True
    )
)

data = [trace]
layout = go.Layout(xaxis=dict(tickangle=15),
    title='Mean (Departure Delay - Arrival Delay) by Airlines', 
                   yaxis = dict(title = 'minute')
                  )

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# In[301]:


import seaborn as sns 
import matplotlib.pyplot as plt 

#No. of delays
sns.set(style="whitegrid")

fig_dim = (16,14)
f, ax = plt.subplots(figsize=fig_dim)
sns.despine(bottom=True, left=True)

#Showing each observation with a scatterplot
sns.stripplot(x="ARRIVAL_DELAY", y="AIRLINE",
              data=df_flights, dodge=True, jitter=True
            )


# In[281]:


df_city = df_flights.groupby('CITY_x').DEPARTURE_DELAY.mean().to_frame().sort_values(by='DEPARTURE_DELAY',
                                                        ascending=False)[:10].round(2)
trace1 = go.Bar(
    x=df_city.index,
    y=df_flights.DEPARTURE_DELAY,
    marker=dict(
        color = 'red'
    )
)

df_city = df_flights.groupby('CITY_y').ARRIVAL_DELAY.mean().to_frame().sort_values(by='ARRIVAL_DELAY',
                                                        ascending=False)[:10].round(2)

trace2 = go.Bar(
    x=df_city.index,
    y=df_city.ARRIVAL_DELAY,
    marker=dict(
        color = 'navy'
    )
)

fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('Mean Departure Delay by City', 
                                                          'Mean Arrival Delay by City'))
fig.append_trace(trace1, 1,1)
fig.append_trace(trace2, 1,2)

fig['layout'].update(yaxis = dict(title = 'Minutes'), height=500, width=850, 
                     title='Systematic delay related to departure or arrival city?',  
                     showlegend=False)                    
py.iplot(fig)


# In[306]:


df_city = df_flights.groupby('CITY_x').DEPARTURE_DELAY.mean().to_frame().sort_values(by='DEPARTURE_DELAY',
                                                        ascending=False).round(2)
trace1 = go.Bar(
    x=df_city.index,
    y=df_flights.DEPARTURE_DELAY,
    marker=dict(
        color = 'red'
    )
)

df_city = df_flights.groupby('CITY_y').ARRIVAL_DELAY.mean().to_frame().sort_values(by='ARRIVAL_DELAY',
                                                        ascending=False).round(2)

trace2 = go.Bar(
    x=df_city.index,
    y=df_city.ARRIVAL_DELAY,
    marker=dict(
        color = 'navy'
    )
)

fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('Mean Departure Delay by City', 
                                                          'Mean Arrival Delay by City'))
fig.append_trace(trace1, 1,1)
fig.append_trace(trace2, 1,2)

fig['layout'].update(yaxis = dict(title = 'Minutes'), height=500, width=850, 
                     title='Systematic delay related to departure or arrival city?',  
                     showlegend=False)                    
py.iplot(fig)


# In[302]:


# Dataframe correlation
Corr_matrix = df_flights.corr()


f, ax = plt.subplots(figsize=(11, 9))

sns.heatmap(Corr_matrix)


# Results from Correlation Matrix
# Positive correlations(x > 0.6) and the less positive correlations(0.2 < x < 0.6):
#     
# Positive correlations between:
# DEPARTURE_DELAY and
# -ARRIVAL_DELAY
# -LATE_AIRCRAFT_DELAY
# -AIRLINE_DELAY
# 
# ARRIVAL_DELAY and
# -DEPARTURE_DELAY
# -LATE_AIRCRAFT_DELAY
# -AIRLINE_DELAY
# 
# Less positive correlations between:
# ARRIVAL_DELAY and
# -AIR_SYSTEM_DELAY
# -WEATHER_DELAY
# 
# DEPARTURE_DELAY and
# -AIR_SYSTEM_DELAY
# -WEATHER_DELAY
# 
# TAXI_OUT and
# -AIR_SYSTEM_DELAY
# -ELAPSED_TIME
# 
# This leads to the following factors of influence:
# Which features have the most counted influence on different other features?
# 
# Positive Value	Count	Type
# ++	2	LATE_AIRCRAFT_DELAY
# ++	2	AIRLINE_DELAY
# ++	1	ARRIVAL_DELAY
# +-	3	AIR_SYSTEM_DELAY
# +-	2	WEATHER_DELAY
# +-	1	ELAPSED_TIME
# 
# Need to proof the above written down feature correlation count with a ML algo. Do they really correlate as good with the ARRIVAL_DELAY as perceived ? 
# Will split the data into delayed and not delayed data and define a label (DELAYED) for that in the dataframe. 
# Then, will show the feature importance for the given attributes.
# 
# 
# 

# In[32]:


#First month df : easier computation 
df_flights_jan = df_flights.loc[(df_flights.loc[:,'YEAR'] == 2015 ) & (df_flights.loc[:,'MONTH'] == 1 )]


# In[33]:


df_flights_jan['DELAYED'] = df_flights_jan.loc[:,'ARRIVAL_DELAY'].values > 0 


# In[34]:



# Choosing the predictors
feature_list_s = [
    'LATE_AIRCRAFT_DELAY'
    ,'AIRLINE_DELAY'
    ,'AIR_SYSTEM_DELAY'
    ,'WEATHER_DELAY'
    ,'ELAPSED_TIME']

# New dataframe based on a small feature list
x_small = df_flights_jan[feature_list_s]
y = df_flights_jan.DELAYED


# In[35]:


#maschine learning libraries
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# RandomForestClassifier(test classifier) fitted on the small feature set :
clf = RandomForestClassifier(n_estimators = 10, random_state=20) 
clf.fit(x_small, y)


# In[36]:


#Extracting feature importance for each feature
i=0
df_feature_small = pd.DataFrame(columns=['FEATURE','IMPORTANCE'])
for val in (clf.feature_importances_):
    df_feature_small.loc[i] = [feature_list_s[i],val]
    i = i + 1
    
df_feature_small.sort_values('IMPORTANCE', ascending=False)


# Now the AIR_SYSTEM__DELAY has got most influences on a flight that has been delayed.
# We have classified the data into delayed and not
# delayed data and want to find out now which of these features affects arrival delay of a flight the most. 

# In[37]:


# choosing the predictors
feature_list = ['YEAR','MONTH','DAY','AIRLINE' ,'LATE_AIRCRAFT_DELAY' ,'AIRLINE_DELAY'
    ,'AIR_SYSTEM_DELAY','WEATHER_DELAY','ELAPSED_TIME'   ,'DEPARTURE_DELAY' ,'SCHEDULED_TIME' ,
    'AIR_TIME' ,'DISTANCE' ,'TAXI_IN','TAXI_OUT','DAY_OF_WEEK','SECURITY_DELAY']

X = df_flights_jan[feature_list]


# In[38]:


# Label encoding of AIRLINE and write this back to df
from sklearn.preprocessing import LabelEncoder
labelenc = LabelEncoder()

# Converting "category" airline to integer values
X.iloc[:,feature_list.index('AIRLINE')] = labelenc.fit_transform(X.iloc[:,feature_list.index('AIRLINE')])

# Convert my encoded categories back
labelenc.inverse_transform(X.iloc[:, feature_list.index('AIRLINE')])


# In[39]:


# Fit the new features and the label (based on feature_list)
clf = RandomForestClassifier(n_estimators=10, random_state=20) 
clf.fit(X, y)

i=0
df_feature_selection = pd.DataFrame(columns=['FEATURE','IMPORTANCE'])
for val in (clf.feature_importances_):
    df_feature_selection.loc[i] = [feature_list[i],val]
    i = i + 1
    

df_feature_selection.sort_values('IMPORTANCE', ascending=False)


# - AIR_SYSTEM_DELAY still stays nearly at the top,
# - LATE_AIRCRAFT_DELAY, AIRLINE_DELAY, WEATHER_DELAY and ELAPSED_TIME moved down some positions. 
# 
# The ELAPSED_TIME remains in the top five but our other features have got a different importance given by the other features.
# 
# - ELAPSED_TIME was the worst feature of our first calculation and now stays at the top five.
# 
# Will use this info later, while pruning decision trees from RF to maximize model accuracy. 
# 
# 

# In[40]:


# BUILDING THE MODEL 

forest_model = RandomForestRegressor(n_estimators = 100, random_state=42)
y = df_flights_jan.ARRIVAL_DELAY
y = np.array(y)
X = np.array(X)

train_X, val_X, train_y, val_y = train_test_split(X, y, test_size = 0.35, random_state = 42)


# In[41]:


print('Training Features Shape:', train_X.shape)
print('Training Labels Shape:', train_y.shape)
print('Testing Features Shape:', val_X.shape)
print('Testing Labels Shape:', val_y.shape)


# In[42]:


# Average arrival delay for our dataset(dividing every array element by the given count)
baseline_preds = df_flights_jan['ARRIVAL_DELAY'].agg('sum') / df_flights_jan['ARRIVAL_DELAY'].agg('count') 

# Baseline error by average arrival delay 
baseline_errors = abs(baseline_preds - val_y)
print('Average baseline error: ', round(np.mean(baseline_errors),2))


# In[43]:


# Fit the model
forest_model.fit(train_X, train_y)

# Predict the target based on testdata 
flightdelay_pred= forest_model.predict(val_X)

#Calculate the absolute errors
errors = abs(flightdelay_pred - val_y)

print('Mean Absolute Error: ', round(np.mean(errors),3), 'minutes.')


# In[44]:


val_y


# In[45]:


flightdelay_pred


# In[46]:


len(train_y)


# In[47]:


type(flightdelay_pred)


# In[48]:


type(val_y)


# In[49]:


prob_y = ( val_y > 0 ) 


# In[50]:


flightdelay_prob = ( flightdelay_pred > 0 )


# In[51]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(prob_y, flightdelay_prob)


# In[52]:


cm


# In[53]:


a = (93889+63425)/159955


# In[63]:


a


# In[55]:


from statistics import *

# Calculate the slope and intercept
def best_fit_slope_and_intercept(xs,ys):
    m = ( ((mean(xs) * mean(ys)) - mean(xs*ys)) /
          ((mean(xs) * mean(xs)) - mean(xs*xs)) )
    b = mean(ys) - m*mean(xs)
    return m, b

# Calculate the regression line
def regression_line(m, feature, b):
        regression_line = [(m*x) + b for x in feature]
        return regression_line

# Draw six grid scatter plot and calculate all necessary functions
def draw_sixgrid_scatterplot(feature1, feature2, feature3, feature4, feature5, feature6, target):
    fig = plt.figure(1, figsize=(16,15))
    gs=gridspec.GridSpec(3,3)
    
    # Axis for the grid
    ax1=fig.add_subplot(gs[0,0])
    ax2=fig.add_subplot(gs[0,1])
    ax3=fig.add_subplot(gs[0,2])
    ax4=fig.add_subplot(gs[1,0])
    ax5=fig.add_subplot(gs[1,1])
    ax6=fig.add_subplot(gs[1,2])
    
    # Drawing dots based on feature and target
    ax1.scatter(feature1, target, color = 'g')
    ax2.scatter(feature2, target, color = 'c')
    ax3.scatter(feature3, target, color = 'y')
    ax4.scatter(feature4, target, color = 'k')
    ax5.scatter(feature5, target, color = 'grey')
    ax6.scatter(feature6, target, color = 'm')
    
    # Get best fit for slope and intercept
    m1,b1 = best_fit_slope_and_intercept(feature1, target)
    m2,b2 = best_fit_slope_and_intercept(feature2, target)
    m3,b3 = best_fit_slope_and_intercept(feature3, target)
    m4,b4 = best_fit_slope_and_intercept(feature4, target)
    m5,b5 = best_fit_slope_and_intercept(feature5, target)
    m6,b6 = best_fit_slope_and_intercept(feature6, target)

    # Build regression lines
    regression_line1 = regression_line(m1, feature1, b1)
    regression_line2 = regression_line(m2, feature2, b2)
    regression_line3 = regression_line(m3, feature3, b3)
    regression_line4 = regression_line(m4, feature4, b4)
    regression_line5 = regression_line(m5, feature5, b5)
    regression_line6 = regression_line(m6, feature6, b6)
            
    # Plotting regression lines
    ax1.plot(feature1,regression_line1)
    ax2.plot(feature2,regression_line2)
    ax3.plot(feature3,regression_line3)
    ax4.plot(feature4,regression_line4)
    ax5.plot(feature5,regression_line5)
    ax6.plot(feature6,regression_line6)
    
    # Naming the axis
    ax1.set_xlabel(feature1.name)
    ax1.set_ylabel(target.name)
    ax2.set_xlabel(feature2.name)    
    ax2.set_ylabel(target.name)
    ax3.set_xlabel(feature3.name)
    ax3.set_ylabel(target.name)
    ax4.set_xlabel(feature4.name)
    ax4.set_ylabel(target.name)
    ax5.set_xlabel(feature5.name)
    ax5.set_ylabel(target.name)
    ax6.set_xlabel(feature6.name)
    ax6.set_ylabel(target.name)
    
    # Give the labels space
    plt.tight_layout()
    plt.show()


# In[56]:


# Determine the squared error
def squared_error_reg(ys_orig, ys_line):
    return sum((ys_line-ys_orig)**2)

# Calculating r-squared
def coefficient_of_determination(ys_orig, ys_line):
    y_mean:line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error_reg(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)


# In[97]:


# Draw the grid scatters
draw_sixgrid_scatterplot(df_flights_jan['DEPARTURE_DELAY'], df_flights_jan['AIR_SYSTEM_DELAY'],
                         df_flights_jan['SCHEDULED_TIME'], df_flights_jan['ELAPSED_TIME'],
                         df_flights_jan['TAXI_OUT'], df_flights_jan['AIRLINE_DELAY'],
                         df_flights_jan['ARRIVAL_DELAY'])


# In[57]:


#Checking data for the month of February (unknown data)
df_flights_feb = df_flights.loc[(df_flights.loc[:,'YEAR'] == 2015 ) & (df_flights.loc[:,'MONTH'] == 2 )]

X2 = df_flights_feb[feature_list]
Y2 = df_flights_feb.ARRIVAL_DELAY 


# In[58]:


# Converting "category" airline to integer values
X2.iloc[:,feature_list.index('AIRLINE')] = labelenc.fit_transform(X2.iloc[:,feature_list.index('AIRLINE')])

# Filling the features and the target again
X2 = np.array(X2)
Y2 = np.array(Y2)

# Predict the new data based on the old model (forest_model)
flightdelay_pred_feb = forest_model.predict(X2)

#Calculate the absolute errors
errors_feb = abs(flightdelay_pred_feb - Y2)


# In[59]:


# Mean Absolute Error im comparison
print('Mean Absolute Error January: ', round(np.mean(errors),3), 'minutes.')
print('---------------------------------------------------------------')
print('Mean Absolute Error February: ', round(np.mean(errors_feb),3), 'minutes.')


# In[60]:


df_flights_feb['DELAYED'] = df_flights_feb.loc[:,'ARRIVAL_DELAY'].values > 0 


# In[61]:


Y2_pred = (Y2 > 0)


# In[116]:


cm = confusion_matrix((flightdelay_pred_feb>0), Y2_pred)
cm


# In[117]:


len(flightdelay_pred_feb)


# In[118]:


(226723+174024)/407663


# In[62]:


#Checking the model for month of July 
df_flights_jul = df_flights.loc[(df_flights.loc[:,'YEAR'] == 2015 ) & (df_flights.loc[:,'MONTH'] == 7 )]

X7 = df_flights_jul[feature_list]
Y7 = df_flights_jul.ARRIVAL_DELAY 

# Converting "category" airline to integer values
X7.iloc[:,feature_list.index('AIRLINE')] = labelenc.fit_transform(X7.iloc[:,feature_list.index('AIRLINE')])

# Filling the features and the target again
X7 = np.array(X7)
Y7 = np.array(Y7)

# Predict the new data based on the old model (forest_model)
flightdelay_pred_jul = forest_model.predict(X7)

#Calculate the absolute errors
errors_jul = abs(flightdelay_pred_jul - Y7)

# Mean Absolute Error im comparison
print('Mean Absolute Error January: ', round(np.mean(errors),3), 'minutes.')
print('---------------------------------------------------------------')
print('Mean Absolute Error July: ', round(np.mean(errors_jul),3), 'minutes.')


# In[123]:


df_flights_jul['DELAYED'] = df_flights_jul.loc[:,'ARRIVAL_DELAY'].values > 0 
Y7_pred = (Y7 > 0)
cm = confusion_matrix((flightdelay_pred_feb>0), Y2_pred)
cm


# In[129]:


acc = accuracy_score((flightdelay_pred_jul>0), Y7_pred)
acc


# In[ ]:




