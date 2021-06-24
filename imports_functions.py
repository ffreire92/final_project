## IMPORTS ##

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import pylab as plt
plt.rcParams['figure.figsize']=(15, 15)
import warnings
warnings.simplefilter('ignore')

from statsmodels.tsa.arima_model import ARMA
from sklearn.metrics import mean_squared_error as mse

## FUNCTIONS ##

# plots #

plt.rcParams['figure.figsize']=(15, 15)

# Chart bars #

def graph_bar(kind, dataframe, column, title, color):
    '''it has as inputs kind ('vertical' or 'horizontal'), the dataframe we'll plot, the respective column, the title of it, and the colour'''
    if kind == 'vertical':

        dataframe[column].plot.barh(title=title, grid=1, yticks=[i for i in range(0, int(dataframe[column].max()), 5)],
                                    rot=0, fontsize=13, color=color, figsize=(15, 10))
        plt.show();

    elif kind == 'horizontal':
        dataframe[column].plot.bar(title=title, grid=1, yticks=[i for i in range(0, int(dataframe[column].max()), 5)],
                                   rot=0, fontsize=13, color=color, figsize=(15, 10))
        plt.show();


def severe_cases(dataframe):
    '''It has as input a dataframe and returns a stacked bar chart filtered by number of hospitalisation, Intensive Car Unit, and deaths'''
    dataframe[['Number of hospitalisation (%)', 'Number of Intensive Care Unit (%)',
               'Number of deaths (%)']].plot.bar(title='Severe Cases', grid=1,
                                                 yticks=[i for i in range(0, 100, 10)],
                                                 rot=0, fontsize=13, color=['Orange', 'Darkblue', 'Darkred'],
                                                 stacked=True, figsize=(15, 10))
# Evolution "

def overview(dataframe, title):
    '''It has as input a dataframe and the title to designate, and returns a line chart of the evolution over time'''
    if title == 'evolution':
        fig, axs = plt.subplots(4)
        fig.suptitle('Pandemic evolution')
        axs[0].plot(dataframe.date, dataframe.num_infections)
        axs[0].set_title('Number of infections')
        axs[1].plot(dataframe.date, dataframe.num_hosp)
        axs[1].set_title('Number of hospitalisations')
        axs[2].plot(dataframe.date, dataframe.num_uci)
        axs[2].set_title('Number of Intensive Care Unit')
        axs[3].plot(dataframe.date, dataframe.num_dead)
        axs[3].set_title('Number of deaths')

    if title == 'Number of infections':
        fig, axs = plt.subplots(3)
        fig.suptitle(title)
        axs[0].plot(dataframe.date, dataframe.num_infections, color='Darkgreen')
        axs[0].set_title('Daily')
        axs[1].plot(dataframe.date, dataframe.ave_7_num_infections, color='Darkgreen', linewidth=3)
        axs[1].set_title('Moving 7 day average')
        axs[2].plot(dataframe.date, dataframe.cumu_num_infections, color='Darkgreen', linewidth=3)
        axs[2].set_title('Cumulative');

    elif title == 'Number of hospitalisations':
        fig, axs = plt.subplots(3)
        fig.suptitle(title)
        axs[0].plot(dataframe.date, dataframe.num_hosp, color='Darkblue')
        axs[0].set_title('Daily')
        axs[1].plot(dataframe.date, dataframe.ave_7_num_hosp, color='Darkblue', linewidth=3)
        axs[1].set_title('Moving 7 day average')
        axs[2].plot(dataframe.date, dataframe.cumu_num_hosp, color='Darkblue', linewidth=3)
        axs[2].set_title('Cumulative');

    elif title == 'Number of Intensive Care Units':
        fig, axs = plt.subplots(3)
        fig.suptitle(title)
        axs[0].plot(dataframe.date, dataframe.num_uci, color='Orange')
        axs[0].set_title('Daily')
        axs[1].plot(dataframe.date, dataframe.ave_7_num_uci, color='Orange', linewidth=3)
        axs[1].set_title('Moving 7 day average')
        axs[2].plot(dataframe.date, dataframe.cumu_num_uci, color='Orange', linewidth=3)
        axs[2].set_title('Cumulative');

    elif title == 'Number of deaths':
        fig, axs = plt.subplots(3)
        fig.suptitle(title)
        axs[0].plot(dataframe.date, dataframe.num_dead, color='Darkred')
        axs[0].set_title('Daily')
        axs[1].plot(dataframe.date, dataframe.ave_7_num_dead, color='Darkred', linewidth=3)
        axs[1].set_title('Moving 7 day average')
        axs[2].plot(dataframe.date, dataframe.cumu_num_dead, color='Darkred', linewidth=3)
        axs[2].set_title('Cumulative');

# Moving averages #

def inspect_mov_ave(dataframe):
    '''it has as input a dataframe an return the comparison of different moving averages'''
    plt.rcParams['figure.figsize'] = (15, 15)
    dataframe = dataframe.copy()

    dataframe = dataframe[['date', 'num_infections']]

    for i in range(1, 15, 2):
        dataframe[f'move_ave_{i}'] = dataframe.iloc[:, 1].rolling(window=i).mean()

    fig, axs = plt.subplots(len(dataframe.columns))
    for i in range(1, len(dataframe.columns)):
        axs[i].plot(dataframe.date, dataframe[str(dataframe.columns[i])], color='Darkgreen')
        axs[i].set_title(str(dataframe.columns[i]))
        plt.rcParams['figure.figsize'] = (20, 20)

# Compare moving averages #

def compare_7mov_ave(bydate):
    '''receives a dataframe as input and returns a plot with two curves, the daily infections and the moving average of 7 days'''
    plt.plot(bydate[['date', 'ave_7_num_infections']].set_index('date'), 'r', label='7 days moving average',
             linewidth=4)
    plt.plot(bydate[['date', 'num_infections']].set_index('date'), label='Number of infections', color='Darkgreen')
    plt.legend(fontsize='xx-large')
    plt.show();

##Dataframe##

# Sorting data #

def sort_data(dataframe, column):
    '''It has as inputs the dataframe and a column, so it returns the same dataframe sorted in ascending order by that column'''
    return pd.DataFrame(dataframe[column].sort_values())

# Data cleaning #

def clean_data(db1):
    '''This functions receives db1 as input and performs data some data cleaning'''

    # Number of Infections greater than 1:
    db1 = db1[(db1.num_infections > 0)].reset_index()
    db1 = db1.drop(columns=['index'], axis=1)

    # Sort ascending order by date:
    db1['date'] = pd.to_datetime(db1.date)
    db1 = db1.sort_values(by="date").reset_index()
    db1 = db1.drop(columns=['index'], axis=1)

    # Data cleaning autonomous_region:
    db1['autonomous_region'] = db1.autonomous_region.apply(
        lambda x: 'Comunidad Valenciana' if x == 'Valenciana, Comunidad' else x)
    db1['autonomous_region'] = db1.autonomous_region.apply(
        lambda x: 'Comunidad de Madrid' if x == 'Madrid, Comunidad de' else x)
    db1['autonomous_region'] = db1.autonomous_region.apply(
        lambda x: 'Región de de Murcia' if x == 'Murcia, Región de' else x)
    db1['autonomous_region'] = db1.autonomous_region.apply(
        lambda x: 'Comunidad Foral de Navarra' if x == 'Navarra, Comunidad Foral de' else x)
    db1['autonomous_region'] = db1.autonomous_region.apply(
        lambda x: 'Principado de Asturias' if x == 'Asturias, Principado de' else x)

    # Data cleaning province:
    db1['province'] = db1.province.apply(lambda x: 'Alicante' if x == 'Alicante/Alacant' else x)
    db1['province'] = db1.province.apply(lambda x: 'Castellón' if x == 'Castellón/Castelló' else x)
    db1['province'] = db1.province.apply(lambda x: 'Araba' if x == 'Araba/Álava' else x)
    db1['province'] = db1.province.apply(lambda x: 'Valencia' if x == 'Valencia/València' else x)

    # Data cleaning sex and age_interval:
    db1 = db1[(db1.sex != 'NC') & (db1.age_interval != 'NC')]

    return db1

# Cumulative #

def cumulative(dataframe):
    '''It has as input a dataframe and calculates the cumulative frequency of the number of infections, hospitalisations, ICU and deaths'''
    cumu_num_infections = dataframe.num_infections.cumsum()
    cumu_num_hosp = dataframe.num_hosp.cumsum()
    cumu_num_uci = dataframe.num_uci.cumsum()
    cumu_num_dead = dataframe.num_dead.cumsum()

    cumulative = pd.DataFrame({'cumu_num_infections': cumu_num_infections, 'cumu_num_hosp': cumu_num_hosp,
                               'cumu_num_uci': cumu_num_uci, 'cumu_num_dead': cumu_num_dead})
    dataframe = pd.concat([dataframe, cumulative], axis=1)
    return dataframe

# Relative frequencies #

def freq_rel(dataframe):
    '''It has as input a dataframe and calculates the relative frequency of the number of infections, hospitalisations, ICU and deaths'''
    dataframe['Number of infection (%)'] = [round(i / dataframe['num_infections'].sum(), 3) * 100 for i in
                                            dataframe['num_infections']]
    dataframe['Number of hospitalisation (%)'] = [round(i / dataframe['num_hosp'].sum(), 3) * 100 for i in
                                                  dataframe['num_hosp']]
    dataframe['Number of Intensive Care Unit (%)'] = [round(i / dataframe['num_uci'].sum(), 3) * 100 for i in
                                                      dataframe['num_uci']]
    dataframe['Number of deaths (%)'] = [round(i / dataframe['num_dead'].sum(), 3) * 100 for i in dataframe['num_dead']]

    return dataframe


def organise(db1, column):
    '''It has as inputs db1 and specific columns of the dataframe, and returns a dataframe grouped by that column with four additional columns with the relative frequency of the number of infections, hospitalisations, ICU and deaths'''

    name = db1.groupby(column).sum()
    freq_rel(name)

    return name

# Moving averages: 7 days moving average #

def mov_7_ave(dataframe):
    '''It has as inputs a dataframe and it calculates the 7 day moving average for the number of infections, hospitalisations, ICU and deaths'''
    dataframe['ave_7_num_infections'] = dataframe.loc[:, 'num_infections'].rolling(window=7).mean()
    dataframe['ave_7_num_hosp'] = dataframe.loc[:, 'num_hosp'].rolling(window=7).mean()
    dataframe['ave_7_num_uci'] = dataframe.loc[:, 'num_uci'].rolling(window=7).mean()
    dataframe['ave_7_num_dead'] = dataframe.loc[:, 'num_dead'].rolling(window=7).mean()

    return dataframe

# Evolution #

def evolution(dataframe):
    '''It has as input a dataframe and it a new dataframe, grouped by date and with the cumulative and the 7 day moving average for the number of infections, hospitalisations, ICU and deaths. Note that date is not the index!'''
    bydate = dataframe.groupby('date').sum().reset_index()
    bydate = cumulative(bydate)
    bydate = mov_7_ave(bydate)

    return bydate

# Training models #

def training_models(bydate, measure):
    '''it has as input the by date dateframe, and returns the m7 day moving average according to the measure defined'''
    if measure == 'infections':

        dataframe = pd.DataFrame(bydate.loc[:, 'ave_7_num_infections'], columns=['ave_7_num_infections'])


    elif measure == 'hospitalisations':

        dataframe = pd.DataFrame(bydate.loc[:, 'ave_7_num_hosp'], columns=['ave_7_num_hosp'])


    elif measure == 'icu':

        dataframe = pd.DataFrame(bydate.loc[:, 'ave_7_num_uci'], columns=['ave_7_num_uci'])

    elif measure == 'deaths':

        dataframe = pd.DataFrame(bydate.loc[:, 'ave_7_num_dead'], columns=['ave_7_num_dead'])

    return dataframe[6:]


def add_days(dataframe):
    '''It has as input a dataframe and adds to it the days to be predicted'''
    dataframe = dataframe.append(pd.Series(name='2021-06-11 00:00:00'))
    dataframe = dataframe.append(pd.Series(name='2021-06-12 00:00:00'))
    dataframe = dataframe.append(pd.Series(name='2021-06-13 00:00:00'))
    dataframe = dataframe.append(pd.Series(name='2021-06-14 00:00:00'))
    dataframe = dataframe.append(pd.Series(name='2021-06-15 00:00:00'))
    dataframe = dataframe.append(pd.Series(name='2021-06-16 00:00:00'))
    dataframe = dataframe.append(pd.Series(name='2021-06-17 00:00:00'))

    return dataframe

def pred_infections(dataframe):
    '''It has as input a dataframe (infections) and trains it with ARMA model'''
    model = ARMA(dataframe, order=(4, 2)).fit(disp=False)
    aux = add_days(dataframe)
    train = aux[:-7]
    return pd.DataFrame(model.predict(len(train), len(aux) - 1), columns=['pred_num_infections'])

def pred_hosp(dataframe):
    '''It has as input a dataframe (hospitalisations) and trains it with ARMA model'''
    model = ARMA(dataframe, order=(19, 11)).fit(disp=False)
    aux = add_days(dataframe)
    train = aux[:-7]
    return pd.DataFrame(model.predict(len(train), len(aux) - 1), columns=['pred_num_hosp'])

def pred_uci(dataframe):
    '''It has as input a dataframe (icu) and trains it with ARMA model'''
    model = ARMA(dataframe, order=(2, 6)).fit(disp=False)
    aux = add_days(dataframe)
    train = aux[:-7]
    return pd.DataFrame(model.predict(len(train), len(aux) - 1), columns=['pred_num_uci'])

def pred_deaths(dataframe):
    '''It has as input a dataframe (deaths) and trains it with ARMA model'''
    model = ARMA(dataframe, order=(2, 6)).fit(disp=False)
    aux = add_days(dataframe)
    train = aux[:-7]
    return pd.DataFrame(model.predict(len(train), len(aux) - 1), columns=['pred_num_dead'])


