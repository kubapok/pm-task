import pandas as pd
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
sns.set(color_codes=True)

str_weekday = {0: 'Monday',
               1: 'Tuesday',
               2: 'Wendsday',
               3: 'Thursday',
               4: 'Friday',
               5: 'Saturday',
               6: 'Sunday'}


report = pd.read_csv('PRSA_data_2010.1.1-2014.12.31.csv', sep=',')
r = report
r.describe()
r['datetime'] = pd.Series([datetime.datetime(y, m, d, h)
                           for y, m, d, h in zip(r.year, r.month, r.day, r.hour)])
r['weekday'] = pd.Series([x.weekday() for x in r['datetime']])


def plot_days(r, start_date, end_date):
    plot_days_report = r[(r.datetime > start_date) & (r.datetime < end_date)]

    timedelta = end_date - start_date
    plt.locator_params(numticks=timedelta.days)
    ax = sns.tsplot(plot_days_report['pm2.5'])
    ax.set(xticks=[24 * i for i in range(timedelta.days)])
    ax.set_xticklabels([str_weekday[(start_date.weekday() + i) % 7] + ' 0h'
                        for i in range(timedelta.days)], rotation=45, fontsize=8)
    ax.set_xlabel('Day of week')
    ax.set_ylabel('PM2.5 concentration (ug/m^3) ')
    plt.title(str(start_date.year) + ' ' + start_date.strftime('%b') +
              ' ' + str(start_date.day) + ' (' + start_date.strftime('%a') + ') - ' + 
    str(end_date.year) + ' ' + end_date.strftime('%b') +
              ' ' + str(end_date.day) + ' (' + end_date.strftime('%a') + ')')
    plt.show()


plot_days(r, datetime.datetime(2012, 1, 1), datetime.datetime(2012, 1, 15))

# sns.tsplot(r20120101_01_07['pm2.5'], time = r20120101_01_07['datetime'])
