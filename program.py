import pandas as pd
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
sns.set(color_codes=True)

str_weekday = {1: 'Monday',
               2: 'Tuesday',
               3: 'Wendsday',
               4: 'Thursday',
               5: 'Friday',
               6: 'Saturday',
               7: 'Sunday'}


report = pd.read_csv('PRSA_data_2010.1.1-2014.12.31.csv', sep=',')
r = report
r.describe()
r['datetime'] = pd.Series([datetime.datetime(y, m, d, h)
                           for y, m, d, h in zip(r.year, r.month, r.day, r.hour)])
r['weekday'] = pd.Series([x.weekday() for x in r['datetime']])

r2012 = r[r.year == 2012]


r20120101_01_07 = r[(r.year == 2012) & (r.month == 1)
                    & (2 <= r.day) & (r.day < 8)]
r20120101_01_07 = r[(r.year == 2010) & (r.month == 1)
                    & (1 <= r.day) & (r.day < 7)]
plt.locator_params(numticks=7)
ax = sns.tsplot(r20120101_01_07['pm2.5'])
ax.set(xticks=[24 * i for i in range(7)])  # $, xticklabels=list(range(7)))
ax.set_xticklabels(['Monday 0h', 'Tuesday 0h', 'Wendsday 0h',
                    'Th', 'F', 'Sa', 'Su'], rotation=45, fontsize=8)
ax.set_xlabel('Day of week')
ax.set_ylabel('PM2.5 concentration (ug/m^3) ')
plt.title('January 2012, Days: 02(Mon) - 08 (Sun)')
plt.show()





# sns.tsplot(r20120101_01_07['pm2.5'], time = r20120101_01_07['datetime'])
