import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import datetime
import seaborn as sns
import copy
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def get_report(report_name='PRSA_data_2010.1.1-2014.12.31.csv'):
    report = pd.read_csv(report_name, sep=',')
    report['datetime'] = pd.Series([datetime.datetime(y, m, d, h)
                                    for y, m, d, h in zip(report.year, report.month, report.day, report.hour)])
    return report


def plot_days(report, start_date, end_date):
    plot_days_report = report[(report.datetime > start_date) & (
        report.datetime < end_date)]

    timedelta = end_date - start_date
    plt.locator_params(numticks=timedelta.days)
    ax = sns.tsplot(plot_days_report['pm2.5'])
    ax.set(xticks=[24 * i for i in range(timedelta.days)])
    ax.set_xticklabels([(start_date + datetime.timedelta(days=i)).strftime("%d %b (%a) 0h")
                        for i in range(timedelta.days)], rotation=45, fontsize=8)
    ax.set_ylabel('PM2.5 concentration (ug/m^3) ')
    plt.title(start_date.strftime("%d %B %Y (%a)") +
              ' - ' + end_date.strftime("%d %B %Y (%a)"))
    plt.show()


def get_means_in_months(report, year):
    report_year = report[report.year == year]
    month_means = pd.Series([report_year[report_year['month'] == x]['pm2.5'].mean()
                             for x in range(1, 13)])
    return month_means


def plot_months(report,):
    pal = sns.color_palette()[:5]
    plt.locator_params(numticks=12)
    legend_col = []
    for year, color in zip(range(2010, 2015), pal):
        ax = sns.tsplot(get_means_in_months(report, year), color=color)
        ax.legend(color)
        legend_col.append(mpatches.Patch(
            color=color, label='year ' + str(year)))
    plt.legend(handles=legend_col)
    ax.set(xticks=[i for i in range(12)])
    ax.set_xticklabels([datetime.datetime(2000, i, 1).strftime("%B")
                        for i in range(1, 13)], rotation=45, fontsize=8)
    ax.set_ylabel('PM2.5 concentration (ug/m^3) ')
    plt.show()


def plot_normalized(report, start_date, end_date):
    r2 = copy.deepcopy(report)
    r2 = r2[(r2.datetime > start_date) &
            (r2.datetime < end_date)]
    del r2['No'], r2['year'], r2['month'], r2['day'], r2['hour'], r2['cbwd'], r2['Ir']
    del r2['datetime']

    r2['pm2.5'] = r2['pm2.5'].fillna(r2['pm2.5'].mean())
    scaler = MinMaxScaler()
    r2_scaled = pd.DataFrame(scaler.fit_transform(r2), columns=r2.columns)
    r2_scaled.plot()
    plt.show()


def plot_lms(report):
    r2 = copy.deepcopy(report)
    r2['pm2.5'] = r2['pm2.5'].fillna(r2['pm2.5'].mean())
    print(np.corrcoef(report['pm2.5'], report['TEMP']))
    print(np.corrcoef(report['pm2.5'], report['DEWP']))

    for col in ("DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"):
        sns.lmplot(x=col, y="pm2.5", data=report)
        plt.show()


def plot_lms2(report):
    r2 = copy.deepcopy(report)
    r2['pm2.5'] = r2['pm2.5'].fillna(r2['pm2.5'].mean())
    print(np.corrcoef(report['pm2.5'], report['TEMP']))
    print(np.corrcoef(report['pm2.5'], report['DEWP']))

    sns.lmplot(x="Iws", y="pm2.5", data=report, col="cbwd")
    plt.show()


if __name__ == "__main__":
    sns.set(color_codes=True)
    report = get_report()
    report.isnull().sum()
    report.describe()

    # sns.violinplot(x=report["pm2.5"])
    # plt.show()
    # plot_days(report, datetime.datetime(2012, 1, 1),
    #           datetime.datetime(2012, 1, 8))
    # plot_months(report)
    # plot_normalized(report, datetime.datetime(2012, 1, 1),
    #                 datetime.datetime(2012, 1, 8))
    # plot_lms(report)
    # plot_lms2(report)
    # sns.boxplot(x="cbwd", y="pm2.5", data=report)
    # plt.show()

    sns.boxplot(x="month", y="pm2.5", data=report)
    plt.show()

    sns.boxplot(x="year", y="pm2.5", data = report, hue = "month")
    plt.show()

    # del r2['No'], r2['year'], r2['month'], r2['day'], r2['hour'], r2['cbwd'], r2['Ir']
