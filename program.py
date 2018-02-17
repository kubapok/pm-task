import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import datetime
import seaborn as sns


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


if __name__ == "__main__":
    sns.set(color_codes=True)
    report = get_report()
    report.isnull().sum()
    report.describe()

    pal = sns.color_palette()[:5]
    plt.locator_params(numticks=12)
    for year, color in zip(range(2010,2015),pal):
        ax = sns.tsplot(get_means_in_months(report, year), color = color)
        ax.legend(color)

    red_patch = mpatches.Patch(color='red', label='The red data', linewidth = 0.01)
    blue_patch = mpatches.Patch(color='blue', label='The blue data')

    plt.legend(handles=[red_patch, blue_patch])

    ax.set(xticks=[i for i in range(12)])
    ax.set_xticklabels([datetime.datetime(2000,i,1).strftime("%B") for i in range(1,13)], rotation=45, fontsize=8)
    plt.show()
    # ax.set(xticks=[24 * i for i in range(timedelta.days)])

    # plot_months(report, plot_months
    #           datetime.datetime(2013, 1))
