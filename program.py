import pandas as pd
import matplotlib.pyplot as plt
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


if __name__ == "__main__":
    sns.set(color_codes=True)
    report = get_report()
    report.describe()
    plot_days(report, datetime.datetime(2012, 5, 1),
              datetime.datetime(2012, 6, 1))
