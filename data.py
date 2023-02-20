import pandas as pd

datasets = ["kospi200f_index.csv", "nasdaq100f_index.csv", "samsungelect_stock.csv"]

DATE_TOKEN = "Date"
PRICE_TOKEN = "Price"
OPEN_TOKEN = "Open"
HIGH_TOKEN = "High"
LOW_TOKEN = "LOW"
ChangeP_TOKEN = "Change %"

#args = 1/2/3, return pandas
def D(i):
    return pd.read_csv(datasets[i-1]).loc[::-1] #역순

#args = df, [TOKEN, TOKEN ...], return df[tokens]
def Select(D, tokens):
    return D[tokens]