import data
import string
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt  
plt.rc('font', family='Malgun Gothic')

MIN = 0
MAX = 2000 #-1
scaler = MinMaxScaler()

#fix_noise_on_stksplit = function parameter that returns normal price from noised price
def get_normalized_prices(price_series, fix_noise_on_stksplit=False):
    price_series = price_series.apply(lambda pstr:str(pstr).translate(str.maketrans('', '', ',')))
    if(fix_noise_on_stksplit):
        price_series = price_series.apply(fix_noise_on_stksplit)
    return scaler.fit_transform(price_series.values.reshape(-1, 1))

x = data.D(1).iloc[MIN:MAX][data.DATE_TOKEN].values
for i in range(1, 3):
    d = data.Select(data.D(i).iloc[MIN:MAX], [data.PRICE_TOKEN])
    plt.plot(x, get_normalized_prices(d[data.PRICE_TOKEN]))
d = data.Select(data.D(3).iloc[MIN:MAX], [data.PRICE_TOKEN])
plt.plot(x, get_normalized_prices(d[data.PRICE_TOKEN], lambda p: p if(float(p) <= 1000000.0) else float(p)/50.0))

plt.legend(["KOSPI200FIDX", "NASDAQ100FIDX", "SAMSUNGELECSTK"])
plt.ylabel("정규화된 수치")
plt.xlabel("{0}--> 날짜 진행 -->{1}".format(x[0], x[-1]))
plt.xticks([], color='w')

plt.show()