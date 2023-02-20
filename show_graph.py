import data
import matplotlib.pyplot as plt  
plt.rc('font', family='Malgun Gothic')

fig, ax = plt.subplots()

b = 2000
d1 = data.Select(data.D(1).iloc[:b], [data.PRICE_TOKEN, data.DATE_TOKEN])
d2 = data.Select(data.D(2).iloc[:b], [data.PRICE_TOKEN, data.DATE_TOKEN])
#d3 = data.Select(data.D(3).iloc[:b], [data.PRICE_TOKEN, data.DATE_TOKEN])

c = 30
date_x = d1[data.DATE_TOKEN].iloc[::c]
ax.plot(date_x, d1[data.PRICE_TOKEN].iloc[::c], color="red")
ax.set_xlabel("2013~2022 한달 간격")
ax.set_ylabel("KOSPI200F IDX")

ax2 = ax.twinx()
ax2.plot(date_x, d2[data.PRICE_TOKEN].iloc[::c], color="blue")
ax2.set_yticks(ax2.get_yticks()[::7])
ax2.set_ylabel("NASDAQ100F IDX")

plt.xticks([], color='w')
plt.show()
