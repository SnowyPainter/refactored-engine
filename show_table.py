import sys
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')
fig, ax = plt.subplots()
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

df = pd.read_csv(sys.argv[1])
df = df.iloc[0:6]
ax.table(cellText=df.values, colLabels=df.columns, loc='center')

fig.tight_layout()
plt.show()