from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf

# input: dataframe
df = caiso_dataset
df_array = df['iso']
plot_acf(df_array)
pyplot.show()
