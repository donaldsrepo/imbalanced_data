# Summarize the result.csv file per model algo
import pandas as pd

df = pd.read_csv('../input/creditcardzip/results.csv', index_col=0)
df['algo'].describe()
df.dtypes

df.groupby('algo')['Avg'].mean()
df.groupby('algo')['Avg'].median()
df.groupby('algo')['Avg'].std()
df.groupby('algo')['Avg'].max()
df.groupby('algo')['Avg'].min()

df_mean = df.groupby('algo')['Avg'].mean()
df_mean
type(df_mean)
df_final = pd.DataFrame(df_mean)#,columns = ['algo','mean'])
df_final
df_sort = df_final.sort_values(by='Avg', ascending=False)

print(df_sort)