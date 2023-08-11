import numpy as np
import pandas as pd

outCsv = str(input("csv?"))
expected = pd.read_csv("submission100.csv")

salida = pd.read_csv(outCsv, delimiter=',')
salida['match'] = np.where(salida['Label'] == expected['Label'], 'True', 'False')

matches = salida['match'].value_counts()

print(matches["True"], " matches")
percent = str((matches["True"] / salida.shape[0]) * 100)
print(percent)
