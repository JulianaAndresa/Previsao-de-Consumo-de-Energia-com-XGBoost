# Previsão de Consumo de Energia com XGBoost

Repositório criado para disponibilizar os arquivos referente á previsão de consumo de energia com Python usando um modelo de machine learning XGBoost.

## 1. Importação de Bibliotecas

```shell
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from sklearn.metrics import mean_squared_error
color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')
```

## 2. Leitura e Preparação dos Dados

```
df = pd.read_csv('../input/hourly-energy-consumption/PJME_hourly.csv')
df = df.set_index('Datetime')
df.index = pd.to_datetime(df.index)
```


## 3. Visualização Inicial dos Dados

```
df.plot(style='.', figsize=(15, 5), color=color_pal[0], title='Uso de energia PJME em MW')
plt.show()
```

## 4. Divisão dos Dados em Conjunto de Treinamento e Teste

```
train = df.loc[df.index < '01-01-2015']
test = df.loc[df.index >= '01-01-2015']
```

## 5. Visualização da Divisão Treino/Teste

```
fig, ax = plt.subplots(figsize=(15, 5))
train.plot(ax=ax, label='Conjunto de treinamento', title='Divisão de dados de treinamento/teste')
test.plot(ax=ax, label='Conjunto de teste')
ax.axvline('01-01-2015', color='black', ls='--')
ax.legend(['Conjunto de treinamento', 'Conjunto de teste'])
plt.show()
```

## 6. Visualização de uma Semana de Dados

```
df.loc[(df.index > '01-01-2010') & (df.index < '01-08-2010')].plot(figsize=(15, 5), title='Semana de Dados')
plt.show()
```

## 7. Criação de Recursos

```
def create_features(df):
    """
    Crie recursos de séries temporais com base no índice de séries temporais.
    """
    df = df.copy()
    df['hora'] = df.index.hour
    df['diadasemana'] = df.index.dayofweek
    df['trimestre'] = df.index.quarter
    df['mês'] = df.index.month
    df['ano'] = df.index.year
    df['diadoano'] = df.index.dayofyear
    df['diadomes'] = df.index.day
    df['semanadoano'] = df.index.isocalendar().week
    return df

df = create_features(df)
```

## 8. Visualização de Relações entre Características e Alvo

```
fig, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(data=df, x='hora', y='PJME_MW')
ax.set_title('MW por hora')
plt.show()

fig, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(data=df, x='mês', y='PJME_MW', palette='Blues')
ax.set_title('MW por mês')
plt.show()
```

## 9. Preparação dos Conjuntos de Dados para Modelagem

```
train = create_features(train)
test = create_features(test)

FEATURES = ['diadoano', 'hora', 'diadasemana', 'trimestre', 'mês', 'ano']
TARGET = 'PJME_MW'

X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]
```

## 10. Treinamento do Modelo XGBoost

```
reg = xgb.XGBRegressor(
    base_score=0.5, 
    booster='gbtree',  # Utiliza o booster baseado em árvores de decisão
    n_estimators=2000,  # Número de árvores a serem construídas
    early_stopping_rounds=50,
    objective='reg:squarederror',  # Objetivo de regressão com erro quadrático
    max_depth=5,  # Profundidade máxima das árvores
    learning_rate=0.005,  # Taxa de aprendizado
    min_child_weight=3,  # Peso mínimo de uma folha
    reg_alpha=0.1,  # Regularização L1
    reg_lambda=0.1  # Regularização L2
)

reg.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=100
)
```

## 11. Importância do Recurso

```
fi = pd.DataFrame(data=reg.feature_importances_,
             index=reg.feature_names_in_,
             columns=['importância'])
fi.sort_values('importância').plot(kind='barh', title='Importância do recurso')
plt.show()
```

## 12. Previsão no Conjunto de Teste

```
test['predição'] = reg.predict(X_test)
df = df.merge(test[['predição']], how='left', left_index=True, right_index=True)
ax = df[['PJME_MW']].plot(figsize=(15, 5))
df['predição'].plot(ax=ax, style='.')
plt.legend(['Dados Verdade', 'Previsões'])
ax.set_title('Dados brutos e previsão')
plt.show()
```

## 13. Visualização de Previsões em uma Semana Específica

```
ax = df.loc[(df.index > '04-01-2018') & (df.index < '04-08-2018')]['PJME_MW'].plot(figsize=(15, 5), title='Semana de Dados')
df.loc[(df.index > '04-01-2018') & (df.index < '04-08-2018')]['predição'].plot(style='.')
plt.legend(['Dados Verdade', 'predição'])
plt.show()
```

## 14. Pontuação (RMSE)

```
score = np.sqrt(mean_squared_error(test['PJME_MW'], test['predição']))
print(f'Pontuação RMSE no conjunto de teste: {score:0.2f}')
```

## 15. Análise do Erro

```
test['erro'] = np.abs(test[TARGET] - test['predição'])
test['data'] = test.index.date
test.groupby(['data'])['erro'].mean().sort_values(ascending=False).head(10)
```
