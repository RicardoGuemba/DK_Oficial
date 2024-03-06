import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from itertools import product
from statsmodels.tsa.stattools import adfuller

# Funções de métricas de erro
def MAPE(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def RMSE(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def MAE(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Carregar dados
file_path = "/Users/ricardohausguembarovski/Desktop/rah.csv"
df = pd.read_csv(file_path)

# Pré-processamento dos dados
df['Tempo de Indexação'] = pd.to_datetime(df['Tempo de Indexação'])
df['Tempo de Indexação'] = df['Tempo de Indexação'].dt.floor('6H')  # Arredondamento para o intervalo de 6 horas
df.set_index('Tempo de Indexação', inplace=True)
codigo_produto = int(input("Digite o código do produto: "))
produto_df = df[df['Cód. Produto'] == codigo_produto]

# Agrupamento e soma das quantidades nos intervalos de 6 horas
demand_series = produto_df.resample('6H').sum()['Quantidade'].fillna(0)

# Suprimir avisos durante o processamento
import warnings
warnings.filterwarnings("ignore")

# Verificar estacionariedade da série temporal
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('Estatísticas ADF:', result[0])
    print('Valor-p:', result[1])
    print('Valores críticos:')
    for key, value in result[4].items():
        print('\t', key, ':', value)
    if result[1] <= 0.05:
        print("A série temporal é estacionária.")
    else:
        print("A série temporal não é estacionária.")

check_stationarity(demand_series)

# Busca dos melhores parâmetros do ARIMA
p_range = range(6)
d_range = range(0,1)
q_range = range(4)
param_grid = list(product(p_range, d_range, q_range))
aic_values = []
best_order = None

for param in param_grid:
    order = (param[0], param[1], param[2])
    try:
        model = ARIMA(demand_series, order=order).fit()
        aic_values.append(model.aic)
    except:
        continue

best_aic = min(aic_values)
best_order = param_grid[aic_values.index(best_aic)]

# Ajuste do modelo ARIMA com os melhores parâmetros
model = ARIMA(demand_series, order=best_order).fit()

# Previsões para os próximos 16 períodos de 6 horas
forecast_steps = 4
forecast = np.round(model.forecast(steps=forecast_steps)).astype(int)  # Arredonda os valores previstos para inteiros
forecast = np.maximum(forecast, 0)  # Garante que os valores previstos sejam maiores ou iguais a 0

# Plotagem da série temporal e previsão
plt.figure(figsize=(10, 6))
plt.plot(demand_series, label='Série Temporal', color='blue')
plt.plot(demand_series.index[-1] + pd.to_timedelta(np.arange(1, forecast_steps + 1) * 6, unit='H'), forecast, color='red', linestyle='--', label='Previsão')
plt.xlabel('Dia e Hora')
plt.ylabel('Demanda')
plt.title(f'Série Temporal e Previsão de Demanda para o Produto {codigo_produto}')
plt.legend()
plt.show()

# Apresentação dos últimos valores
print("Últimos valores da série temporal:")
print(demand_series.tail())

# Apresentação dos valores previstos de demanda
print("\nValores previstos de demanda e respectivo dia e hora:")
for i, value in enumerate(forecast):
    forecast_time = demand_series.index[-1] + pd.to_timedelta((i + 1) * 6, unit='H')
    print(f"Dia {forecast_time.date()} Hora {forecast_time.hour}: {int(value)}")

# Avaliação do modelo
mape = MAPE(demand_series.values[-forecast_steps:], forecast)
rmse = RMSE(demand_series.values[-forecast_steps:], forecast)
mae = MAE(demand_series.values[-forecast_steps:], forecast)

# Impressão das métricas de erro
print(f"\nMétricas de Erro:")
print(f"MAPE: {mape}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
