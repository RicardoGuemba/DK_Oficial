# Previsão ARIMA de demanda por produto

O script agrega o consumo de um produto em intervalos de seis horas e prepara uma série para análise ARIMA a partir de um CSV.

## Dados de entrada

O script lê um CSV com as colunas `Tempo de Indexação`, `Cód. Produto` e `Quantidade`. O caminho do arquivo está definido na variável `file_path`. Ajuste-o para seu arquivo antes da execução. O código solicita o produto e agrupa os registros em intervalos de seis horas.

## Onde começar

- [Dankia_Arima_TOP_v6.py](Dankia_Arima_TOP_v6.py)
