# Preparando-um-dataset-para-modelagem-de-dados-
# Linhas de código em Python para a preparação de dados

import pandas as pd
import matplotlib.pyplot as plt

# Carregar o dataset
def load_data(file_path):
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    print("Resumo estatístico dos dados:")
    print(df.describe())
    print("\nTipos de dados:")
    print(df.dtypes)
    return df

# Limpeza dos dados
def clean_data(df):
    # Remover valores nulos em CustomerID
    df = df.dropna(subset=['CustomerID'])

    # Converter CustomerID para inteiro
    df['CustomerID'] = df['CustomerID'].astype(int)

    # Remover valores de Quantity e UnitPrice menores ou iguais a zero
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

    # Converter InvoiceDate para datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # Remover duplicatas
    df = df.drop_duplicates()

    return df

# Remover outliers extremos
def remove_outliers(df):
    df = df[(df['Quantity'] <= 10000) & (df['UnitPrice'] <= 5000)]
    return df

# Criar coluna de valor total da compra
def add_total_price(df):
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    return df

# Criar gráficos
def plot_graphs(df):
    # Top 10 países com maior valor em vendas
    top_countries = df.groupby('Country')['TotalPrice'].sum().nlargest(10)
    top_countries.plot(kind='bar', title='Top 10 Países com Maior Valor em Vendas')
    plt.show()

    # Top 10 produtos mais vendidos
    top_products = df.groupby('Description')['Quantity'].sum().nlargest(10)
    top_products.plot(kind='bar', title='Top 10 Produtos Mais Vendidos')
    plt.show()

    # Valor de venda total por mês
    df['Month'] = df['InvoiceDate'].dt.to_period('M')
    sales_per_month = df.groupby('Month')['TotalPrice'].sum()
    sales_per_month.plot(kind='line', title='Valor de Venda Total por Mês')
    plt.show()

    # Valor de venda total por mês e por país (top 10 países)
    top_country_sales = df[df['Country'].isin(top_countries.index)].groupby(['Month', 'Country'])['TotalPrice'].sum().unstack()
    top_country_sales.plot(kind='line', title='Valor de Venda Total por Mês e por País')
    plt.show()

# Calcular métricas RFM
def calculate_rfm(df):
    # Definir a última data disponível no dataset
    last_date = df['InvoiceDate'].max()

    # Agrupar por cliente e pedido para obter o valor total gasto por compra
    df_rfm = df.groupby(['CustomerID', 'InvoiceNo']).agg(
        LastPurchase=('InvoiceDate', 'max'),
        TotalSpent=('TotalPrice', 'sum')
    ).reset_index()

    # Calcular as métricas RFM
    rfm = df_rfm.groupby('CustomerID').agg(
        R=('LastPurchase', lambda x: (last_date - x.max()).days),
        F=('InvoiceNo', 'count'),
        M=('TotalSpent', 'mean')
    ).reset_index()

    return rfm

# Salvar resultado
def save_rfm(rfm, output_file):
    rfm.to_csv(output_file, index=False)

# Pipeline completo
def main(input_file, output_file):
    df = load_data(input_file)
    df = clean_data(df)
    df = remove_outliers(df)
    df = add_total_price(df)
    plot_graphs(df)
    rfm = calculate_rfm(df)
    save_rfm(rfm, output_file)
    print("Arquivo RFM gerado e salvo com sucesso!")

# Exemplo de uso:
# main('ecommerce_data.csv', 'rfm_output.csv')
