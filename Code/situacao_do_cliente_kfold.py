import pandas as pd
from collections import Counter
from sklearn.model_selection import cross_val_score
import numpy as np


df = pd.read_csv('../Data/situacao_do_cliente.csv')
X_df = df[['recencia','frequencia', 'semanas_de_inscricao']]
Y_df = df['situacao']

Xdummies_df = pd.get_dummies(X_df)
Ydummies_df = Y_df

X = Xdummies_df.values
Y = Ydummies_df.values

porcentagem_de_treino = 0.8

tamanho_de_treino = int(porcentagem_de_treino * len(Y))

treino_dados = X[:tamanho_de_treino]
treino_marcacoes = Y[:tamanho_de_treino]

validacao_dados = X[tamanho_de_treino:]
validacao_marcacoes = Y[tamanho_de_treino:]


def fit_and_predict(nome, modelo, treino_dado, treino_marcacoes):
    k = 10
    scores = cross_val_score(modelo,treino_dado,treino_marcacoes, cv=k)
    taxa_de_acerto = np.mean(scores) 

    msg = f"taxa de acerto de {nome}: {taxa_de_acerto}"
    print(msg)
    return taxa_de_acerto


resultados = {}

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
#Setando o modelo one vcs rest que por tras utiliza Linear SVC
modelo_One_Vs_Rest = OneVsRestClassifier(LinearSVC(random_state=0))
resultado_One_Vs_Rest = fit_and_predict("OneVsRest", modelo_One_Vs_Rest, treino_dados, treino_marcacoes)
resultados[resultado_One_Vs_Rest] = modelo_One_Vs_Rest

from sklearn.multiclass import OneVsOneClassifier
modelo_One_Vs_One = OneVsOneClassifier(LinearSVC(random_state=0))
resultado_One_Vs_One = fit_and_predict("OneVsOne", modelo_One_Vs_One, treino_dados, treino_marcacoes)
resultados[resultado_One_Vs_One] = modelo_One_Vs_One

from sklearn.naive_bayes import MultinomialNB
modeloMultinomial = MultinomialNB()
resultadoMultinomial = fit_and_predict("MultinomialNB", modeloMultinomial, treino_dados, treino_marcacoes)
resultados[resultadoMultinomial] = modeloMultinomial

from sklearn.ensemble import AdaBoostClassifier
modeloAdaBoost = AdaBoostClassifier()
resultadoAdaBoost = fit_and_predict("AdaBoostClassifier", modeloAdaBoost, treino_dados, treino_marcacoes)
resultados[resultadoAdaBoost] = modeloAdaBoost

print(resultados)
maximo = max(resultados)
vencedor = resultados[maximo]
print(f"Vencedor: {vencedor}")

vencedor.fit(treino_dados, treino_marcacoes)
resultado = vencedor.predict(validacao_dados)
acertos = (resultado == validacao_marcacoes)

total_acertos = sum(acertos)
taxa_de_acerto = 100.0 * total_acertos / len(validacao_marcacoes)
total_de_elementos = len(validacao_dados)
print("Total de teste: %d" % total_de_elementos)
print("Taxa de acerto : %f" % taxa_de_acerto)



"""def fit_and_predict(nome, modelo, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes):
    modelo.fit(treino_dados, treino_marcacoes)

    resultado = modelo.predict(teste_dados)

    acertos = resultado == teste_marcacoes

    total_de_acertos = sum(acertos)
    total_de_elementos = len(teste_dados)

    taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

    msg = "Taxa de acerto do algoritmo {0}: {1}".format(nome, taxa_de_acerto)

    print(msg)
    return taxa_de_acerto

def teste_real(modelo, validacao_dados, validacao_marcacoes):
    resultado = modelo.predict(validacao_dados)
    acertos = resultado == validacao_marcacoes

    total_de_acertos = sum(acertos)
    total_de_elementos = len(validacao_marcacoes)

    taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

    msg = "Taxa de acerto do vencedor entre os dois algoritmos no mundo real: {0}".format(taxa_de_acerto)
    print(msg)

resultados = {}

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
#Setando o modelo one vcs rest que por tras utiliza Linear SVC
modelo_One_Vs_Rest = OneVsRestClassifier(LinearSVC(random_state=0))
resultado_One_Vs_Rest = fit_and_predict("OneVsRest", modelo_One_Vs_Rest, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)
resultados[resultado_One_Vs_Rest] = modelo_One_Vs_Rest

from sklearn.multiclass import OneVsOneClassifier
modelo_One_Vs_One = OneVsOneClassifier(LinearSVC(random_state=0))
resultado_One_Vs_One = fit_and_predict("OneVsOne", modelo_One_Vs_One, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)
resultados[resultado_One_Vs_One] = modelo_One_Vs_One

from sklearn.naive_bayes import MultinomialNB
modeloMultinomial = MultinomialNB()
resultadoMultinomial = fit_and_predict("MultinomialNB", modeloMultinomial, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)
resultados[resultadoMultinomial] = modeloMultinomial

from sklearn.ensemble import AdaBoostClassifier
modeloAdaBoost = AdaBoostClassifier()
resultadoAdaBoost = fit_and_predict("AdaBoostClassifier", modeloAdaBoost, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)
resultados[resultadoAdaBoost] = modeloAdaBoost

print(resultados)
maximo = max(resultados)
vencedor = resultados[maximo]
print(f"Vencedor: {vencedor}")

teste_real(vencedor, validacao_dados, validacao_marcacoes)

acerto_base = max(Counter(validacao_marcacoes).values())
taxa_de_acerto_base = 100.0 * acerto_base / len(validacao_marcacoes)
print("Taxa de acerto base: %f" % taxa_de_acerto_base)

total_de_elementos = len(validacao_dados)
print("Total de teste: %d" % total_de_elementos)"""