#Para ler caracteres especiais
#!-*- coding: utf8 -*-
from nltk import stem
import pandas as pd
from sklearn.model_selection import cross_val_score
import numpy as np
import nltk

#setando classificações
classificacoes = pd.read_csv('../Data/emails.csv', encoding='utf-8')
#setando os textos como vieram
textosPuros = classificacoes['email']
#Transformando em minusculo
frases = textosPuros.str.lower()
#transformando os emails em palavras
#nltk.download("punkt")
textosQuebrados = [nltk.tokenize.word_tokenize(frase) for frase in frases]

#nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words("portuguese")
#Extraindo apenas as raizes das palavras
#nltk.download('rslp')
stemmer = nltk.stem.RSLPStemmer()

#Fazendo um dicionário de palavras como conjuntos 
#Pois sets não permitem repetições de elementos
dicionario = set()

#Separando por emails
"""
    lista -> emails presentes em textos quebrados
    textosQuebrados -> CSV já tratado
"""
for lista in textosQuebrados:
    #Adicionando emails no set dicionário
    #Apenas as palavras com tamanho maior que 2 letras, palavra não raizes, as originais
    validas = [stemmer.stem(palavra) for palavra in lista if palavra not in stopwords and len(palavra) > 2]
    dicionario.update(validas)

#Cria um lista de tuplas com palavras presentes np set dicionario
#Com indices indo de 0 até o final de dicionario
totalDePalavras = len(dicionario)
tuplas = list(zip(dicionario, range(totalDePalavras)))


#Um dicionário com palavras:indices
tradutor = {palavra:indice for palavra, indice in tuplas}
#Criando um tradutor de textos para números
#Se a palavra está no dicionário "tradutor"...
#Adiciona na posição dela +1


def vetorizar_texto(texto, tradutor):
    posicao = 0
    vetor = [0] * len(tradutor)
    for palavra in texto:
        if len(palavra) > 0:
            raiz = stemmer.stem(palavra)
            if raiz in tradutor:
                posicao = tradutor[raiz]
                vetor[posicao] += 1
    return vetor

#Vetorizando os textos em cada número
vetores_de_texto = [vetorizar_texto(texto, tradutor) for texto in textosQuebrados]
marcas = classificacoes['classificacao']

#X,Y em arrays para poderem sofrer shape, ou seja, fateamento
#Definindo X e Y após o tratamento de textos:
#X são os textos tratados
X = np.array(vetores_de_texto)
#Y são as classificações dos emails
Y =  np.array(marcas.tolist())

porcentagem_de_treino = 0.8

tamanho_de_treino = int(porcentagem_de_treino * len(Y))
tamanho_de_validacao = len(Y) - tamanho_de_treino

treino_dados = X[0:tamanho_de_treino]
treino_marcacoes = Y[0:tamanho_de_treino]

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
print("Taxa de acerto no mundo real: %f" % taxa_de_acerto)