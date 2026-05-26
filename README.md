# Identificação de Tweets sobre Desastres Reais — Modelo de Classificação com Processamento de Linguagem Natural

Solução para a competição [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started) do Kaggle.

---

## Resultado

| Métrica | Valor |
|---|---|
| F1-Score (Kaggle) | **0.80324** |
| Algoritmo | Word2Vec + Feature Engineering + Ensemble (Soft Voting) |
| Dataset | 7.613 tweets de treino |

---

## Estrutura

```
disaster-tweets-kaggle/
├── identificacao_tweets_desastres.ipynb
├── README.md
└── submission.csv
```

---

## Etapas do Projeto

### 1. Análise Exploratória (EDA)

- Distribuição das classes — leve desbalanceamento (43% desastre, 57% não-desastre)
- Análise do comprimento dos tweets por classe — tweets de desastre tendem a ser mais longos
- Top 10 keywords por classe — evidencia que a keyword é a feature mais preditiva do dataset
- Verificação de valores ausentes em `keyword` e `location`

### 2. Pré-processamento

Etapas aplicadas ao texto de cada tweet:

| Etapa | Descrição | Justificativa |
|---|---|---|
| URLs | Remoção de `http://...` | Não carregam significado semântico |
| Menções | Remoção de `@usuario` | Ruído sem valor preditivo |
| Hashtags | Mantém o texto, remove `#` | O conteúdo da hashtag é relevante |
| Lowercasing | Converte para minúsculas | Unifica tokens (`Fire` = `fire`) |
| Pontuação | Remoção de caracteres especiais | Reduz vocabulário sem perda semântica |
| Stopwords | Remoção de palavras funcionais | Foco em palavras de conteúdo |
| Lematização | Reduz palavras à forma base | `burning → burn`, `fires → fire` |

### 3. Feature Engineering

Além dos embeddings, metadados do tweet são extraídos e concatenados ao vetor final (207 dimensões).

| Feature | Descrição | Hipótese |
|---|---|---|
| `keyword_encoded` | Keyword do tweet (label encoded) | Feature mais preditiva — indica diretamente o tema |
| `text_len` | Número de caracteres | Tweets de desastre tendem a ser mais longos |
| `word_count` | Número de palavras | Mais palavras = mais contexto informativo |
| `hashtag_count` | Número de hashtags | Desastres geram mais hashtags de alerta |
| `url_count` | Número de URLs | Notícias reais costumam ter links |
| `upper_count` | Palavras em maiúsculas | Urgência e alerta são escritos em caps |
| `exclamation_count` | Pontos de exclamação | Indicador de emoção e urgência |

### 4. Word2Vec — Embeddings

Word2Vec aprende representações densas de palavras a partir do contexto em que aparecem.  
Cada tweet é representado pela **média dos vetores** de seus tokens.

| Parâmetro | Valor | Justificativa |
|---|---|---|
| `vector_size` | 200 | Maior capacidade semântica |
| `window` | 5 | Captura contexto local dos tweets |
| `min_count` | 1 | Mantém tokens raros — corpus pequeno, todo token importa |
| `epochs` | 20 | Mais passagens pelo corpus = embeddings mais refinados |
| `sg` | 1 | Skip-gram — melhor para vocabulário pequeno e palavras raras |

> O modelo Word2Vec é treinado em todo o corpus (treino + teste) sem usar os labels, portanto sem data leakage.

### 5. Modelos Avaliados

Três classificadores avaliados via cross-validation (5 folds estratificados).  
Métrica: **F1-Score** — mais adequado que acurácia para classes desbalanceadas.

| Modelo | Vantagem |
|---|---|
| Logistic Regression | Rápido, interpretável, bom baseline linear |
| Random Forest | Captura não-linearidades, robusto a outliers |
| SVM | Eficaz em espaços de alta dimensão como embeddings |

### 6. Ensemble — Soft Voting

`VotingClassifier` combina os três modelos pela média das probabilidades (soft voting).  
O `StandardScaler` está dentro de cada `Pipeline` — normalização feita por fold, sem data leakage.

### 7. Avaliação

- F1-Score avaliado com StratifiedKFold (5 folds) — preserva proporção das classes em cada fold
- Matriz de confusão e relatório de classificação do ensemble
- Modelo retreinado no conjunto completo antes de gerar o submission

---

## Conclusão

O modelo ensemble com Soft Voting atingiu **F1-Score de 0,803** no Kaggle, demonstrando que a combinação de embeddings semânticos com features estruturais do tweet é uma abordagem eficaz para classificação de texto em domínios específicos.

Os principais fatores que contribuíram para o resultado:

- **Word2Vec (Skip-gram, 200d):** embeddings treinados no corpus completo capturam relações semânticas específicas do domínio de desastres
- **Feature Engineering:** metadados como `keyword_encoded`, comprimento do tweet e contagem de maiúsculas adicionam informação estrutural que os embeddings não capturam
- **Soft Voting:** a combinação dos três modelos foi mais robusta do que qualquer modelo individual
- **Pipeline com StandardScaler:** a normalização dentro de cada fold do cross-validation garantiu avaliação sem data leakage

---

## Tecnologias

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-4C9BE8?style=flat&logoColor=white)
![Gensim](https://img.shields.io/badge/Gensim-4CA86B?style=flat&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)

---

*Juliana Burato — 2026*

*Juliana Burato — 2026*
