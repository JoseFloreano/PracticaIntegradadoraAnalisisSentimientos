# Análisis de Sentimientos en Textos Turísticos de México

Proyecto de procesamiento de lenguaje natural (NLP) para clasificar reseñas turísticas en sentimientos positivos, neutros y negativos utilizando técnicas de machine learning.

## Descripción

Este proyecto implementa un pipeline completo de análisis de sentimientos sobre un dataset de reseñas turísticas mexicanas. Incluye desde la obtención y preprocesamiento de datos hasta el entrenamiento y evaluación de múltiples modelos de clasificación.

## Dataset

Se utiliza el dataset de Hugging Face:
- **Nombre**: `alexcom/analisis-sentimientos-textos-turisitcos-mx-polaridad`
- **Contenido**: Reseñas de turismo en español con etiquetas de polaridad (1-5)
- **División**: Train y Test

## Estructura del Notebook

### 1. Obtención de Datos
- Carga del dataset desde Hugging Face
- Conversión a DataFrames de pandas

### 2. Análisis Estadístico
- Exploración de la distribución de clases
- Identificación de desbalance (predominio de reseñas positivas)

### 3. Etiquetado
- Transformación de etiquetas numéricas (1-5) a categorías:
  - **Positivo**: Rating 4-5
  - **Neutro**: Rating 3
  - **Negativo**: Rating 1-2

### 4. Preprocesamiento
- Limpieza y normalización de texto
- Eliminación de stopwords (incluye archivo `stopwords_polares.csv`)
- Tokenización con spaCy

### 5. Nubes de Palabras
- Visualización de términos frecuentes por categoría:
  - Dataset completo
  - Clase positiva
  - Clase neutra
  - Clase negativa

### 6. Modelo Espacio Vectorial
- **FastText Embeddings**: Representación semántica de palabras
- **TF-IDF + LSA**: Reducción de dimensionalidad con análisis semántico latente
- Comparación entre ambos enfoques

### 7. Balanceo de Clases
- Aplicación de técnicas de submuestreo (NearMiss) para manejar el desbalance

### 8. Covalores
- Enriquecimiento de features mediante diccionarios de sentimientos

### 9. Entrenamiento
Se implementan tres clasificadores:
- **Random Forest**: Ensamble de árboles de decisión
- **Naive Bayes (Gaussian)**: Clasificador probabilístico
- **SVM (Support Vector Machine)**: Clasificador de márgenes

### 10. Refinamiento / Testing
- Cross-Validation para validación robusta
- GridSearchCV para búsqueda de hiperparámetros óptimos

### 11. Evaluación
- Métricas: Accuracy, Precision, Recall, F1-Score
- Matrices de confusión
- Comparación modelos sin refinar vs refinados

## Requisitos

```
datasets
pandas
numpy
seaborn
matplotlib
spacy
wordcloud
gensim
scikit-learn
imbalanced-learn
tqdm
```

### Modelo de spaCy
```bash
python -m spacy download es_core_news_sm
```

## Uso

1. Clonar el repositorio
2. Instalar dependencias: `pip install -r requirements.txt`
3. Ejecutar el notebook `PI.ipynb`

## Estructura del Proyecto

```
.
├── PI.ipynb                 # Notebook principal
├── README.md                # Este archivo
└── stopwords_polares.csv    # Lista de stopwords personalizadas
```

## Resultados

El modelo SVM refinado muestra el mejor desempeño, logrando:
- Clasificación efectiva de sentimientos positivos y negativos
- Mejor reconocimiento de la categoría neutra comparado con versiones anteriores
- Reducción del "sesgo de optimismo" automático

## Contexto Académico

Práctica Integradora para la materia de **Tratamiento del Lenguaje Natural (TLN)** - ESCOM IPN