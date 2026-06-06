# Lab02-DL-2026-01

Proyecto completo para el Laboratorio 02 de Deep Learning.

El laboratorio estudia deterioro cognitivo con redes neuronales poco profundas
en PyTorch. El problema se trabaja como seis experimentos independientes:

- `GDS`
- `GDS_R1`
- `GDS_R2`
- `GDS_R3`
- `GDS_R4`
- `GDS_R5`

Cada experimento toma una columna objetivo, la transforma a codificacion
one-hot multilabel y entrena una red neuronal con tantas salidas como clases
tenga ese experimento.

## Estructura del repositorio

```text
Lab02-DL-2026-01/
|-- dataset/
|   `-- README.md
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- data_loader.py
|   |-- evaluation.py
|   |-- models.py
|   |-- preprocessing.py
|   `-- uncertainty.py
|-- main.py
|-- environment.yml
|-- .gitignore
`-- README.md
```

## Dataset

```text
dataset/deterioro_cognitivo.sav
```

El dataset esperado contiene 1119 observaciones, 15 atributos binarios de
entrada, una columna `ID` que no se usa como predictor y los seis targets del
laboratorio.

## Preparar ambiente

Con Conda:

```bash
conda env create -f environment.yml
conda activate lab_pytorch
```

## Ejecutar

El laboratorio completo se ejecuta con:

```bash
python main.py
```

Ese comando realiza todo el flujo:

1. Carga `dataset/deterioro_cognitivo.sav`.
2. Revisa la distribucion de clases de cada target.
3. Ejecuta los seis experimentos independientes.
4. Convierte cada target a one-hot multilabel.
5. Entrena una red neuronal poco profunda en PyTorch.
6. Usa `BCEWithLogitsLoss`.
7. Aplica validacion anidada con folds externos e internos.
8. Busca hiperparametros en la validacion interna.
9. Calcula metricas multilabel adicionales a Hamming Loss.
10. Estima incertidumbre con Monte Carlo Dropout.
11. Guarda tablas para el informe.
12. Guarda graficos para el informe.

## Resultados

Los resultados quedan en:

```text
outputs/tables/outer_results.csv
outputs/tables/inner_results.csv
outputs/tables/summary_by_target.csv
outputs/tables/uncertainty_examples.csv
outputs/figures/distribucion_clases.png
outputs/figures/comparacion_metricas.png
outputs/figures/hamming_loss_por_fold.png
outputs/figures/validacion_interna_f1_macro.png
outputs/figures/incertidumbre_mc_dropout.png
```

`summary_by_target.csv` resume la comparacion entre experimentos.
`uncertainty_examples.csv` contiene ejemplos de probabilidad media y desviacion
estandar por etiqueta usando Monte Carlo Dropout.

## Nota metodologica

`GDS` se mantiene como referencia de un caso peor condicionado. En el dataset
real existe una clase con muy pocas muestras, por lo que puede no alcanzar para
validacion estratificada anidada. Cuando eso ocurre, el codigo registra el
problema en `outer_results.csv` en vez de detener la ejecucion completa.
