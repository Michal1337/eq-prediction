# eq_prediction
Repository for earthquake prediction project.

## PIPELINE:
1. utils/get_data.ipynb - downloads data (~2h, many requests)
2. utils/add_features.ipynb - filters data and adds features and labels (~30min)
3. utils/make_npys.ipynb - makes .npy files for each region (~30min)
4. utils/make_datasets.ipynb - transforms .npy files into tf.Dataset (~)
5. utils/merge_datasets.ipynb - merges datasets into one (TODO)
6. notebooks/train.ipynb - trains the model (TODO)


## TODO :
1. STRONA
2. Lepsze EDA
3. Wymyślenie nowych featerów i napisanie funkcji do obliczania ich - liczba dni od ostatniego trzęsienia?, jakieś stosunki czegoś do czegoś?

## Majkel TODO:
1. Te same modele dla drugiego zbioru danych
2. Ogarnąć threshold, timesplit, geosplit, minimalny magnitude na label, potrzebna opinia eksperta XD
3. 2 OMEGA MODELE - embeddinngi trzęsień z conv na spectogramie (i inne feature z łatwiejszych modeli). 2 wersje - dodwanie embeddingow na tym sam
timestepie, embeddingi przerw

## Google docs:
https://docs.google.com/document/d/1-TtjlulbLiVpVFpqWxrW3lChgbOfQPjZgN9pGFdF2fc/edit?fbclid=IwAR3njF9Zsnh2QCJTB4ldNGQlvKxJObMcXMOpnOj6f6Su9E7dgHNcwPWgdfs
