[![Python](https://img.shields.io/badge/Python-3.8.10-%233776AB?logo=Python)](https://www.python.org/)

# FairUP
The official implmentation of "FairUP: a Framework for Fairness Analysis of Graph Neural Network-Based User Profiling Models"

![fairup_architecture](https://user-images.githubusercontent.com/45569039/220563974-905756a9-eb1f-4140-9a17-73b8c3a52529.png)

The framework currently supports these GNN models:
- [FairGNN](https://arxiv.org/abs/2009.01454)
- [RHGN](https://arxiv.org/abs/2110.07181)
- [CatGCN](https://arxiv.org/abs/2009.05303)
## Abstract
 Modern user profiling approaches capture different forms of interactions with the data, from user-item  to item-item relationships. Hence, Graph Neural Networks (GNNs) have become the natural way to model and process these forms of interactions and build efficient and effective user profiles. However, each GNN-based user profiling approach has its own way to process information, thus creating heterogeneity that does not favour the benchmarking of these approaches. To overcome this issue, we present FairUP, a framework that standardises the input needed to run three state-of-the-art GNN-based user profiling profiles. Moreover, given the importance that algorithmic fairness is getting in the evaluation of Machine Learning tasks, FairUP also includes two modules that (i) allow to assess the presence of unfairness by measuring disparate impact metrics and (ii) mitigate the presence of unfairness via three debiasing techniques that pre-process the data. The framework, while extensible in multiple directions, currently offers the possibility to be run on four real-world dataset.

## Requirements
TBA

## Web application
Available [here](https://mohamedabdelrazek9-fairup-homepage-gv365a.streamlit.app/)

## Later updates
- Integration of new GNN models.
- Integration of new datasets.

## Contact
- M.Sc. Erasmo Purificato: erasmo.purificato@ovgu.com
- M.Sc. Mohamed Abdelrazek: mimo.1998@hotmail.com 
