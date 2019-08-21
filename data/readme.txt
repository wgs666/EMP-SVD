Data set used in paper "Prediction of drug-disease associations based on ensemble meta paths and singular value decomposition".

If you use this data set, please kindly cite this paper:
@article{wu2019EMP-SVD,
  title={Prediction of drug-disease associations based on ensemble meta paths and singular value decomposition},
  author={Wu, Guangsheng and Liu, Juan and Yue, Xiang},
  journal={BMC bioinformatics},
  volume={20},
  number={3},
  pages={134},
  year={2019},
  publisher={BioMed Central}
}


drugList.txt: The first column is DrugBank ID of drugs, the second column is index, there are 1186 drugs in all, index:0-1185

diseaseList.txt: The first column is OMIM ID of diseases, the second column is index, there are 449 diseases in all, index:0-448

proteinList.txt: The first column is Uniprot ID of proteins, the second column is index, there are 1467 proteins in all, index:0-1466

drugSimilarity.txt: Similarity matrix of 1186 drugs, drugs are ordered by index

diseaseSimilarity.txt: Similarity matrix of 449 diseases, diseases are ordered by index

proteinSimilarity.txt: Similarity matrix of 1467 proteins, proteins are ordered by index

drugDiseaseInteraction.txt: The drug-disease interaction matrix, there are 1827 non-zero associations.  Drugs and diseases are ordered by index

drugProteinInteraction.txt: The drug-protein interaction matrix, there are 4642 non-zero associations.  Drugs and proteins are ordered by index

diseaseProteinInteraction.txt: The disease-protein interaction matrix, there are 1365 non-zero associations.  Diseases and proteins are ordered by index
