# 2AMM20
This repository consists of the codes for the project created during the course 2AMM20 for NLP Group 01.


## Structure 
**Scripts** folder contain all Python scripts ran to produce the code.

  1) BERT_Polysemy-Biomedical.py is the final code of our approach to obtain contextual cluster files.
  
  2) BERT_similarity.ipynb contains functions to compute additional results. Produces BERT embeddings which can be used to calculate cosine similarity sentences and variance. 

  3) FastText.ipynb contains the code executed to collect results for fastText embeddings.


**Data** folder contains all data used to perform the experiment.

  1) MSHCorpus.zip consists of the NLM-WSD Dataset. This Data Needs to be converted into a single text file.

  2) NLM_FULL_DATA.zip consists of a single txt file ( NLM_FULL_DATA.txt) created using the above-mentioned MSHCorpus.

## Execution

### BERT_Polysemy-Biomedical.py
After running the .py file the user will be presented to input some arguments. We explain these arguments below:-
1) **"Enter the number of target terms that you wish to disambiguate"** :- The user shall enter the integer terms planned to disambiguate.
2) **"Enter the target term"** :- The user shall enter the target term (string input).
3) **"Enter input for a starting label for which the threshold plot is to be obtained "** :- The user shall input an arbitary label (int) for the target term t for which they wish to see a [Threshold Plot](https://github.com/a-moharil/2AMM20/blob/main/data/cold_1_cluster1__label_37_scatter.png). The plots will be created from "starting label" to "ending label" (which will be the consecutive argument). 
4) **It is recommended for the user to kindly change the directory input in the line 152 to run the code also to mention their respective corpus in line 148.**


### BERT_similarity.ipynb
The Jupyter notebook requires paths to be modified depending on the function being executed. These paths include location of the data as a CSV, location to save etc.

### FastText.ipynb
The Jupyter notebook requires the file path to the CSV to be adjusted.

## Results
All obtained results of the experimentations are uploaded to a [Google Drive Folder](https://drive.google.com/drive/u/2/folders/1I30CCJ1ZzuXLs5020nWh7DD-YbGFOhcf) which can be accessed publicly through the link.
