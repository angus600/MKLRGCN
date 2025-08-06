# MKLRGCN: Predicting miRNA-Disease Associations by Multi-Kernel Learning and Relational Graph Convolutional Neural Network
## Introduction
Biological research has found that miRNA (microRNA) plays an important role in explaining the mechanisms of diseases. Employing advanced algorithms for inferring potential miRNA-disease associations (MDA) is a good approach to the discovery of disease-related miRNAs, since traditional biological experiments are time-consuming and labor-intensive. Currently, traditional graph convolutional neural Network (GCN) has been widely applied but cannot effectively distinguish various biological relationships among the same or different types of nodes, limiting the ability of identification. Therefore, we propose a novel end-to-end model called MKLRGCN by multi-kernel learning and relational graph convolutional neural network (RGCN) to predict MDAs more effectively. First, a multi-kernel learning module is constructed to learn combined kernels of miRNAs/diseases and their initial feature representations. Then, multiple types of relations are derived by identifying high co-occurrences between miRNAs/diseases, and hub-ordinary associations between miRNAs and diseases, which feed into RGCN to learn embeddings of miRNAs/diseases from a heterogeneous network of MDAs. Finally, the above embeddings after integrating with those from homogeneous network of miRNAs/diseases are used to infer potential MDAs by a multi-layer perception. Experimental results demonstrate the effectiveness of distinct modules, and the good performance of our model compared to state-of-the-art models in predicting MDAs. This work provides a useful approach for computationally identifying potential MDAs, which would be helpful for enhancing the research of complex diseases. The code is available at .
## Requirements
  * Python 3.9 or higher
  * PyTorch 2.0.0 
  * torch-geometric 2.3.0
  * numpy 1.26.4
  * scikit-learn 1.1.3
## Data
* ./HMDD3/d_fs.csv: The disease functional kernel
* ./HMDD3/d_gs.csv: The disease Gaussian kernel
* ./HMDD3/d_ss.csv: The disease semantic kernel
* ./HMDD3/m_fs.csv: The miRNA functinal kernel
* ./HMDD3/m_gs.csv: The miRNA Gaussian kernel
* ./HMDD3/m_ss.csv: The miRNA sequence kernel
## Usage 
### 1. Data Preprocessing 
For detailed implementation instructions, please refer to getData.py 
### 2. Get miRNA and disease embedding 
First, a multi-kernel learning module is constructed to learn combined kernels of miRNAs/diseases and their initial feature representations. Then, multiple types of relations are derived by identifying high co-occurrences between miRNAs/diseases, and hub-ordinary associations between miRNAs and diseases, which feed into RGCN to learn embeddings of miRNAs/diseases from a heterogeneous network of MDAs. For detailed implementation instructions, please refer to model.py 
### 3. Predicting and Model Output 
Infer potential MDAs by a multi-layer perceptron. We use AUROC and AUPRC as two standard evaluation metrics. For detailed implementation instructions, please refer to model.py and train.py
## Contact
Please contact us for any further questions:
* Ju Xiang xiang.ju@foxmail.com
## Quick Run
Execute python main.py to run the code

