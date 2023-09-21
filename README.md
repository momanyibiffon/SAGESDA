<h2>Model Overview</h2>
The SAGESDA is a snoRNA-disease association prediction model based on the GraphSAGE Graph Neural Network (GNN) architecture.
Designed for the prediction of the associations between small nucleolar RNAs (snoRNAs) and diseases, SAGESDA leverages the similarity information between snoRNAs and diseases, 
represented as matrices, along with association data to create an adjacency matrix used to generate the heterogeneous graph. 
SAGESDA learns to predict potential associations between snoRNAs and diseases by utilizing the new feature embeddings obtained 
through the GraphSAGE architecture which utilizes the local neighbourhood of a given node to obtain node embeddings. After training and evaluation using the training and evaluation data sets, 
the model is tested on the unseen test set to establish the model's performance. The link-prediction task of predicting potential snoRNA-disease associations utilizes 
the dot-product classifier to determine the possibility of a connection between two adjacent snoRNA-disease nodes based on the dot products.

<h2>Datasets</h2>
The SAGESDA model relies on three main data sets obtained from Zijie et al. as explained in the manuscript;, these are:
<ol>
   <li>SnoRNA Similarity Matrix which represents the similarity between snoRNAs.</li>
   <li>Disease Similarity Matrix which represents the similarity between diseases.</li>
   <li>Association Data which contains association information based on the known snoRNA-disease associations.</li>
</ol>
 
Note that the association data was used to generate an adjacency matrix in which values 0 and 1 represent negative and positive associations, respectively.
The heterogeneous graph was then generated using the adjacency matrix information.

<h2>Model Architecture</h2>
The SAGESDA model was implemented based on the GraphSAGE GNN architecture, specifically designed for graph-based semi-supervised learning tasks. 
It involves the following stages of development:
<ol>
   <li>Graph Construction:</li>
   <p>The snoRNA and disease similarity matrices alongside the adjacency matrix information (association data) were utilized to construct a heterogeneous graph,
   whereby each snoRNA and disease are represented as a node in the graph with the links representing the association information.</p>
   <li>GraphSAGE Layers:</li>
   <p>SAGESDA consisted of multiple GraphSAGE layers which facilitated feature aggregation utilizing a node's local neighborhood to generate its embedding.
   These layers were stacked for the purpose of capturing complex patterns in the graph.
   Furthermore, the model utilized three heterogeneous networks trained on the same parameters to obtain rich node embeddings.</p>
   <li>The output Layer:</li>
   <p>The final layer of the model is a binary classification layer that predicts the probability of association between snoRNAs and diseases. 
The model uses a dot-product classifier to determine potential associations based on the dot products after which popular evaluation metrics such as AUC are used to establish model performance.</p>
</ol>

<h2>Model usage</h2>
The usage of the SAGESDA model involves data preparation, processing, model training and evaluation, model testing and finally prediction of snoRNA-disease associations.

<h2>Model dependencies</h2>
The following Python packages are essential for the usage of the SAGESDA model:

<ul>
   <li>PyTorch and torch_geometric</li>
   <li>NetworkX for graph construction</li>
   <li>NumPy and pandas for data manipulation</li>
   <li>Scikit-learn for model evaluation</li>
   <li>Matplotlib and Seaborn for plot generation</li>
</ul>

<h2>Acknowledgments</h2>
We would like to acknowledge the contributions of GraphSAGE and the broader GNN research community for their valuable work in developing GNN architectures which have been widely adopted.

<h2>Citation</h2>
If you use the SAGESDA model in your research, please cite our work alongside other relevant materials and documentation associated with the GraphSAGE architecture, 
especially the main paper by Hamilton et al. titled "Inductive Representation Learning on Large Graphs".
