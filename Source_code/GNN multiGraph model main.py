#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


# datasets import
# ss: snorna similarity
# ds: disease similarity
# AM: adjacency matrix
ss = pd.read_csv('dataset/snoRNA_sim/snoRNA_4mer_similarity.csv', index_col=0)
ds = pd.read_csv('dataset/disease_sim_graph_filtered.csv', index_col=0)
AM = pd.read_csv('dataset/relationship_matrix_filtered.csv', index_col=0)


# In[3]:


ss


# In[4]:


ds


# In[5]:


AM


# In[6]:


# generating association information, this can aso be used as edge_index
association = []
for i in AM.index:
    for j in AM.columns:
        if AM.loc[i][j] == 1:
            association.append([i, int(j)])
# saving association data into a dataframe
association_df = pd.DataFrame(association, index=range(0, len(association)), columns=["disease","snorna"])
association_df.to_csv("../Research2/dataset/association_df.csv", index=False)
type(association_df)


# In[7]:


association_df.iloc[:,0]


# In[8]:


# creating a dataframe of diseases and snornas using index
diseases = []
snornas = []
for i in ds.index:
    diseases.append(i)
    
for i in ss.index:
    snornas.append(i)

# converting diseases and snornas lists to dataframe with a unique index (0 to n-1)
diseases_df = pd.DataFrame(diseases, index=range(len(diseases)), columns=['disease'])
snornas_df = pd.DataFrame(snornas, index=range(len(snornas)), columns=['snornas'])
len(diseases), len(snornas)


# In[9]:


diseases_df


# In[10]:


snornas_df


# In[11]:


import torch

# mapping a unique disease ID to the disease ID
unique_disease_id = association_df['disease'].unique()
unique_disease_id = pd.DataFrame(data={
    'disease': unique_disease_id,
    'mappedID': pd.RangeIndex(len(unique_disease_id)),
})
print("Mapping of disease IDs to consecutive values:")
print("==========================================")
print(unique_disease_id.head())

# mapping a unique snorna ID to the snorna ID
unique_snorna_id = association_df['snorna'].unique()
unique_snorna_id = pd.DataFrame(data={
    'snorna': unique_snorna_id,
    'mappedID': pd.RangeIndex(len(unique_snorna_id)),
})
print("Mapping of snorna IDs to consecutive values:")
print("==========================================")
print(unique_snorna_id.head())

# Perform merge to obtain the edges from snornas and diseases:
association_disease_id = pd.merge(association_df["disease"], unique_disease_id,
                            left_on='disease', right_on='disease', how='left')
association_disease_id = torch.from_numpy(association_disease_id['mappedID'].values)


association_snorna_id = pd.merge(association_df['snorna'], unique_snorna_id,
                            left_on='snorna', right_on='snorna', how='left')
association_snorna_id = torch.from_numpy(association_snorna_id['mappedID'].values)

# construct our `edge_index` in COO format
edge_index_disease_to_snorna = torch.stack([association_disease_id, association_snorna_id], dim=0)
# edge_index_snorna_to_disease = torch.stack([association_snorna_id, association_disease_id], dim=0) # reverse edge index

print("Final edge indices from diseases to snornas")
print("=================================================")
print(edge_index_disease_to_snorna)
print(edge_index_disease_to_snorna.shape)


# In[12]:


# disease and snorna features
disease_feat = torch.from_numpy(ds.values).to(torch.float) # disease features in total
snorna_feat = torch.from_numpy(ss.values).to(torch.float) # snorna features in total
disease_feat.size(), snorna_feat.size()


# In[13]:


# plotting feature distribution
fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=False, figsize=(11, 4))
fig.text(0.5, 0.0004, 'Similarity scores', ha='center')
fig.text(0.0008, 0.5, '', va='center', rotation='vertical')
# plt.suptitle("Feature distribution")
plt.tight_layout()

plt.subplot(1, 2, 1)
plt.hist(snorna_feat,bins=5)
plt.title("SnoRNAs")
plt.ylabel("Samples")

plt.subplot(1, 2, 2)
plt.hist(disease_feat,bins=8)
plt.title("Diseases")
plt.savefig('Similarity_score_distributions500dpi.png', dpi=500)
plt.show()


# In[14]:


#  initialize HeteroData object and pass in the necessary information
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

data = HeteroData()
# Saving node indices
data["disease"].node_id = torch.arange(len(unique_disease_id))
data["snorna"].node_id = torch.arange(len(ss))
# Adding node features and edge indices
data["disease"].x = disease_feat
data["snorna"].x = snorna_feat

data["disease", "associates_with", "snorna"].edge_index = edge_index_disease_to_snorna
# Adding reverse edges(GNN used this to pass messages in both directions)
data = T.ToUndirected()(data)
print(data)

# In[15]:


data.edge_index_dict


# In[16]:


data.x_dict


# In[17]:


data.edge_types # returns types of edges


# In[18]:


# plotting the graph
import networkx as nx
from matplotlib import pyplot as plt
import torch_geometric

snrs = []
for i in unique_disease_id['mappedID']:
    snrs.append(i)

G = torch_geometric.utils.to_networkx(data.to_homogeneous())
pos = nx.kamada_kawai_layout(G)
nx.draw(G, pos=pos, with_labels=False, node_color="#3d7336", edgecolors="#96bd2b", linewidths=0.5, node_size=200)  # Draw the original graph
# Draw a subgraph, reusing the same node positions
nx.draw(G.subgraph(snrs), pos=pos, node_color="#96bd2b", edgecolors="#3d7336", linewidths=0.5, node_size=300)
plt.axis('off')
plt.savefig('HetGraph500dpi.png', dpi=500)
plt.show()


# In[19]:


G.number_of_edges(), G.number_of_nodes()


# In[22]:


# split associations into training, validation, and test splits
transform = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    disjoint_train_ratio=0.3,
    neg_sampling_ratio=1.0,
    add_negative_train_samples=False,
    is_undirected = True, # added
    edge_types=("disease", "associates_with", "snorna"),
    rev_edge_types=("snorna", "rev_associates_with", "disease"),
)
train_data, val_data, test_data = transform(data)
print(train_data)


# In[23]:


# creating a mini-batch loader for generating subgraphs used as input into our GNN
import torch_sparse
from torch_geometric.loader import LinkNeighborLoader

# Defining seed edges
edge_label_index = train_data["disease", "associates_with", "snorna"].edge_label_index
edge_label = train_data["disease", "associates_with", "snorna"].edge_label
train_loader = LinkNeighborLoader(
    data=train_data,
    num_neighbors=[20] * 2,
    neg_sampling_ratio=2.0,
    edge_label_index=(("disease", "associates_with", "snorna"), edge_label_index),
    edge_label=edge_label,
    batch_size=3 * 32,
    shuffle=True,
)  

# Inspecting a sample
sampled_data = next(iter(train_loader))

print("Sampled mini-batch:")
print("===================")
print(sampled_data)

G = torch_geometric.utils.to_networkx(sampled_data.to_homogeneous())
# Plot the graph
nx.draw(G, with_labels=False)
plt.show()


# In[25]:


# Creating an Heterogeneous Link-level GNN
from torch_geometric.nn import SAGEConv, to_hetero, GATConv
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear
from torch import nn

from sklearn.ensemble import RandomForestClassifier

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels=32, num_heads=4):
        super(GNN, self).__init__()
        self.conv1 = SAGEConv((-1,-1), hidden_channels)
        self.attn1 = GATConv(hidden_channels, hidden_channels, heads=num_heads, add_self_loops=False) # attention mechanism
        self.conv2 = SAGEConv((-1,-1), hidden_channels)
        self.attn2 = GATConv(hidden_channels, hidden_channels, heads=num_heads, add_self_loops=False) # attention mechanism
        self.conv3 = SAGEConv((-1,-1), out_channels)
        self.attn3 = GATConv(out_channels, out_channels, heads=num_heads, add_self_loops=False) # attention mechanism
        self.num_heads=num_heads
        
        # Linear layer for node contribution penalty
        self.penalty_linear = Linear(out_channels, 1)
        
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor: # encoder
        # first GraphSAGE layer
        x = self.conv1(x, edge_index)
        x = x.relu()
        
        # Multihead attention
        x = self.attn1(x, edge_index)
        x = x.relu()
        x = x.view(-1, self.num_heads, x.shape[1] // self.num_heads)
        x = x.mean(dim=1)
        
        # second GraphSAGE layer
        x = self.conv2(x, edge_index)
        x = x.relu()
        
        # Multihead attention
        x = self.attn2(x, edge_index)
        x = x.relu()
        x = x.view(-1, self.num_heads, x.shape[1] // self.num_heads)
        x = x.mean(dim=1)
        
        # third GraphSAGE layer
        x = self.conv3(x, edge_index)
        x = x.relu()
        
        # Multihead attention
        x = self.attn3(x, edge_index)
        x = x.relu()
        x = x.view(-1, self.num_heads, x.shape[1] // self.num_heads)
        x = x.mean(dim=1)
        
        penalty = self.penalty_linear(x) # Compute node contribution penalty factor
        x = x * torch.exp(penalty) # Add penalty factor to output embeddings
        x = F.normalize(x, p=2, dim=1)  # Normalize output embeddings
        
        return x
    
# Our final classifier applies the dot-product between source and destination node embeddings
# to derive edge-level predictions
class Classifier(torch.nn.Module): # decoder
    def forward(self, x_disease: Tensor, x_snorna: Tensor, edge_label_index: Tensor) -> Tensor:
        # Converting node embeddings to edge-level representations
        edge_feat_disease = x_disease[edge_label_index[0]] # source node
        edge_feat_snorna = x_snorna[edge_label_index[1]] # target node
        concat_dot_product = (edge_feat_disease * edge_feat_snorna).sum(dim=-1)
#         print(concat_dot_product)
        return concat_dot_product

class Model(torch.nn.Module):
    def __init__(self, hidden_channels=128, num_graphs=3):
        super(Model, self).__init__()
        self.num_graphs = num_graphs
        self.graphs = torch.nn.ModuleList()
        for i in range(num_graphs):
            self.graphs.append(GNN(hidden_channels))
        
            # Since the dataset does not come with rich features, 
            # we also learn two embedding matrices for diseases and snornas
            self.disease_lin = torch.nn.Linear(27, hidden_channels)
            self.snorna_lin = torch.nn.Linear(220, hidden_channels)
            self.disease_emb = torch.nn.Embedding(data["disease"].num_nodes, hidden_channels)
            self.snorna_emb = torch.nn.Embedding(data["snorna"].num_nodes, hidden_channels)

            self.gnn = GNN(hidden_channels) # Instantiate homogeneous GNN
            self.gnn = to_hetero(self.gnn, metadata=data.metadata()) # Convert GNN model into a heterogeneous variant
            self.classifier = Classifier()
        
    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
          "disease": self.disease_lin(data["disease"].x) + self.disease_emb(data["disease"].node_id),
          "snorna": self.snorna_lin(data["snorna"].x) + self.snorna_emb(data["snorna"].node_id),
        } 
        # 'x_dict' holds feature matrices of all node types
        # 'edge_index_dict' holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict["disease"],
            x_dict["snorna"],
            data["disease", "associates_with", "snorna"].edge_label_index,
        )
        return pred
        
model = Model(hidden_channels=128)
print(model)


# In[26]:


# training heterogeneous link-level GNN
import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: {}".format(device))
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

loss_values =[] # training loss values

for epoch in range(1, 1501):
    total_loss = total_examples = 0
    for sampled_data in tqdm.tqdm(train_loader):
        optimizer.zero_grad()
        sampled_data.to(device)
        pred = model(sampled_data)
        ground_truth = sampled_data["disease", "associates_with", "snorna"].edge_label
        loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * pred.numel()
        total_examples += pred.numel()
        
        loss_values.append(total_loss / total_examples)

    print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")


# In[27]:


# plotting the loss curve
plt.figure(figsize=(6,4))
plt.xlabel("Total examples")
plt.ylabel("Loss")
plt.title("Training loss curve")
plt.plot(loss_values)
plt.savefig('loss_curve500dpi.png', dpi=500)
plt.show()


# In[28]:


# evaluate the GNN model
# we define a new LinkNeighborLoader that iterates over edges in the validation set
# obtaining predictions on validation edges
# then evaluate the performance by computing the AUC

# Define the validation seed edges:
edge_label_index = val_data["disease", "associates_with", "snorna"].edge_label_index
edge_label = val_data["disease", "associates_with", "snorna"].edge_label

val_loader = LinkNeighborLoader(
    data=val_data,
    num_neighbors=[30] * 3,
    edge_label_index=(("disease", "associates_with", "snorna"), edge_label_index),
    edge_label=edge_label,
    batch_size=128,
    shuffle=True,
)
sampled_data = next(iter(val_loader))
print(sampled_data)

G = torch_geometric.utils.to_networkx(sampled_data.to_homogeneous())
# Networkx seems to create extra nodes from our heterogeneous graph, so we remove them
isolated_nodes = [node for node in G.nodes() if G.out_degree(node) == 0]
[G.remove_node(i_n) for i_n in isolated_nodes]
# Plot the graph
nx.draw(G, with_labels=False)
plt.show()


# In[29]:


# training and testing accuracy
from sklearn.metrics import roc_auc_score, classification_report, RocCurveDisplay, roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix, f1_score
from scipy.interpolate import interp1d
import seaborn as sns

# this function converts predictions from continous to binary specifically for 
# use in the classification report which doesn't accept continuous labels
def binary_predictions(threshold, x):
    predictions_binary = (x > threshold).astype(int)
    return predictions_binary
    
# main model for training and testing
def train_val_accuracy(loader):
    tv_preds = []
    tv_ground_truths = []
    for sampled_data in tqdm.tqdm(loader):
        with torch.no_grad():
            sampled_data.to(device)
            tv_preds.append(model(sampled_data))
            tv_ground_truths.append(sampled_data["disease", "associates_with", "snorna"].edge_label)
    tv_preds = torch.cat(tv_preds, dim=0).cpu().numpy()
    tv_ground_truths = torch.cat(tv_ground_truths, dim=0).cpu().numpy()
    # tv_auc = roc_auc_score(tv_ground_truths, tv_preds)
    
    # plotting AUC Curve
    binary_ground_truths = np.array([1 if label == 2.0 or label == 1.0 else 0 for label in tv_ground_truths]) # converting ground truth values to {0, 1}
    
    # plotting the AUC using seaborn
    sns.set_style('white')
    sfpr, stpr, _ = roc_curve(binary_ground_truths, tv_preds)
    roc_auc = round(auc(sfpr, stpr), 2)
    sns.lineplot(x=sfpr, y=stpr, label=f'SAGESDA (AUC = {roc_auc})', errorbar=('ci', 99))
    sns.lineplot(x=[0,1], y=[0,1], color='black', linestyle='dashed')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC')
    plt.legend(loc='lower right')
    plt.savefig('AUC500dpi.png', dpi=500)
    plt.show()
    
    # Calculate AUPRC
    auprc = average_precision_score(binary_ground_truths, tv_preds)
    
    # Plotting AUPRC Curve
    sns.set_style('white')
    precision, recall, _ = precision_recall_curve(binary_ground_truths, tv_preds)
    auprc = round(auprc, 2)
    sns.lineplot(x=recall, y=precision, label=f'SAGESDA (AUPRC = {auprc})', errorbar=('ci', 99))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.savefig('AUPRC500dpi.png', dpi=500)
    plt.show()
    
    # converting predictions to binary so as to print a classification report
    binary_preds = binary_predictions(0.5, tv_preds)
    # classification report
    clf_report = classification_report(binary_ground_truths, binary_preds)
    print("Classification report")
    print(clf_report)
#     print(binary_preds)
#     print(binary_ground_truths)
#     print(tv_preds)

    # plotting confusion matrix
    cm = confusion_matrix(binary_ground_truths, binary_preds)    
    class_labels = ["Negative", "Positive"] # Define class labels for the confusion matrix
    sns.set(font_scale=1.2)  # Adjusting font size
    plt.figure(figsize=(6, 4))  # Adjusting figure size
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
    
    # f1-score
    f1 = f1_score(binary_ground_truths, binary_preds)
    print("F1-score:", f1)
    
    return roc_auc


# In[49]:


# training and validation accuracy
print("=============================================================================================================")
print("Training results")
training_auc = train_val_accuracy(train_loader)
print("Training AUC: {}".format(training_auc))
print("=============================================================================================================")
print("Testing results")
testing_auc = train_val_accuracy(val_loader)
print("Testing AUC: {}".format(testing_auc))


# In[52]:


# how to check for specific disease/snoRNA from the mapped dataframe
for i in unique_disease_id.index:
    print(unique_disease_id['disease'][i], unique_disease_id['mappedID'][i])


# In[54]:


# node_embeddings = model.get_node_embeddings()


# In[ ]:
