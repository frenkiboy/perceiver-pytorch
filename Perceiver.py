# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## Multi - Modal - Perceiver

# %%
# ---------------------------------------------------------------------------- #
import anndata as ad
import scanpy as sc
import pandas as pd
import torch
import numpy as np
import torch.optim as optim
import sys
import os
from importlib import reload
from einops import rearrange
sys.path.append('./perceiver-multi-modality-pytorch')
from perceiver_pytorch.multi_modality_perceiver import MultiModalityPerceiver, InputModality

# %% [markdown]
# ## MultiModality
# %%
# ---------------------------------------------------------------------------- #
from torch.utils.data import Dataset
from torch.utils.data import DataLoader



# ---------------------------------------------------------------------------- #
class ArcasDataset(Dataset):
    def __init__(model, path, variable = 'cancer_type', select = None):

      cdata  = pd.read_csv(os.path.join(path, 'cdata.tsv'), sep='\t')
      cdata['variable'] = cdata[variable]
      cdata = cdata[~cdata.variable.isnull()]

      nclass = cdata.variable.value_counts()
      nclass = nclass[nclass > 10]
      cdata  = cdata[cdata.variable.isin(nclass.index)]


      adata = pd.read_table(os.path.join(path, 'gex.tsv'))
      adata.index = adata['gene_id']
      adata = adata.drop(['gene_id'], axis=1)


      if not select == None:
          adata = adata.iloc[1:select,:]

      adata = adata[cdata.sample_id]


      model.cdata = cdata
      model.adata = adata

    def __len__(model):
        return model.cdata.shape[0]

    def getData(model, data = None):

        if data == None:
            data = torch.tensor(model.adata.to_numpy().T).float()

        rdat = rearrange(data, "(b w) h -> b h w", w = 1)
        rdat = rdat.float()
        return rdat

    def getLabs(model):
        labs = torch.tensor(pd.Categorical(model.cdata.variable).codes)
        return labs

    def __getitem__(model, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = model.getData()[idx,:,:]
        labs = model.getLabs()[idx]

        return (data, labs)

# ---------------------------------------------------------------------------- #
batch_size = 256

n_epochs   = 10

learning_rate = 0.1

select = 30000

path = '/data/local/arcas/datasets/primary/CCLE'

# ---------------------------------------------------------------------------- #
if __name__ == '__main__':

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'

    arcas_dataset = ArcasDataset(path = path, select = select)
    train_dataloader = DataLoader(arcas_dataset, batch_size=batch_size, shuffle=True)

    # %%
    data1 = InputModality(
        name           = 'data1',
        input_channels = 1,  # number of channels for mono audio
        input_axis     = 1,  # number of axes, 2 for images
        num_freq_bands = 0,  # number of freq bands, with original value (2 * K + 1)
        max_freq       = 8.,  # maximum frequency, hyperparameter depending on how fine the data is
    )

    data2 = InputModality(
        name           = 'data2',
        input_channels = 1,  # number of channels for mono audio
        input_axis     = 1,  # number of axes, 2 for images
        num_freq_bands = 0,  # number of freq bands, with original value (2 * K + 1)
        max_freq       = 8.,  # maximum frequency, hyperparameter depending on how fine the data is
    )

    model = MultiModalityPerceiver(
        modalities      = (data1, data2),
        depth           = 2,  # depth of net, combined with num_latent_blocks_per_layer to produce full Perceiver
        num_latents     = 32,
        # number of latents, or induced set points, or centroids. different papers giving it different names
        latent_dim      = 32,  # latent dimension
        cross_heads     = 1,  # number of heads for cross attention. paper said 1
        latent_heads    = 4,  # number of heads for latent model attention, 8
        cross_dim_head  = 32,
        latent_dim_head = 32,
        num_classes     = len(arcas_dataset.cdata.variable.unique()),  # output number of classes
        attn_dropout    = 0.5,
        ff_dropout      = 0.5,
        weight_tie_layers = True,
        num_latent_blocks_per_layer = 1,
        use_gelu        = True  # Note that this parameter is 1 in the original Lucidrain implementation
        # whether to weight tie layers (optional, as indicated in the diagram)
    )
    model = model.float()
    model = model.to(device)
    model.train()


    # %%
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, amsgrad=True)

    # loss      = torch.nn.CrossEntropyLoss()
    # loss      = torch.nn.MSELoss()


    # %% [markdown]
    # ## Model Training

    # %%
    for epoch in range(n_epochs):  # loop over the dataset multiple times

      running_loss = 0.0

      for i, batch in enumerate(train_dataloader):

        data, labels = batch
        data   = data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        # out = model({"data1" : data, "data2" : data})
        out = model({"data1" : data})
        target = torch.max(out, axis=1)[1]

        loss = torch.nn.CrossEntropyLoss()(out, labels.long())
        # loss = loss(target.float(), labels.float())
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss = loss.item()
        if i % 5 == 0:    # print every 2000 mini-batches
            print('[%d, %6d] loss: %.6f' % (epoch + 1, i + 1, running_loss))
        # running_loss = 0.0

    # %% [markdown]
    # ## Testing

    # %%
    # testing the trained model

    model = model.eval()

    rdat = arcas_dataset.getData()
    labs = arcas_dataset.getLabs()

    predict = model({"data1" : rdat})

    input_max, input_indexes = torch.max(predict, axis=1)


    out = pd.DataFrame({'true' : labs, 'pred' : input_indexes.numpy()})
    # %%
    print(out.head)

# %% [markdown]
# ## Test
