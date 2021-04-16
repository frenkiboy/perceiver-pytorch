![PyPI](https://img.shields.io/pypi/v/perceiver-multi-modality-pytorch.svg)
![PyPI](https://img.shields.io/pypi/pyversions/perceiver-multi-modality-pytorch.svg)
![PyPI](https://img.shields.io/github/license/fac2003/perceiver-mutli-modality-pytorch.svg)
<img src="./perceiver.png" width="600px"></img>

## Multi Modality Perceiver - Pytorch

Implementation of <a href="https://arxiv.org/abs/2103.03206">Perceiver</a>, with support for multi-modality 
inputs. Fork of (lucidrains repo)[https://github.com/lucidrains/perceiver-pytorch] extended for
multi-modality and support for text embedding splits chunking across layers.

## Install
To install the Perceiver implementation with multi-modality (also includes without multi-modality):
```bash
$ pip install perceiver-multi-modality-pytorch
```
Import with:
```python
from perceiver_pytorch.modalities import modality_encoding
from perceiver_pytorch.multi_modality_perceiver import  MultiModalityPerceiver, InputModality
```
See tests/test_multimodality_perceiver.py
or 
```python
from perceiver_pytorch.modalities import InputModalityWithEmbedding
from perceiver_pytorch.multi_modality_with_text_perceiver import MultiModalityWithTextPerceiver
```
See tests/test_multimodality_with_text_perceiver.py

To install the Perceiver implementation, follow instructions at the 
(lucidrains repo)[https://github.com/lucidrains/perceiver-pytorch]:


## Usage

```python
import torch
from perceiver_pytorch import Perceiver

model = Perceiver(
    input_channels = 3,          # number of channels for each token of the input
    input_axis = 2,              # number of axis for input data (2 for images, 3 for video)
    num_freq_bands = 6,          # number of freq bands, with original value (2 * K + 1)
    max_freq = 10.,              # maximum frequency, hyperparameter depending on how fine the data is
    depth = 6,                   # depth of net
    num_latents = 256,           # number of latents, or induced set points, or centroids. different papers giving it different names
    latent_dim = 512,            # latent dimension
    cross_heads = 1,             # number of heads for cross attention. paper said 1
    latent_heads = 8,            # number of heads for latent self attention, 8
    cross_dim_head = 64,
    latent_dim_head = 64,
    num_classes = 1000,          # output number of classes
    attn_dropout = 0.,
    ff_dropout = 0.,
    weight_tie_layers = False    # whether to weight tie layers (optional, as indicated in the diagram)
)

img = torch.randn(1, 224, 224, 3) # 1 imagenet image, pixelized

model(img) # (1, 1000)
```

## Multi-modality perceiver
An attractive feature of the perceiver architecture is that it can process multiple modalities of data 
in the same batch. This is not obvious from the perceiver forward signature shown above, but a relatively
modest change can support processing video, images and audio with a single model, in one forward.
This feature is demonstrated by the MultiModalityPerceiver, contributed by Fabien Campagne.

```python
from perceiver_pytorch.multi_modality_perceiver import MultiModalityPerceiver, InputModality


image_inputs= torch.rand(size=(3, 260, 260, 3), requires_grad=True)
video_inputs= torch.rand(size=(3, 32, 260, 260, 3), requires_grad=True)
audio_inputs= torch.rand(size=(3, 44100,1), requires_grad=True)

video_modality = InputModality(
    name='video',
    input_channels=3,  # number of channels for each token of the input
    input_axis=3,  # number of axes, 3 for video)
    num_freq_bands=6,  # number of freq bands, with original value (2 * K + 1)
    max_freq=4.,  # maximum frequency, hyperparameter depending on how fine the data is
)
image_modality = InputModality(
    name='image',
    input_channels=3,  # number of channels for each token of the input
    input_axis=2,  # number of axes, 2 for images
    num_freq_bands=6,  # number of freq bands, with original value (2 * K + 1)
    max_freq=4.,  # maximum frequency, hyperparameter depending on how fine the data is
)
audio_modality = InputModality(
    name='audio',
    input_channels=1,  # number of channels for mono audio
    input_axis=1,  # number of axes, 2 for images
    num_freq_bands=6,  # number of freq bands, with original value (2 * K + 1)
    max_freq=8.,  # maximum frequency, hyperparameter depending on how fine the data is
)
model = MultiModalityPerceiver(
    modalities=(video_modality, image_modality, audio_modality),
    depth=8,  # depth of net, combined with num_latent_blocks_per_layer to produce full Perceiver
    num_latents=12,
    # number of latents, or induced set points, or centroids. different papers giving it different names
    latent_dim=64,  # latent dimension
    cross_heads=1,  # number of heads for cross attention. paper said 1
    latent_heads=8,  # number of heads for latent self attention, 8
    cross_dim_head=64,
    latent_dim_head=64,
    num_classes=1000,  # output number of classes
    attn_dropout=0.,
    ff_dropout=0.,
    weight_tie_layers=True,
    num_latent_blocks_per_layer=6 # Note that this parameter is 1 in the original Lucidrain implementation
    # whether to weight tie layers (optional, as indicated in the diagram)
)
result = model({'image': image_inputs,
                'video': video_inputs,
                'audio': audio_inputs})
```
### Text perceiver
While the Perceiver architecture described by [jaegle2021perceiver] could support text if text was 
embedded and each dimension of the embedding provided as a channel in the input, this introduces a
mismatch between the text embedding dimension (typically large, 512/768 or more) and the number of 
channels used for video and images (typically 3 channels, one for red, green and blue),  or  audio 
(1 for mono or 2 for stereo channels). When training text embeddings from scratch, this creates an 
opportunity, because there should be no need for the perceiver to attend to the entire text 
embedding in each layer. If we  split the text embedding into as many chunks as there are layers in
a perceiver, we reduce how much we need to pad other modalities,  and introduce a structure to the 
learned embeddings, were  parts of the text embedding can specialize according to the needs of each
layer. The perceiver implementation provided in this repo can be used to explore the question of 
whether splitting text embeddings across layers is beneficial (you would compare the performance of
MultiModalityWithTextPerceiver with that of MultiModalityPerceiver). 

## Citations

```bibtex
@misc{jaegle2021perceiver,
    title   = {Perceiver: General Perception with Iterative Attention},
    author  = {Andrew Jaegle and Felix Gimeno and Andrew Brock and Andrew Zisserman and Oriol Vinyals and Joao Carreira},
    year    = {2021},
    eprint  = {2103.03206},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
@misc{campagne2021textperceiver,
    title   = {Adapting Perceiver for learning with text modalities},
    author  = {Fabien Campagne},
    year    = {2021},
    eprint  = {unpublished results},
}
```
