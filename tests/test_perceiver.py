from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from fixtures import *
from perceiver_pytorch import Perceiver
from tests.compare_params import capture_params, compare_parameters


def test_all_parameters_change(image_inputs, targets):
    model = Perceiver(
        input_channels=3,  # number of channels for each token of the input
        input_axis=2,  # number of axis for input data (2 for images, 3 for video)
        num_freq_bands=6,  # number of freq bands, with original value (2 * K + 1)
        max_freq=10.,  # maximum frequency, hyperparameter depending on how fine the data is
        depth=depth,  # depth of net
        num_latents=256,
        # number of latents, or induced set points, or centroids. different papers giving it different names
        cross_dim=512,  # cross attention dimension
        latent_dim=512,  # latent dimension
        cross_heads=1,  # number of heads for cross attention. paper said 1
        latent_heads=8,  # number of heads for latent self attention, 8
        cross_dim_head=64,
        latent_dim_head=64,
        num_classes=num_classes,  # output number of classes
        attn_dropout=0.,
        ff_dropout=0.,
        weight_tie_layers=False  # whether to weight tie layers (optional, as indicated in the diagram)
    )

    result = model(image_inputs)

    optimizer = SGD(
        # Make learning rate large enough that differences in paramerers are clear:
        lr=0.1,
        params=model.parameters())
    criterion = CrossEntropyLoss()
    loss = criterion(result, targets)
    loss.backward()
    before_params = capture_params(model)
    optimizer.step()
    after_params = capture_params(model)
    compare_parameters(before_params, after_params)
