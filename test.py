import torch


def test_resnet():
    from model.resnet import ResNet
    model = ResNet("6b128f", 2, 6, 128, 128, False)
    state = torch.zeros(size=(16, 2, 7, 7))
    p, v = model(state)
    print(p.shape)
    print(v.shape)


def test_conv3x_nnue():
    from model.conv3x_nnue import conv3NNUE
    model = conv3NNUE("conv3x_nnue", 2, 128, 128, 2)
    state = torch.zeros(size=(16, 2, 7, 7))
    v = model(state)
    print(v.shape)


def test_flat_conv3x_nnue():
    from model.flat_conv3x_nnue import FlatConv3x3NNUE
    model = FlatConv3x3NNUE("flat_conv3x_nnue", 2, 128, 128)
    state = torch.zeros(size=(16, 2, 7, 7))
    v = model(state)
    print(v.shape)


def test_ladder_conv3x_nnue():
    from model.ladder_conv3x_nnue import LadderConvNNUE
    model = LadderConvNNUE("ladder_conv3x_nnue", 2, 128, 128, 2)
    state = torch.zeros(size=(16, 2, 7, 7))
    v = model(state)
    print(v.shape)


def test_ladder_conv6x_nnue():
    from model.ladder_conv6x_nnue import LadderConvNNUE
    model = LadderConvNNUE("ladder_conv6x_nnue", 2, 128, 128, 2)
    state = torch.zeros(size=(16, 2, 7, 7))
    v = model(state)
    print(v.shape)


def test_splited_conv5x_nnue():
    from model.splited_conv5x_nnue import SplitedConv5NNUE
    model = SplitedConv5NNUE("splited_conv2x_nnue", 2, 128, 128, 2)
    state = torch.zeros(size=(16, 2, 7, 7))
    v = model(state)
    print(v.shape)


test_splited_conv5x_nnue()
