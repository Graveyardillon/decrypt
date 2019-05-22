# Copyright (c) 2015-2016 Anish Athalye. Released under GPLv3.

import tensorflow as tf
# tensorflowをimportする
import numpy as np
# 行列演算用のライブラリnumpyをimportする
import scipy.io
# ファイルの入出力用のライブラリscipy.ioをimportする
import pdb
# デバッグ用のライブラリpdbをimportする

MEAN_PIXEL = np.array([ 123.68 ,  116.779,  103.939])
# vgg用のRGB平均値

# data_pathは読み取りたいmatlabファイルのパス
# input_imageは画像を格納する引数
def net(data_path, input_image):
    # net関数は実際に畳み込みなどの処理を行うための関数
    layers = (
        # 画風変換で用いるネットワークの層を格納しておく
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    data = scipy.io.loadmat(data_path)
    # matlabファイルを読み取る関数scipy.io.loadmat()を使う
    # data_pathに格納されているvggのパスから、matlabファイルの変数をすべて読み取る
    mean = data['normalization'][0][0][0]
    # matlabファイルから読み取ったdata変数に格納されている、
    # 3次元のnormalizationワークスペース変数の中身をmean変数に代入する
    mean_pixel = np.mean(mean, axis=(0, 1))
    # mean_pixel変数に、正規化するための値meanを1次元の変数に行列を直して格納する
    weights = data['layers'][0]
    # matlabファイルの中に書かれているlayersワークスペース変数から重みの値を取り出し、
    # layers変数に格納する

    net = {}
    # net変数を初期化する
    current = input_image
    # current変数にinput_imageに格納されている画像を格納する
    for i, name in enumerate(layers):
        # layersの中身を、enumerate関数を用いてインデックス番号付きの要素としてforループさせる
        # この場合、iがインデックス番号でnameが層の名前（convとかreluとか）
        kind = name[:4]
        # インデックス4つ分をnameから取り出して、kind変数に格納する
        # conv, relu, conv, relu
        if kind == 'conv':
            # kind変数の中身がconvである場合true
            kernels, bias = weights[i][0][0][0][0]
            # kernels変数の方には実際の重みの値を格納する（たぶん）
            # bias変数の方にはバイアスとなるベクトルが格納される

            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            # kernelsに入っている行列を転置行列にしてdeconvする
            bias = bias.reshape(-1)
            # bias変数の中身を一次ベクトルに変換して再格納する
            current = _conv_layer(current, kernels, bias)
            # current変数を畳み込みレイヤを通したあとに再度格納する
        elif kind == 'relu':
            # kind変数の中身がreluだった場合
            current = tf.nn.relu(current)
            # current変数をrelu関数で活性化して、current変数に再格納する
        elif kind == 'pool':
            # kind変数の中身がpoolだった場合
            current = _pool_layer(current)
            # current変数をプーリングして再度格納する
        net[name] = current
        # netリストのnameの要素の場所にcurrentの中身を格納する

    assert len(net) == len(layers)
    # net変数の要素の数とlayers変数の要素の数が違ったらAssertionError
    return net
    # net変数をreturnする

# inputは畳み込むための画像
# weightsは重みの値を格納する変数
# biasはバイアスの値が格納されている
def _conv_layer(input, weights, bias):
    # input画像をweightsの重みとbiasのバイアスを加算してreturnする関数
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1),
            padding='SAME')
    # inputを畳み込み層に入れて畳み込み、conv変数に格納する
    return tf.nn.bias_add(conv, bias)
    # 畳み込んだ画像convにbiasを加算してreturnする

# inputはプーリングするための画像
def _pool_layer(input):
    # 画像をプーリングするための関数_pool_layer()
    return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
            padding='SAME')
    # 最大プーリングをしてreturnする

# image変数は画像を格納するための変数
def preprocess(image):
    # 訓練データの前処理を行うための関数preprocess()
    return image - MEAN_PIXEL
    # 画像からRGBの平均値をマイナスする

# image変数は画像を格納するための変数
def unprocess(image):
    # 訓練データの前処理を打ち消すための関数unprocess()
    return image + MEAN_PIXEL
    # 画像にRGBの平均値をプラスする
