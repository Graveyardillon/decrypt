import tensorflow as tf, pdb
#tensorflowをimportする
#デバッグ用のモジュールpdbをimportする

WEIGHTS_INIT_STDEV = .1
#重みの初期標準偏差を0.1とする

#imageは画像を読み込むための引数で、バッチサイズ、画像の縦、横、深さを保有している
def net(image):
    #画風変換のネットワークを通すための関数net()
    conv1 = _conv_layer(image, 32, 9, 1)
    #CNNを通す
    conv2 = _conv_layer(conv1, 64, 3, 2)
    #CNNを通す
    conv3 = _conv_layer(conv2, 128, 3, 2)
    #CNNを通す
    resid1 = _residual_block(conv3, 3)
    #ResNetを通す
    resid2 = _residual_block(resid1, 3)
    #ResNetを通す
    resid3 = _residual_block(resid2, 3)
    #ResNetを通す
    resid4 = _residual_block(resid3, 3)
    #ResNetを通す
    resid5 = _residual_block(resid4, 3)
    #ResNetを通す
    conv_t1 = _conv_tranpose_layer(resid5, 64, 3, 2)
    #transposeされたレイヤを通して、deconvすることでサイズを再現する
    conv_t2 = _conv_tranpose_layer(conv_t1, 32, 3, 2)
    #またdeconvする
    conv_t3 = _conv_layer(conv_t2, 3, 9, 1, relu=False)
    #もう一回convする
    preds = tf.nn.tanh(conv_t3) * 150 + 255./2
    #活性化関数tanhを使って活性化して変数preds
    return preds
    #preds変数をreturnする

#netはニューラルネットワークのレイヤーもしくは画像
#num_filtersはフィルタの数（厚さ？）
#filter_sizeはフィルタの大きさ
#stridesはフィルタの移動量
#reluは活性化関数としてrelu関数を使うかどうかの指定
def _conv_layer(net, num_filters, filter_size, strides, relu=True):
    #畳み込み層を再現した関数_conv_layer()
    weights_init = _conv_init_vars(net, num_filters, filter_size)
    #畳み込み層の重みを初期化する
    strides_shape = [1, strides, strides, 1]
    #要素をまたぐときのstrideの形を定義し、strides_shapeに代入する
    net = tf.nn.conv2d(net, weights_init, strides_shape, padding='SAME')
    #CNNに画像もしくはレイヤーを入力
    net = _instance_norm(net)
    #netを正規化する
    if relu:
        #relu関数を使うことが指定されていたら
        net = tf.nn.relu(net)
        #netをrelu関数にかけて活性化する

    return net
    #netをreturnする

#netは画像もしくはネットワークの層
#num_filtersはフィルタの数（厚さ）
#filter_sizeはフィルタの大きさ
#stridesはフィルタの移動量
def _conv_tranpose_layer(net, num_filters, filter_size, strides):
    #転置行列を用いてdeconvを行う
    weights_init = _conv_init_vars(net, num_filters, filter_size, transpose=True)
    #重みの値を初期化する
    batch_size, rows, cols, in_channels = [i.value for i in net.get_shape()]
    #ネットワークの型をそれぞれ抽出し、またそれぞれの変数に格納していく
    #batch_sizeにはバッチサイズが代入される
    #rowsには画像の横の大きさが格納される
    #colsには画像の縦の大きさが格納される
    #in_channelsには画像の深さが格納される
    new_rows, new_cols = int(rows * strides), int(cols * strides)
    #フィルタの移動量と画像の横の大きさの積をnew_rows変数に格納する
    #フィルタの移動量を画像の縦の大きさの積をnew_cols変数に格納する

    # new_shape = #tf.pack([tf.shape(net)[0], new_rows, new_cols, num_filters])

    new_shape = [batch_size, new_rows, new_cols, num_filters]
    #上で作ったnew_rowsとnew_colsを用いて新しい変数new_shapeを定義する
    tf_shape = tf.stack(new_shape)
    #new_shapeに格納されている4つの要素をtf.stack()で結合させてテンソルを作り出す
    strides_shape = [1,strides,strides,1]
    #フィルタの移動量を定義してstrides_shape変数に格納する
    net = tf.nn.conv2d_transpose(net, weights_init, tf_shape, strides_shape, padding='SAME')
    #conv2dを転置行列で行うことで、deconvolutionをすることができる
    net = _instance_norm(net)
    #行列をDeconvした後に正規化する
    return tf.nn.relu(net)
    #正規化した後の行列を活性化関数Reluを使って活性化し、returnする

# netは画像やネットワーク層
# filter_sizeはフィルタの大きさを指定する
def _residual_block(net, filter_size=3):
    # ResNetを再現した関数_residual_block()
    tmp = _conv_layer(net, 128, filter_size, 1)
    # CNNを通す
    return net + _conv_layer(tmp, 128, filter_size, 1, relu=False)
    # CNNを通した後の行列と最初の行列を足し算してreturnする

# netはニューラルネットワークのレイヤーもしくは画像
# trainはよくわからんけどtrueの引数
def _instance_norm(net, train=True):
    # 行列を正規化するための関数
    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    # ネットワークの形をそれぞれ抽出し、またそれぞれの変数に格納していく
    # batch_size変数にバッチサイズを代入する
    # rowsには画像の横の大きさが格納される
    # colsには画像の縦の大きさが格納される
    # channelsには、現在のノードの数が格納される
    var_shape = [channels]
    # channelsのリストをvar_shapeに格納する
    mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
    # tf.nn.moments関数を使って、netの平均と、[1,2]からの差を分散として求める
    # keep_dimsをtrueにして、入力時の形を保つようにする
    shift = tf.Variable(tf.zeros(var_shape))
    # 現在のノードの数（形）のゼロ行列を作り出し、shift変数に代入する
    scale = tf.Variable(tf.ones(var_shape))
    # 現在のノードの数（形）の行列を１で初期化し、scale変数に代入する
    epsilon = 1e-3
    # 1x10^-3をepsilon変数に代入する
    normalized = (net-mu)/(sigma_sq + epsilon)**(.5)
    # 値を正規化する (X_old - X_new) / X_std
    return scale * normalized + shift
    # scaleに入っている行列を使って初期化することで、元の形に合わせる。

# netはニューラルネットワークのレイヤーもしくは画像
# out_channelsは出力する側のノードの数
# filter_sizeはCNNのフィルタのサイズを指定する引数
# transposeは、転置行列かどうかを指定するためのBool型の引数
def _conv_init_vars(net, out_channels, filter_size, transpose=False):
    # 畳み込み層の重みを初期化するための関数_conv_init_vars()
    _, rows, cols, in_channels = [i.value for i in net.get_shape()]
    # ネットワークの形をそれぞれ抽出し、またそれぞれの変数に格納していく
    # _（ワイルドカード）にはバッチサイズが代入されるが、ワイルドカードなので破棄される
    # rowsには画像の横の大きさが格納される
    # colsには画像の縦の大きさが格納される
    # in_channelsには画像の深さが格納される
    if not transpose:
        # transpose引数で転置行列を指定されていない場合にこちら側に分岐する
        weights_shape = [filter_size, filter_size, in_channels, out_channels]
        # 正方形で縦横の大きさがfilter_sizeのフィルタを定義する
        # 重みの形をweight_shape変数に格納する
        # 深さはそのままノードの数となりうるので、in_channelsとout_channelsがそれぞれ
        # 入力される側のノードの数、出力する側のノードの数となることができる
    else:
        # 転置行列が指定されていた場合
        weights_shape = [filter_size, filter_size, out_channels, in_channels]
        # 転置行列は行列の縦と横が入れ替わっているので、in_channelsとout_channelsを入れ替えると
        # 逆伝播法のときに i*j x j*k = i*k を i*k x k*j = i*jにすることができる

    weights_init = tf.Variable(tf.truncated_normal(weights_shape, \
        stddev=WEIGHTS_INIT_STDEV, seed=1), dtype=tf.float32)
    # 2σの位置で切断された正規分布（95%の事象が含まれる）で、tf.Variableを用いて重みを初期化する
    # 戻り値のテンソルの次元を指定し、生成する切断正規分布の標準偏差をWEIGHTS_INIT_STDEVで指定
    # そそいて戻り値のテンソルの型をfloat32に指定する
    return weights_init
    # 初期化されたテンソルのweights_initをreturnする
