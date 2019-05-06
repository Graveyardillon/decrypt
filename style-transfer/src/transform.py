import tensorflow as tf, pdb
#tensorflowをimportする
#デバッグ用のモジュールpdbをimportする

WEIGHTS_INIT_STDEV = .1
#重みの初期標準偏差を0.1とする

#imageは画像を読み込むための引数で、バッチサイズ、画像の縦、横、深さを保有している
def net(image):
    #画風変換のネットワークを通すための関数net()
    conv1 = _conv_layer(image, 32, 9, 1)
    #
    conv2 = _conv_layer(conv1, 64, 3, 2)
    conv3 = _conv_layer(conv2, 128, 3, 2)
    resid1 = _residual_block(conv3, 3)
    resid2 = _residual_block(resid1, 3)
    resid3 = _residual_block(resid2, 3)
    resid4 = _residual_block(resid3, 3)
    resid5 = _residual_block(resid4, 3)
    conv_t1 = _conv_tranpose_layer(resid5, 64, 3, 2)
    conv_t2 = _conv_tranpose_layer(conv_t1, 32, 3, 2)
    conv_t3 = _conv_layer(conv_t2, 3, 9, 1, relu=False)
    preds = tf.nn.tanh(conv_t3) * 150 + 255./2
    return preds

#netはニューラルネットワークのレイヤーも4しくは画像
#num_filtersはフィルタの数（厚さ？）
def _conv_layer(net, num_filters, filter_size, strides, relu=True):
    #畳み込み層を再現した関数_conv_layer()
    weights_init = _conv_init_vars(net, num_filters, filter_size)
    #畳み込み層の重みを初期化する
    strides_shape = [1, strides, strides, 1]
    #要素をまたぐときのstrideの形を定義し、strides_shapeに代入する
    net = tf.nn.conv2d(net, weights_init, strides_shape, padding='SAME')
    #CNNに画像もしくはレイヤーを入力
    net = _instance_norm(net)
    #
    if relu:
        net = tf.nn.relu(net)

    return net

def _conv_tranpose_layer(net, num_filters, filter_size, strides):
    weights_init = _conv_init_vars(net, num_filters, filter_size, transpose=True)

    batch_size, rows, cols, in_channels = [i.value for i in net.get_shape()]
    new_rows, new_cols = int(rows * strides), int(cols * strides)
    # new_shape = #tf.pack([tf.shape(net)[0], new_rows, new_cols, num_filters])

    new_shape = [batch_size, new_rows, new_cols, num_filters]
    tf_shape = tf.stack(new_shape)
    strides_shape = [1,strides,strides,1]

    net = tf.nn.conv2d_transpose(net, weights_init, tf_shape, strides_shape, padding='SAME')
    net = _instance_norm(net)
    return tf.nn.relu(net)

def _residual_block(net, filter_size=3):
    tmp = _conv_layer(net, 128, filter_size, 1)
    return net + _conv_layer(tmp, 128, filter_size, 1, relu=False)

#netはニューラルネットワークのレイヤーもしくは画像
#trainはよくわからんけどtrueの引数
def _instance_norm(net, train=True):
    #
    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    #ネットワークの形をそれぞれ抽出し、またそれぞれの変数に格納していく
    #batch_size変数にバッチサイズを代入する
    #rowsには画像の横の大きさが格納される
    #colsには画像の縦の大きさが格納される
    #channelsには、現在のノードの数が格納される
    var_shape = [channels]
    #channelsのリストをvar_shapeに格納する
    mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
    
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))
    epsilon = 1e-3
    normalized = (net-mu)/(sigma_sq + epsilon)**(.5)
    return scale * normalized + shift

#netはニューラルネットワークのレイヤーもしくは画像
#out_channelsは出力する側のノードの数
#filter_sizeはCNNのフィルタのサイズを指定する引数
#transposeは、転置行列かどうかを指定するためのBool型の引数
def _conv_init_vars(net, out_channels, filter_size, transpose=False):
    #畳み込み層の重みを初期化するための関数_conv_init_vars()
    _, rows, cols, in_channels = [i.value for i in net.get_shape()]
    #ネットワークの形をそれぞれ抽出し、またそれぞれの変数に格納していく
    #_（ワイルドカード）にはバッチサイズが代入されるが、ワイルドカードなので破棄される
    #rowsには画像の横の大きさが格納される
    #colsには画像の縦の大きさが格納される
    #in_channelsには画像の深さが格納される
    if not transpose:
        #transpose引数で転置行列を指定されていない場合にこちら側に分岐する
        weights_shape = [filter_size, filter_size, in_channels, out_channels]
        #正方形で縦横の大きさがfilter_sizeのフィルタを定義する
        #重みの形をweight_shape変数に格納する
        #深さはそのままノードの数となりうるので、in_channelsとout_channelsがそれぞれ
        #入力される側のノードの数、出力する側のノードの数となることができる
    else:
        #転置行列が指定されていた場合
        weights_shape = [filter_size, filter_size, out_channels, in_channels]
        #転置行列は行列の縦と横が入れ替わっているので、in_channelsとout_channelsを入れ替えると
        #逆伝播法のときに i*j x j*k = i*k を i*k x k*j = i*jにすることができる

    weights_init = tf.Variable(tf.truncated_normal(weights_shape, \
        stddev=WEIGHTS_INIT_STDEV, seed=1), dtype=tf.float32)
    #2σの位置で切断された正規分布（95%の事象が含まれる）で、tf.Variableを用いて重みを初期化する
    #戻り値のテンソルの次元を指定し、生成する切断正規分布の標準偏差をWEIGHTS_INIT_STDEVで指定
    #そそいて戻り値のテンソルの型をfloat32に指定する
    return weights_init
    #初期化されたテンソルのweights_initをreturnする
