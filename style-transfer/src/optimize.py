from __future__ import print_function
#python3系の機能からprint関数を持ってくる
import functools
#高階関数のためのモジュールfunctoolsをimportする
import vgg, pdb, time
#vgg.pyをimportする
#デバッグ用のライブラリpdbをimportする
#時間の概念を扱うためんおライブラリtimeをimportする
import tensorflow as tf, numpy as np, os
#tensorflowをimportする
#行列演算用のライブラリnumpyをimportする
#ファイルやディレクトリ操作用のライブラリosをimportする
import transform
#transform.pyをimportする
from utils import get_img
#utils.pyからget_img関数をimportする

STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
#NNで使うためのレイヤの名前
CONTENT_LAYER = 'relu4_2'
#
DEVICES = 'CUDA_VISIBLE_DEVICES'
#GPUの設定を読み込むための変数

# np arr, np arr
#content_targetsはコンテンツ画像
#style_targetはスタイル画像
#content_weightはコンテンツ画像の損失の重み
#style_weightはスタイル画像の損失の重み
#tv_weightは
#vgg_pathはvggへのパス
#print_iterationsはチェックポイントに到達するまでの数
#batch_sizeはバッチサイズ
#save_pathはチェックポイントのファイルを保存するためのディレクトリのパス
#slowはデバッグ用の機能"slow"を利用するかを指定するための引数
#learning_rateは学習率
#debugはデバッグモードがONかどうかを指定する引数
def optimize(content_targets, style_target, content_weight, style_weight,
             tv_weight, vgg_path, epochs=2, print_iterations=1000,
             batch_size=4, save_path='saver/fns.ckpt', slow=False,
             learning_rate=1e-3, debug=False):
    #パラメータの最適化をするための関数optimizer()
    if slow:
        #slowが指定されていればTrue
        batch_size = 1
        #バッチサイズを１と指定する
    mod = len(content_targets) % batch_size
    #コンテンツ画像を効率良くバッチサイズに割り当てて学習できるかを見ておく
    #content_targetsとバッチサイズの剰余をmodに格納する
    if mod > 0:
        #modが0より大きければTrue
        print("Train set has been trimmed slightly..")
        #訓練用の画像を少しだけ削減して使うことをコンソールに書き込む
        content_targets = content_targets[:-mod]
        #content_targetsに格納されている画像セットから、剰余の分を後ろから削減する

    style_features = {}
    #style_features変数を初期化する

    batch_shape = (batch_size,256,256,3)
    #バッチサイズを考慮して4次元のテンソルを作り、batch_shape変数に格納する
    style_shape = (1,) + style_target.shape
    #スタイル画像にもう一次元追加して4次元のテンソルを作り、style_shape変数に格納する

    print(style_shape)
    #スタイル画像の形をコンソールに出力する

    # precompute style features
    with tf.Graph().as_default(), tf.device('/cpu:0'), tf.Session() as sess:
        #tensorflowのグラフを作成し、デバイスをcpuに設定し、tf.Session()でグラフの実行環境を独立させる
        style_image = tf.placeholder(tf.float32, shape=style_shape, name='style_image')
        #入力用のノードをplaceholderを使って作成する
        style_image_pre = vgg.preprocess(style_image)
        #style_imageに格納されている画像をpreprocess関数を用いて前処理を行う
        net = vgg.net(vgg_path, style_image_pre)
        #定義したCNNをnet変数に格納する
        style_pre = np.array([style_target])
        #スタイル画像を行列に変換する
        for layer in STYLE_LAYERS:
            #STYLE_LAYERS変数に格納されている文字列をlayer変数に格納していく
            features = net[layer].eval(feed_dict={style_image:style_pre})
            #layer変数の中身をひとつの要素としてnet変数にスタイル画像の行列をfeed_dictとして与え、
            #対応するNNを実行して結果をfeaturesに格納する
            features = np.reshape(features, (-1, features.shape[3]))
            #features変数を[features[3], -1]に成形する
            gram = np.matmul(features.T, features) / features.size
            #features行列の転置行列とオリジナルの行列をmatmulで乗算し、
            #その結果取得した値をfeaturesの要素数で除算し、gram変数に格納する
            style_features[layer] = gram
            #style_featuresリストのlayer(ループ毎に変わる)要素のものにgram変数の中身を格納しておく
            #なんでかはしらん

    with tf.Graph().as_default(), tf.Session() as sess:
        #tensorflowのグラフを作成し、セッションを開始して独立空間を作り出す
        X_content = tf.placeholder(tf.float32, shape=batch_shape, name="X_content")
        #入力用のノードをplaceholderを使って作成し、名前はX_content、サイズはbatch_shapeと同じようにしておく
        X_pre = vgg.preprocess(X_content)
        #X_contentの入力ノードに入力されるのは基本的に画像なので、preprocess関数を使って前処理を行い、
        #その結果をX_pre変数に格納する

        # precompute content features
        content_features = {}
        #content_features変数を定義する
        content_net = vgg.net(vgg_path, X_pre)
        #content_net変数に、X_preをVGGのCNNに通した結果を格納する
        content_features[CONTENT_LAYER] = content_net[CONTENT_LAYER]
        #content_features変数の、'relu4_2'の要素のところにCNNを通して出力された値の'relu4_2'の結果を格納
        #つまり、この変数に格納されている値もnetである（tensor）

        if slow:
            #slowモードのとき
            preds = tf.Variable(
                tf.random_normal(X_content.get_shape()) * 0.256
                #入力ノードの形で正規分布で初期化をし、値に0.256をかける
            )
            #そしてその値（行列）をpredsに格納する
            preds_pre = preds
            #preds_pre変数にpredsを格納する
        else:
            #slowモードでないとき
            preds = transform.net(X_content/255.0)
            #入力ノードの値を255で割ってマッピングのネットワークを通す
            preds_pre = vgg.preprocess(preds)
            #preds変数の中の値を画像として前処理を行う

        net = vgg.net(vgg_path, preds_pre)
        #preds_preをvggの方にあるCNNに通し、VGGの畳み込み層（合成のネットワーク）を通す

        content_size = _tensor_size(content_features[CONTENT_LAYER])*batch_size
        #_tensor_size関数を使って取得したテンソルの形（サイズ）にbatch_sizeを掛け合わせ、
        #それをcontent_sizeとして変数に格納する
        assert _tensor_size(content_features[CONTENT_LAYER]) == _tensor_size(net[CONTENT_LAYER])
        #content_featuresのCONTENT_LAYERの中身とnetのCONTENT_LAYERの中身が同じでなければAssertionError
        #同じ層を動作させているかを確認するためと思われる
        content_loss = content_weight * (2 * tf.nn.l2_loss(
            #
            net[CONTENT_LAYER] - content_features[CONTENT_LAYER]) / content_size
        )

        style_losses = []
        for style_layer in STYLE_LAYERS:
            layer = net[style_layer]
            bs, height, width, filters = map(lambda i:i.value,layer.get_shape())
            size = height * width * filters
            feats = tf.reshape(layer, (bs, height * width, filters))
            feats_T = tf.transpose(feats, perm=[0,2,1])
            grams = tf.matmul(feats_T, feats) / size
            style_gram = style_features[style_layer]
            style_losses.append(2 * tf.nn.l2_loss(grams - style_gram)/style_gram.size)

        style_loss = style_weight * functools.reduce(tf.add, style_losses) / batch_size

        # total variation denoising
        tv_y_size = _tensor_size(preds[:,1:,:,:])
        tv_x_size = _tensor_size(preds[:,:,1:,:])
        y_tv = tf.nn.l2_loss(preds[:,1:,:,:] - preds[:,:batch_shape[1]-1,:,:])
        x_tv = tf.nn.l2_loss(preds[:,:,1:,:] - preds[:,:,:batch_shape[2]-1,:])
        tv_loss = tv_weight*2*(x_tv/tv_x_size + y_tv/tv_y_size)/batch_size

        loss = content_loss + style_loss + tv_loss

        # overall loss
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        sess.run(tf.global_variables_initializer())
        import random
        uid = random.randint(1, 100)
        print("UID: %s" % uid)
        for epoch in range(epochs):
            num_examples = len(content_targets)
            iterations = 0
            while iterations * batch_size < num_examples:
                start_time = time.time()
                curr = iterations * batch_size
                step = curr + batch_size
                X_batch = np.zeros(batch_shape, dtype=np.float32)
                for j, img_p in enumerate(content_targets[curr:step]):
                   X_batch[j] = get_img(img_p, (256,256,3)).astype(np.float32)

                iterations += 1
                assert X_batch.shape[0] == batch_size

                feed_dict = {
                   X_content:X_batch
                }

                train_step.run(feed_dict=feed_dict)
                end_time = time.time()
                delta_time = end_time - start_time
                if debug:
                    print("UID: %s, batch time: %s" % (uid, delta_time))
                is_print_iter = int(iterations) % print_iterations == 0
                if slow:
                    is_print_iter = epoch % print_iterations == 0
                is_last = epoch == epochs - 1 and iterations * batch_size >= num_examples
                should_print = is_print_iter or is_last
                if should_print:
                    to_get = [style_loss, content_loss, tv_loss, loss, preds]
                    test_feed_dict = {
                       X_content:X_batch
                    }

                    tup = sess.run(to_get, feed_dict = test_feed_dict)
                    _style_loss,_content_loss,_tv_loss,_loss,_preds = tup
                    losses = (_style_loss, _content_loss, _tv_loss, _loss)
                    if slow:
                       _preds = vgg.unprocess(_preds)
                    else:
                       saver = tf.train.Saver()
                       res = saver.save(sess, save_path)
                    yield(_preds, losses, iterations, epoch)

#テンソルを格納するための引数tensor
def _tensor_size(tensor):
    #テンソルのサイズ（形？）を取得するための関数_tensor_size()
    from operator import mul
    #operatorライブラリ（演算子）から、掛け算をするための関数mul()を読み込む
    return functools.reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)
    #受け取ったテンソルの形を取得して、0番目の要素をスライスして考慮しないようにする
    #reduce関数を使って、後置for文によって取得されるdの値と1をmulメソッドでノードの数だけ掛け算し、
    #最終的なタプルの値をreturnする
