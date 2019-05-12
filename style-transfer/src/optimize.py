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
#
CONTENT_LAYER = 'relu4_2'
#
DEVICES = 'CUDA_VISIBLE_DEVICES'
#

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
        #読みこんだVGGファイルと画像を使って、重なったCNNを通す
        style_pre = np.array([style_target])
        #スタイル画像を行列に変換する
        for layer in STYLE_LAYERS:
            #STYLE_LAYERS変数に格納されている文字列をlayer変数に格納していく
            features = net[layer].eval(feed_dict={style_image:style_pre})
            #
            features = np.reshape(features, (-1, features.shape[3]))
            gram = np.matmul(features.T, features) / features.size
            style_features[layer] = gram

    with tf.Graph().as_default(), tf.Session() as sess:
        X_content = tf.placeholder(tf.float32, shape=batch_shape, name="X_content")
        X_pre = vgg.preprocess(X_content)

        # precompute content features
        content_features = {}
        content_net = vgg.net(vgg_path, X_pre)
        content_features[CONTENT_LAYER] = content_net[CONTENT_LAYER]

        if slow:
            preds = tf.Variable(
                tf.random_normal(X_content.get_shape()) * 0.256
            )
            preds_pre = preds
        else:
            preds = transform.net(X_content/255.0)
            preds_pre = vgg.preprocess(preds)

        net = vgg.net(vgg_path, preds_pre)

        content_size = _tensor_size(content_features[CONTENT_LAYER])*batch_size
        assert _tensor_size(content_features[CONTENT_LAYER]) == _tensor_size(net[CONTENT_LAYER])
        content_loss = content_weight * (2 * tf.nn.l2_loss(
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

def _tensor_size(tensor):
    from operator import mul
    return functools.reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)
