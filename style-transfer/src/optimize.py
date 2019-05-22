from __future__ import print_function
# python3系の機能からprint関数を持ってくる
import functools
# 高階関数のためのモジュールfunctoolsをimportする
import vgg, pdb, time
# vgg.pyをimportする
# デバッグ用のライブラリpdbをimportする
# 時間の概念を扱うためのライブラリtimeをimportする
import tensorflow as tf, numpy as np, os
# tensorflowをimportする
# 行列演算用のライブラリnumpyをimportする
# ファイルやディレクトリ操作用のライブラリosをimportする
import transform
# transform.pyをimportする
from utils import get_img
# utils.pyからget_img関数をimportする

STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
# NNで使うためのレイヤの名前
CONTENT_LAYER = 'relu4_2'
#
DEVICES = 'CUDA_VISIBLE_DEVICES'
# GPUの設定を読み込むための変数

# np arr, np arr
# content_targetsはコンテンツ画像セット
# style_targetはスタイル画像
# content_weightはコンテンツ画像の損失の重み
# style_weightはスタイル画像の損失の重み
# tv_weightはt
# vgg_pathはvggへのパス
# print_iterationsはチェックポイントに到達するまでの数
# batch_sizeはバッチサイズ
# save_pathはチェックポイントのファイルを保存するためのディレクトリのパス
# slowはデバッグ用の機能"slow"を利用するかを指定するための引数
# learning_rateは学習率
# debugはデバッグモードがONかどうかを指定する引数
def optimize(content_targets, style_target, content_weight, style_weight,
             tv_weight, vgg_path, epochs=2, print_iterations=1000,
             batch_size=4, save_path='saver/fns.ckpt', slow=False,
             learning_rate=1e-3, debug=False):
    # パラメータの最適化をするための関数optimizer()
    #AdaINで最適化をしている
    if slow:
        # slowが指定されていればTrue
        batch_size = 1
        # バッチサイズを１と指定する
    mod = len(content_targets) % batch_size
    # コンテンツ画像を効率良くバッチサイズに割り当てて学習できるかを見ておく
    # content_targetsとバッチサイズの剰余をmodに格納する
    if mod > 0:
        # modが0より大きければTrue
        print("Train set has been trimmed slightly..")
        # 訓練用の画像を少しだけ削減して使うことをコンソールに書き込む
        content_targets = content_targets[:-mod]
        # content_targetsに格納されている画像セットから、剰余の分を後ろから削減する

    style_features = {}
    # style_features変数を初期化する

    batch_shape = (batch_size,256,256,3)
    # バッチサイズを考慮して4次元のテンソルを作り、batch_shape変数に格納する
    style_shape = (1,) + style_target.shape
    # スタイル画像にもう一次元追加して4次元のテンソルを作り、style_shape変数に格納する

    print(style_shape)
    # スタイル画像の形をコンソールに出力する

    # precompute style features
    with tf.Graph().as_default(), tf.device('/cpu:0'), tf.Session() as sess:
        # tensorflowのグラフを作成し、デバイスをcpuに設定し、tf.Session()でグラフの実行環境を独立させる
        style_image = tf.placeholder(tf.float32, shape=style_shape, name='style_image')
        # 入力用のノードをplaceholderを使って作成する
        style_image_pre = vgg.preprocess(style_image)
        # style_imageに格納されている画像をpreprocess関数を用いて前処理を行う
        net = vgg.net(vgg_path, style_image_pre)
        # 定義したCNNをnet変数に格納する
        style_pre = np.array([style_target])
        # スタイル画像を行列に変換する
        for layer in STYLE_LAYERS:
            # STYLE_LAYERS変数に格納されている文字列をlayer変数に格納していく
            features = net[layer].eval(feed_dict={style_image:style_pre})
            # layer変数の中身をひとつの要素としてnet変数にスタイル画像の行列をfeed_dictとして与え、
            # 対応するNNを実行して結果をfeaturesに格納する
            features = np.reshape(features, (-1, features.shape[3]))
            # features変数を[features[3], -1]に成形する
            gram = np.matmul(features.T, features) / features.size
            # features行列の転置行列とオリジナルの行列をmatmulで乗算し、
            # その結果取得した値をfeaturesの要素数で除算し、gram変数に格納する
            # ここでグラム行列を取得できる
            style_features[layer] = gram
            # style_featuresリストのlayer(ループ毎に変わる)要素のものにgram変数の中身を格納しておく
            # これはスタイル元画像のグラム行列

    with tf.Graph().as_default(), tf.Session() as sess:
        # tensorflowのグラフを作成し、セッションを開始して独立空間を作り出す
        X_content = tf.placeholder(tf.float32, shape=batch_shape, name="X_content")
        # 入力用のノードをplaceholderを使って作成し、名前はX_content、サイズはbatch_shapeと同じようにしておく
        X_pre = vgg.preprocess(X_content)
        # X_contentの入力ノードに入力されるのは基本的に画像なので、preprocess関数を使って前処理を行い、
        # その結果をX_pre変数に格納する

        # precompute content features
        content_features = {}
        # content_features変数を定義する
        content_net = vgg.net(vgg_path, X_pre)
        # content_net変数に、X_preをVGGのCNNに通した結果を格納する
        content_features[CONTENT_LAYER] = content_net[CONTENT_LAYER]
        # content_features変数の、'relu4_2'の要素のところにCNNを通して出力された値の'relu4_2'の結果を格納
        # つまり、この変数に格納されている値もnetである（tensor）

        if slow:
            # slowモードのとき
            preds = tf.Variable(
                tf.random_normal(X_content.get_shape()) * 0.256
                # 入力ノードの形を正規分布で初期化をし、値に0.256をかける
            )
            # そしてその値（行列）をpredsに格納する
            preds_pre = preds
            # preds_pre変数にpredsを格納する
        else:
            # slowモードでないとき
            preds = transform.net(X_content/255.0)
            # 入力ノードの値を255で割ってマッピングのネットワークを通す
            preds_pre = vgg.preprocess(preds)
            # preds変数の中の値を画像として前処理を行う

        net = vgg.net(vgg_path, preds_pre)
        # preds_preをvggの方にあるCNNに通し、VGGの畳み込み層（合成のネットワーク）を通す

        content_size = _tensor_size(content_features[CONTENT_LAYER])*batch_size
        # _tensor_size関数を使って取得したテンソルの形（サイズ）にbatch_sizeを掛け合わせ、
        # それをcontent_sizeとして変数に格納する
        assert _tensor_size(content_features[CONTENT_LAYER]) == _tensor_size(net[CONTENT_LAYER])
        # content_featuresのCONTENT_LAYERの中身とnetのCONTENT_LAYERの中身が同じでなければAssertionError
        # スタイル用のNNとコンテンツ用のNNが同じ大きさであることを確認するためと思われる
        content_loss = content_weight * (2 * tf.nn.l2_loss(
            # l2_loss()関数は二乗誤差を求めるための関数
            net[CONTENT_LAYER] - content_features[CONTENT_LAYER]) / content_size
            # net変数の方は実際に機械がマッピングして算出した実績値を保持している
            # content_features変数の方は入力された画像をCNNに通したときの予測値を保持している
            # 上の２つの数を減算してcontent_size変数に格納されているテンソルのサイズで除算し、
            # コンテンツの損失を求めるための平均二乗和誤差をここで求めている
        )

        style_losses = []
        # スタイル損失を格納しておくための変数style_lossesを定義
        for style_layer in STYLE_LAYERS:
            # レイヤの名前が格納されている変数STYLE_LAYERSの分だけループする
            layer = net[style_layer]
            # layer変数に現在のレイヤを格納する
            bs, height, width, filters = map(lambda i:i.value,layer.get_shape())
            # 高階関数map()を使って、現在のレイヤ（ネットワーク）の形を取得する
            # bsはバッチサイズ
            # heightはレイヤの縦の大きさ
            # widthはレイヤの横の大きさ
            # filtersはレイヤの深さ？
            size = height * width * filters
            # 取得した数値のうちのbs以外を用いて特徴マップの体積を求める
            feats = tf.reshape(layer, (bs, height * width, filters))
            # 現在のレイヤの形を[バッチサイズ 特徴マップのサイズ 特徴マップの深さ（数）]に変更
            # そしてfeats変数に格納する
            feats_T = tf.transpose(feats, perm=[0,2,1])
            # feats変数の転置行列をfeats_T変数に格納する
            grams = tf.matmul(feats_T, feats) / size
            # feats_T * feats をしてfeatsのグラム行列を取得し、それをsizeで割って
            # gram変数に格納する
            # これは特徴マップのグラム行列
            style_gram = style_features[style_layer]
            # スタイル元画像のグラム行列をstyle_gram変数に格納する
            style_losses.append(2 * tf.nn.l2_loss(grams - style_gram)/style_gram.size)
            # style_losses変数に、特徴マップのグラム行列とスタイル元画像の差分をとり、
            # その平均二乗和誤差を格納する

        style_loss = style_weight * functools.reduce(tf.add, style_losses) / batch_size
        # 層ごとの平均二乗和誤差の和の値と、スタイル損失の重みの値を掛け算し、batch_sizeで割ることで
        # スタイル損失の合計損失を求めることができる

        # total variation denoising
        # deconvの時に発生するノイズの除去用（？）
        tv_y_size = _tensor_size(preds[:,1:,:,:])
        # total variationのy方向のサイズを_tensor_size()関数で取得する
        tv_x_size = _tensor_size(preds[:,:,1:,:])
        # total variationのx方向のサイズを_tensor_size()関数で取得する
        y_tv = tf.nn.l2_loss(preds[:,1:,:,:] - preds[:,:batch_shape[1]-1,:,:])
        # total variationのy方向の損失の値を計算する
        # 一つ一つのノードの二乗和誤差を、行列を用いて一気に算出し、y_tv変数に格納
        # マイナスされる方の変数が実測値でマイナスする方の値が予測値になる
        x_tv = tf.nn.l2_loss(preds[:,:,1:,:] - preds[:,:,:batch_shape[2]-1,:])
        # total vatiationのx方向の損失の値を計算する
        # yの時と同じように、二乗和誤差を求める
        tv_loss = tv_weight*2*(x_tv/tv_x_size + y_tv/tv_y_size)/batch_size
        # tvの重みと、平均二乗和誤差の値を掛け算し、それをtvの損失の値とする
        loss = content_loss + style_loss + tv_loss
        # ３つの値の合計をstyleganの損失の値としてloss変数に格納する

        # overall loss
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        # Adamを使って学習率を最適化する
        # minimize()メソッドの引数に割り当てられているlossの値が最小になるような
        # 学習率の値を探させる
        # そしてその値をtrain_step変数に格納する
        sess.run(tf.global_variables_initializer())
        # グラフ内に定義されているtensorflowのvariablesを初期化する処理を行う
        import random
        # 乱数用のライブラリrandomをimportする
        uid = random.randint(1, 100)
        # uid変数に1から100の中からランダムな値を格納する
        print("UID: %s" % uid)
        # uidの値をコンソールに表示する
        for epoch in range(epochs):
            # ハイパーパラメータのepochsの分だけループする
            num_examples = len(content_targets)
            # content_targetsには画像セットが格納されているので、
            # その数をnum_examplesに格納する
            iterations = 0
            # 1epochの中での繰り返しの数を初期化する
            while iterations * batch_size < num_examples:
                # ちゃんと1epochで全ての訓練用画像をチェックできるように、
                # iterationsとbatch_sizeの積が画像セットの中の
                # 画像の数に到達するまでループを行う
                start_time = time.time()
                # 時間の計測を行う
                curr = iterations * batch_size
                # 繰り返しの数iterationsとバッチサイズを掛け算する
                # そして得られた値が現在の正しいループの回数をcurrとする
                step = curr + batch_size
                # currにバッチサイズの値を足した答えをstepとする
                X_batch = np.zeros(batch_shape, dtype=np.float32)
                # (bs, h, w, d)の４次元のテンソルbatch_shapeの形をした
                # ゼロ行列を作り出す
                for j, img_p in enumerate(content_targets[curr:step]):
                    # 変数jには画像セットの中の何番目の画像を参照しているのかが格納される
                    # img_pには画像のパスが格納される
                    # content_targetsのリストの要素の値がcurrからstepのスライスなので、
                    # 1バッチ分の演算がこのループで順番に行われることを意味している
                    X_batch[j] = get_img(img_p, (256,256,3)).astype(np.float32)
                    # get_img()関数でimg_pのテンソル値を読み込み、
                    # サイズを(256, 256, 3)に指定（リサイズ）し、型をfloat32とする
                    # そしてその値をX_batchのj番目の要素の所に格納する

                iterations += 1
                # 繰り返し数をプラス１する
                assert X_batch.shape[0] == batch_size
                # X_batchの０つめの要素はバッチサイズなはずなので、
                # その値と、プログラム上のbatch_sizeの値が同じでなければAssertionError

                feed_dict = {
                   X_content:X_batch
                   # tensorflowの変数X_contentにX_batchの値を設定しておく
                }
                # それをフィード値としてfeed_dict変数に格納する

                train_step.run(feed_dict=feed_dict)
                # AdamOptimizerに、値をX_batchとして渡し、
                # 最適化を実行する
                end_time = time.time()
                # 時間の計測を再度行う
                delta_time = end_time - start_time
                # 処理にかかった時間を算出し、delta_timeに格納する
                if debug:
                    # debugモードが指定されていればtrue
                    print("UID: %s, batch time: %s" % (uid, delta_time))
                    # 乱数uidの値と 、処理にかかった時間をコンソールに出力する
                is_print_iter = int(iterations) % print_iterations == 0
                # 現在の繰り返し数と、チェックポイントに到達するまでの繰り返しの数の
                # 剰余が0になる時にis_print_iterにtrueを格納する（それ以外はfalse)
                if slow:
                    # slowモードが指定されているときはtrue
                    is_print_iter = epoch % print_iterations == 0
                    # エポック数とチェックポイントに到達するまでの数の剰余が0になるときtrue
                is_last = epoch == epochs - 1 and iterations * batch_size >= num_examples
                # epochs変数は実行したいエポックの数が格納されており、epoch変数には現在のエポック数が格納されている
                # 繰り返し数とバッチサイズの積が画像セットの画像の数以上になっていたり、
                # 現在のエポックが最後のエポック（epochsとepochの間には要素のズレが１つだけ生じている）だったり
                # するときはis_last変数にtrueを格納する
                should_print = is_print_iter or is_last
                # is_print_iterもしくはis_lastがtrueだったshould_print変数もtrue
                if should_print:
                    # should_print変数がtrueであるとき
                    to_get = [style_loss, content_loss, tv_loss, loss, preds]
                    # [スタイル損失, コンテンツ損失, tv損失, 全体の損失, 入力ノードの値]
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
