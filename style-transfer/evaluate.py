from __future__ import print_function
# python3系の機能をfutureモジュールからimportする。この場合はprint関数
import sys
# pythonのインタプリタで使われている変数やインタプリタの動作に関連する関数が定義されているモジュールをimportする
sys.path.insert(0, 'src')
# 自分と同じ深さにあるディレクトリをライブラリ読みこみ対象フォルダとして認識させる
import transform, numpy as np, vgg, pdb, os
# 自作のpythonファイルtransform, vggを読みこむ。そして行列演算用のライブラリnumpyと、デバッグ用のライブラリpdbと、ファイルやディレクトリ操作用のライブラリosをimportする
import scipy.misc
# 計算機能を拡張できるscipyの中のmisc(種々雑多なライブラリ)をimportする
import tensorflow as tf
# DL用のライブラリtensorflowをimportする
from utils import save_img, get_img, exists, list_files
# 自作のpythonファイルutilsから4つの関数をimportする
from argparse import ArgumentParser
# コマンドライン引数をとるためのライブラリArgumentParserをimportする
from collections import defaultdict
# 存在しないキー値にアクセスされたときに自動的にデフォルト値を設定してくれるcollectionsをimportする
import time
# 時間関連の情報や関数をまとめたライブラリtimeをimportする
import json
# json形式のファイルを扱うためのライブラリjsonをimportする
import subprocess
# pythonのプログラムから他のアプリを起動したり、実行結果を得るモジュールのsubprocessをimportする
import numpy
# numpyをimportする
'''
from moviepy.video.io.VideoFileClip import VideoFileClip
import moviepy.video.io.ffmpeg_writer as ffmpeg_writer
'''

BATCH_SIZE = 4
# batch size(super parameter)
DEVICE = '/gpu:0'
# 所持している1つ目のGPUを利用するためにDEVICE定数に文字列を格納する

'''
def ffwd_video(path_in, path_out, checkpoint_dir, device_t='/gpu:0', batch_size=4):
    video_clip = VideoFileClip(path_in, audio=False)
    video_writer = ffmpeg_writer.FFMPEG_VideoWriter(path_out, video_clip.size, video_clip.fps, codec="libx264",
                                                    preset="medium", bitrate="2000k",
                                                    audiofile=path_in, threads=None,
                                                    ffmpeg_params=None)

    g = tf.Graph()
    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True
    with g.as_default(), g.device(device_t), \
            tf.Session(config=soft_config) as sess:
        batch_shape = (batch_size, video_clip.size[1], video_clip.size[0], 3)
        img_placeholder = tf.placeholder(tf.float32, shape=batch_shape,
                                         name='img_placeholder')

        preds = transform.net(img_placeholder)
        saver = tf.train.Saver()
        if os.path.isdir(checkpoint_dir):
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise Exception("No checkpoint found...")
        else:
            saver.restore(sess, checkpoint_dir)

        X = np.zeros(batch_shape, dtype=np.float32)

        def style_and_write(count):
            for i in range(count, batch_size):
                X[i] = X[count - 1]  # Use last frame to fill X
            _preds = sess.run(preds, feed_dict={img_placeholder: X})
            for i in range(0, count):
                video_writer.write_frame(np.clip(_preds[i], 0, 255).astype(np.uint8))

        frame_count = 0  # The frame count that written to X
        for frame in video_clip.iter_frames():
            X[frame_count] = frame
            frame_count += 1
            if frame_count == batch_size:
                style_and_write(frame_count)
                frame_count = 0

        if frame_count != 0:
            style_and_write(frame_count)

        video_writer.close()

'''
# get img_shape

# data_inは変化させたい画像の入ってるディレクトリかファイルを指定する
# paths_outは変化後の画像を格納しておくためのディレクトリかファイルを指定する
# checkpoint_dirは、訓練中のチェックポイントを読み込むためのディレクトリを指定する
def ffwd(data_in, paths_out, checkpoint_dir, device_t='/gpu:0', batch_size=4):
    #
    assert len(paths_out) > 0
    # paths_outの中身がなければAssertionErrorをはく
    is_paths = type(data_in[0]) == str
    # data_inの0番目の要素が文字列型かどうかを判別する
    if is_paths:
        # is_pathsがTrueの時は引数としてディレクトリが正しく認識されていることになる
        assert len(data_in) == len(paths_out)
        # 入力される画像data_inの要素数と出力される画像paths_outの要素数が一致するかを確かめる
        img_shape = get_img(data_in[0]).shape
        # 先頭の入力画像の形をimg_shape変数に格納する

    else:
        # is_pathsの中身がディレクトリでないとき
        assert data_in.size[0] == len(paths_out)
        #data_inに格納されているデータの量と
        img_shape = X[0].shape
        #

    g = tf.Graph()
    # tensorflowのグラフを作成し、変数gに代入する
    batch_size = min(len(paths_out), batch_size)
    # paths_outの要素数と設定されたバッチサイズのうちで小さい方をbatch_size変数にバッチサイズとして代入する
    curr_num = 0
    # なんかわからんけどcurr_num変数に0を代入する
    soft_config = tf.ConfigProto(allow_soft_placement=True)
    # 設定したGPUが存在しない場合にデバイスを自動設定するための引数allow_soft_placementをTrueにし、
    # tensorflowのConfigProtoで読みこませた設定をsoft_configに格納しておく
    soft_config.gpu_options.allow_growth = True
    # GPUのメモリを最初にすべて食い尽くさないようにする設定のallow_growthをTrueにする
    with g.as_default(), g.device(device_t), \
            tf.Session(config=soft_config) as sess:
            # gのグラフについての処理を開始する
            # g.as_default()で、gに設定されたグラフをデフォルトグラフとして設定する
            # g.device()で、この処理の中で使うデフォルトのデバイスを設定する
            # tf.Session()で、グラフが実行することをセッションとして独立させ管理することができる
        batch_shape = (batch_size,) + img_shape
        # この足し算で(batch_size, img_shape)の形のタプルを作り出し、batch_shape変数に格納する（img_shapeは2次元)
        img_placeholder = tf.placeholder(tf.float32, shape=batch_shape,
                                         name='img_placeholder')
        # float32型のplaceholderを作り、名前をimg_placeholderとする
        preds = transform.net(img_placeholder)
        # img_placeholderに保存されている値を画風変換ネットワークに通す
        saver = tf.train.Saver()
        # チェックポイントに現在の状態を保存するためのインスタンスSaverを
        # saver変数に格納する
        if os.path.isdir(checkpoint_dir):
            # チェックポイントのパスがディレクトリだった場合True
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            # チェックポイントのディレクトリに、保存されたデータがあるかどうかを確認し、
            # 存在していればckpt変数にTrueを格納する
            if ckpt and ckpt.model_checkpoint_path:
                # ckptがTrueであるのとなおかつ、
                # チェックポイントのディレクトリが指定されていればTrue
                saver.restore(sess, ckpt.model_checkpoint_path)
                # セッションをチェックポイントのディレクトリに格納する
            else:
                # 上記の2つの条件が当てはまらなかったとき
                raise Exception("No checkpoint found...")
                # 例外を吐く（no checkpoints）
        else:
            # チェックポイントのディレクトリじゃなかった場合
            saver.restore(sess, checkpoint_dir)
            # セッションをcheckpoint_dirのディレクトリに保存する

        num_iters = int(len(paths_out)/batch_size)
        # paths_out（出力する値）とバッチサイズの商をnum_itersに格納する
        for i in range(num_iters):
            # num_iters（繰り返し数）分だけ
            pos = i * batch_size
            # バッチサイズと現在の繰り返し数の積をpos変数に格納する
            curr_batch_out = paths_out[pos:pos+batch_size]
            # 現在のループの数で処理したい要素の分だけpaths_outから値を取り出して
            # curr_batch_out変数に格納する
            if is_paths:
                # data_inがディレクトリを指定していた場合
                curr_batch_in = data_in[pos:pos+batch_size]
                # data_inのディレクトリの中から、現在のループ分だけデータを取り出し、
                # curr_batch_in変数に格納する
                X = np.zeros(batch_shape, dtype=np.float32)
                # batch_shapeの形をしているゼロ行列を作り出し、Xに格納する
                for j, path_in in enumerate(curr_batch_in):
                    # jにはループの番号、path_inにはファイル名が格納される
                    img = get_img(path_in)
                    # path_inで指定されている画像の行列をimg変数に格納する
                    assert img.shape == img_shape, \
                        'Images have different dimensions. ' +  \
                        'Resize images or use --allow-different-dimenszions.'
                    # 事前に取得してあったimg_shape変数の中身と、img変数のshapeが
                    # 一致しなければAssertionError
                    X[j] = img
                    # Xのj番目の要素に画像の行列を格納する
            else:
                # data_inがディレクトリを指定していなかった場合
                X = data_in[pos:pos+batch_size]
                # 現在のループの数だけdata_inからデータを取り出し、
                # 変数Xに格納する

            _preds = sess.run(preds, feed_dict={img_placeholder:X})
            # すでに定義されているノード変数img_placeholderにXの値をいれて
            # セッションを実行し、predsの値を_predsに格納する
            for j, path_out in enumerate(curr_batch_out):
                # jには現在の繰り返し数、path_outには現在のループで処理したい要素の分の画像の出力先のパスを格納する
                save_img(path_out, _preds[j])
                #paths_outに書かれているパスの位置に_preds[j]に格納されている画像データを保存する

        remaining_in = data_in[num_iters*batch_size:]
        # 現在のバッチで読み込まれている画像のパスをremaining_in変数に格納する
        remaining_out = paths_out[num_iters*batch_size:]
        # 現在のバッチでの画像出力先のパスをremaining_out変数に格納する
    if len(remaining_in) > 0:
        # ちゃんとremaining_inの中身が入っていた場合True
        ffwd(remaining_in, remaining_out, checkpoint_dir,
            device_t=device_t, batch_size=1)
        # ffwd関数を再帰呼び出し、
        # data_in引数にremaining_inの値を格納する
        # data_out引数にremaining_outの値を格納する

# in_pathは入力される画像ファイルかディレクトリのパス
# out_pathは加工を加えた画像ファイルを出力する先
# checkpoint_dirはチェックポイントを読み込むためのディレクトリ
def ffwd_to_img(in_path, out_path, checkpoint_dir, device='/cpu:0'):
    #
    paths_in, paths_out = [in_path], [out_path]
    ffwd(paths_in, paths_out, checkpoint_dir, batch_size=1, device_t=device)

def ffwd_different_dimensions(in_path, out_path, checkpoint_dir,
            device_t=DEVICE, batch_size=4):
    in_path_of_shape = defaultdict(list)
    out_path_of_shape = defaultdict(list)
    for i in range(len(in_path)):
        in_image = in_path[i]
        out_image = out_path[i]
        shape = "%dx%dx%d" % get_img(in_image).shape
        in_path_of_shape[shape].append(in_image)
        out_path_of_shape[shape].append(out_image)
    for shape in in_path_of_shape:
        print('Processing images of shape %s' % shape)
        ffwd(in_path_of_shape[shape], out_path_of_shape[shape],
            checkpoint_dir, device_t, batch_size)

def build_parser():
    # オプション解析用の関数build_parser
    parser = ArgumentParser()
    # ArgumentParserのインスタンスを生成
    parser.add_argument('--checkpoint', type=str,
                        dest='checkpoint_dir',
                        help='dir or .ckpt file to load checkpoint from',
                        metavar='CHECKPOINT', required=True)
    # checkpointのディレクトリを指定するための引数を設定する

    parser.add_argument('--in-path', type=str,
                        dest='in_path',help='dir or file to transform',
                        metavar='IN_PATH', required=True)
    # 入力元のファイルもしくはディレクトリを指定するための引数を設定する

    help_out = 'destination (dir or file) of transformed file or files'
    # エラーメッセージを変数help_outに格納しておく
    parser.add_argument('--out-path', type=str,
                        dest='out_path', help=help_out, metavar='OUT_PATH',
                        required=True)
    # 加工を加えた後の画像を格納しておくためのディレクトリもしくはファイルを指定するための引数を設定する

    parser.add_argument('--device', type=str,
                        dest='device',help='device to perform compute on',
                        metavar='DEVICE', default=DEVICE)
    # CPUで処理をするかGPUで処理をするかを指定するための引数を設定する

    parser.add_argument('--batch-size', type=int,
                        dest='batch_size',help='batch size for feedforwarding',
                        metavar='BATCH_SIZE', default=BATCH_SIZE)
    # バッチサイズを指定するための引数を設定する

    parser.add_argument('--allow-different-dimensions', action='store_true',
                        dest='allow_different_dimensions',
                        help='allow different image dimensions')
    # サイズの異なる画像を指定するのを許可するかどうかを指定するための引数を設定する

    return parser
    # parserインスタンスをreturnする

# optsはチェックしてほしいオプションを格納するための引数
def check_opts(opts):
    # オプションとして入力されたものが適切かどうかを判別するための関数
    exists(opts.checkpoint_dir, 'Checkpoint not found!')
    # checkpoint_dir引数がちゃんと実際に存在する場所を示しているかを確認する
    exists(opts.in_path, 'In path not found!')
    # in_path引数がちゃんと実際に存在する場所を示しているかを確認する
    if os.path.isdir(opts.out_path):
        # out_pathがディレクトリなのかファイルなのかを判別し、ディレクトリならTrue
        exists(opts.out_path, 'out dir not found!')
        # out_path引数がちゃんと実際に存在する場所を示しているかを確認する
        assert opts.batch_size > 0
        # batch_sizeが0より大きいかを確認する

def main():
    # 実際に実行される関数
    parser = build_parser()
    # 引数を解析してparserに格納する
    opts = parser.parse_args()
    # 引数として読み込まれたものをoptsに格納する
    check_opts(opts)
    # 引数が不適切なものではないかどうかを確認する

    if not os.path.isdir(opts.in_path):
        # in_path引数がファイルだった場合
        if os.path.exists(opts.out_path) and os.path.isdir(opts.out_path):
            # out_path引数がちゃんと存在していて、ディレクトリであったらTrue
            out_path = \
                    os.path.join(opts.out_path,os.path.basename(opts.in_path))
            # 加工を加えた画像の出力先ディレクトリに入力されたファイル名をつけ足してout_path変数に文字列型として格納する
        else:
            # out_path引数がファイルである場合
            out_path = opts.out_path
            # out_path変数にout_path引数に格納されているディレクトリのパスを格納する

        ffwd_to_img(opts.in_path, out_path, opts.checkpoint_dir,
                    device=opts.device)
        #
    else:
        # in_path引数がファイルではなくディレクトリだった場合
        files = list_files(opts.in_path)
        # in_path引数に内包されているファイルのリストをfilesに格納する
        full_in = [os.path.join(opts.in_path,x) for x in files]
        # 入力データの格納されているディレクトリのパスと、内包されていたファイル名を結合して絶対パスを作り出す
        full_out = [os.path.join(opts.out_path,x) for x in files]
        # 出力先のディレクトリのパスと、内包されていたファイル名を結合して変化後の画像用の名前を作り出す
        if opts.allow_different_dimensions:
            # サイズの違う画像を使うことが許可されている場合
            ffwd_different_dimensions(full_in, full_out, opts.checkpoint_dir,
                    device_t=opts.device, batch_size=opts.batch_size)
            #
        else:
            # サイズの違う画像を使うことが許可されていない場合
            ffwd(full_in, full_out, opts.checkpoint_dir, device_t=opts.device,
                    batch_size=opts.batch_size)
            #

if __name__ == '__main__':
    # evaluate.pyがメインで実行される場合True
    main()
    # main関数を実行する
