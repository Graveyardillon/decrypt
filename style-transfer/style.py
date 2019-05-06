from __future__ import print_function
#python3系の機能（print関数）をfutureモジュールからimportする
import sys, os, pdb
#pythonのインタプリタで使われている変数や関数を操作できるモジュールsysをimportする
#ファイルやディレクトリ操作ができるモジュールosをimportする
#デバッグ用のライブラリpdbをimportする
sys.path.insert(0, 'src')
#srcの中身を直接参照できるようにする
import numpy as np, scipy.misc
#行列演算用のライブラリnumpyと、種々雑多なライブラリscipy.miscをimportする
from optimize import optimize
#optimize.pyからoptimize関数をimportする
from argparse import ArgumentParser
#引数解析用のライブラリArgumentParserをimportする
from utils import save_img, get_img, exists, list_files
#自作のpythonファイルutilsから4つの関数をimportする
import evaluate
#evaluate.pyの関数をimportする

CONTENT_WEIGHT = 7.5e0
#2種類ある損失のうちの、コンテンツの損失の方を算出するために使われる重み
STYLE_WEIGHT = 1e2
#2種類ある損失のうちの、スタイルの損失の方を算出するために使われる重み
TV_WEIGHT = 2e2
#

LEARNING_RATE = 1e-3
#学習率
NUM_EPOCHS = 2
#訓練するときのエポック数
CHECKPOINT_DIR = 'checkpoints'
#チェックポイントのディレクトリのパス
CHECKPOINT_ITERATIONS = 2000
#チェックポイントまでの繰り返しの数
VGG_PATH = 'data/imagenet-vgg-verydeep-19.mat'
#訓練済みのネットワークvgg19のパス
TRAIN_PATH = 'data/train2014'
#訓練用のデータの保存されているパス
BATCH_SIZE = 4
#バッチサイズ
DEVICE = '/gpu:0'
#訓練に使うプロセッサ
FRAC_GPU = 1
#

def build_parser():
    #オプションを解析するための関数build_parser()
    parser = ArgumentParser()
    #ArgumentParserのインスタンスを生成し、parserに格納する
    parser.add_argument('--checkpoint-dir', type=str,
                        dest='checkpoint_dir', help='dir to save checkpoint in',
                        metavar='CHECKPOINT_DIR', required=True)
    #チェックポイントのディレクトリのパス
    #metavarはヘルプが表示されたときにヘルプに表示される名前

    parser.add_argument('--style', type=str,
                        dest='style', help='style image path',
                        metavar='STYLE', required=True)
    #画風変換のために使う画像のパス

    parser.add_argument('--train-path', type=str,
                        dest='train_path', help='path to training images folder',
                        metavar='TRAIN_PATH', default=TRAIN_PATH)
    #訓練用の画像の保存されているディレクトリのパス

    parser.add_argument('--test', type=str,
                        dest='test', help='test image path',
                        metavar='TEST', default=False)
    #チェックポイントに到達毎に実行されるテスト用の画像のパス

    parser.add_argument('--test-dir', type=str,
                        dest='test_dir', help='test image save dir',
                        metavar='TEST_DIR', default=False)
    #チェックポイントに到達毎に実行されるテスト用画像が格納されているディレクトリのパス

    parser.add_argument('--slow', dest='slow', action='store_true',
                        help='gatys\' approach (for debugging, not supported)',
                        default=False)
    #デバッグをしたいときに使うオプションで、Gatyの手法を用いてピクセル単位で損失関数を最適化することができる

    parser.add_argument('--epochs', type=int,
                        dest='epochs', help='num epochs',
                        metavar='EPOCHS', default=NUM_EPOCHS)
    #訓練時のエポック数

    parser.add_argument('--batch-size', type=int,
                        dest='batch_size', help='batch size',
                        metavar='BATCH_SIZE', default=BATCH_SIZE)
    #バッチサイズ

    parser.add_argument('--checkpoint-iterations', type=int,
                        dest='checkpoint_iterations', help='checkpoint frequency',
                        metavar='CHECKPOINT_ITERATIONS',
                        default=CHECKPOINT_ITERATIONS)
    #チェックポイントに到達するまでの繰り返し数

    parser.add_argument('--vgg-path', type=str,
                        dest='vgg_path',
                        help='path to VGG19 network (default %(default)s)',
                        metavar='VGG_PATH', default=VGG_PATH)
    #学習済みネットワークのvgg19のパス（vgg16を試すこともできるらしい）

    parser.add_argument('--content-weight', type=float,
                        dest='content_weight',
                        help='content weight (default %(default)s)',
                        metavar='CONTENT_WEIGHT', default=CONTENT_WEIGHT)
    #コンテンツの損失を算出するために使われる重み

    parser.add_argument('--style-weight', type=float,
                        dest='style_weight',
                        help='style weight (default %(default)s)',
                        metavar='STYLE_WEIGHT', default=STYLE_WEIGHT)
    #スタイルの損失を算出するために使われる重み

    parser.add_argument('--tv-weight', type=float,
                        dest='tv_weight',
                        help='total variation regularization weight (default %(default)s)',
                        metavar='TV_WEIGHT', default=TV_WEIGHT)
    #

    parser.add_argument('--learning-rate', type=float,
                        dest='learning_rate',
                        help='learning rate (default %(default)s)',
                        metavar='LEARNING_RATE', default=LEARNING_RATE)
    #学習率

    return parser
    #parserインスタンスをreturnする

#optsはチェックしてほしいオプションを格納するための引数
def check_opts(opts):
    #オプションの中身をチェックする関数check_opts()
    exists(opts.checkpoint_dir, "checkpoint dir not found!")
    #checkpointのディレクトリがちゃんと存在しているかを確かめる
    exists(opts.style, "style path not found!")
    #スタイル画像のパスがちゃんと存在しているかを確かめる
    exists(opts.train_path, "train path not found!")
    #訓練用の画像の保存されているディレクトリのパスがちゃんと存在しているかを確かめる
    if opts.test or opts.test_dir:
        #testかtest_dirが指定されていればTrue
        exists(opts.test, "test img not found!")
        #test用画像がちゃんと存在しているかを確かめる
        exists(opts.test_dir, "test directory not found!")
        #test用画像の格納されたディレクトリがちゃんと存在しているかを確かめる
    exists(opts.vgg_path, "vgg network data not found!")
    #vggのネットワークのパスがちゃんとした場所を示しているかを確認する
    assert opts.epochs > 0
    #エポック数が0より大きくなければAssertionError
    assert opts.batch_size > 0
    #バッチサイズが0より大きくなければAssertionError
    assert opts.checkpoint_iterations > 0
    #チェックポイントに到達するまでの繰り返し数が0より大きくなければAssertionError
    assert os.path.exists(opts.vgg_path)
    #vgg_pathがちゃんと存在していなければAssertionError
    assert opts.content_weight >= 0
    #コンテンツの損失用の重みの値が0より小さければAssertionnError
    assert opts.style_weight >= 0
    #スタイルの損失用の重みの値が0より小さければAssertionError
    assert opts.tv_weight >= 0
    #
    assert opts.learning_rate >= 0
    #学習率が0より小さければAssertionError

#img_dirは画像の格納されているディレクトリ
def _get_files(img_dir):
    #img_dirで指定されたディレクトリの中身のファイルを取得する関数_get_files()
    files = list_files(img_dir)
    #img_dirのディレクトリの中にあるファイルの名前をfiles変数に格納する
    return [os.path.join(img_dir,x) for x in files]
    #img_dirに格納されているディレクトリのパスとファイル名を結合してパスを作り、returnする


def main():
    #実際に実行される関数
    parser = build_parser()
    #引数を解析してparserに格納する
    options = parser.parse_args()
    #引数として読み込まれたものをoptionsに格納する
    check_opts(options)
    #引数が不適切なものではないかどうかを確認する

    style_target = get_img(options.style)
    #スタイル画像をget_img関数を用いてプログラム内に読み込む
    if not options.slow:
        #slowを指定されていなければこちら側に分岐する
        content_targets = _get_files(options.train_path)
        #_get_files関数を使って訓練用の画像のパスを読み込み、content_targets変数に代入する
    elif options.test:
        #引数としてtestが指定されていた場合
        content_targets = [options.test]
        #テスト用画像をcontent_targetsに代入する

    kwargs = {
        #辞書型のオブジェクトkwargsに値を入れていく
        "slow":options.slow,
        #slowが指定されていれば"slow"として格納する
        "epochs":options.epochs,
        #エポック数を"epochs"として格納する
        "print_iterations":options.checkpoint_iterations,
        #チェックポイントに到達するまでの数を"print_iterations"として格納する
        "batch_size":options.batch_size,
        #バッチサイズを"batch_size"として格納する
        "save_path":os.path.join(options.checkpoint_dir,'fns.ckpt'),
        #チェックポイントとして使うファイルのパスをcheckpoint_dirと結合し、"save_path"として格納する
        "learning_rate":options.learning_rate
        #学習率を"learninng_rate"として格納する
    }


    if options.slow:
        #slowが指定されていた場合
        if options.epochs < 10:
            #エポック数が10よりも小さければTrue
            kwargs['epochs'] = 1000
            #エポック数を1000に上書きする
        if options.learning_rate < 1:
            #学習率が1よりも小さい場合
            kwargs['learning_rate'] = 1e1
            #学習率を1x10^1に上書きする

    args = [
        #argsリストに値を代入していく
        content_targets,
        #コンテンツ画像
        style_target,
        #スタイル画像
        options.content_weight,
        #コンテンツの損失の重み
        options.style_weight,
        #スタイルの損失の重み
        options.tv_weight,
        #
        options.vgg_path
        #vggのある場所へのパス
    ]

    for preds, losses, i, epoch in optimize(*args, **kwargs):
        #
        style_loss, content_loss, tv_loss, loss = losses

        print('Epoch %d, Iteration: %d, Loss: %s' % (epoch, i, loss))
        to_print = (style_loss, content_loss, tv_loss)
        print('style: %s, content:%s, tv: %s' % to_print)
        if options.test:
            assert options.test_dir != False
            preds_path = '%s/%s_%s.png' % (options.test_dir,epoch,i)
            if not options.slow:
                ckpt_dir = os.path.dirname(options.checkpoint_dir)
                evaluate.ffwd_to_img(options.test,preds_path,
                                     options.checkpoint_dir)
            else:
                save_img(preds_path, img)
    ckpt_dir = options.checkpoint_dir
    cmd_text = 'python evaluate.py --checkpoint %s ...' % ckpt_dir
    print("Training complete. For evaluation:\n    `%s`" % cmd_text)

if __name__ == '__main__':
    main()
