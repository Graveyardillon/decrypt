import scipy.misc
#計算機能を拡張するscipyの中からmisc（種々雑多なライブラリ）をimportする
import numpy as np
#行列演算のライブラリnumpyをimoprtする
import os
#ファイルやディレクトリ操作用のライブラリosをimportする
import sys
#pythonのインタプリタで使われている変数や関数を操作できるモジュールsysをimportする

#out_pathは画像の保存先ディレクトリ
#imgは画像用の引数
def save_img(out_path, img):
#画像を保存するための関数
    img = np.clip(img, 0, 255).astype(np.uint8)
    #画像が格納されているであろう変数imgの中の最小値を0,最大値を255にしておく
    scipy.misc.imsave(out_path, img)
    #画像を保存する

#style_pathはスケールしたい画像のパス
#style_scaleは画像の大きさを変更するための目安のスケール
def scale_img(style_path, style_scale):
    #scale_imgは画像をスケールするための関数
    scale = float(style_scale)
    #style_scaleをfloat型でキャストしてscale変数に格納する
    o0, o1, o2 = scipy.misc.imread(style_path, mode='RGB').shape
    #imread関数で画像のソースを読み込み、modeを3x8bitのRGBに設定しておく
    #そしてそれぞれの要素をo0, o1, o2に格納する
    scale = float(style_scale)
    #よくわからんけどstyle_scaleをもう一回floatでキャストする
    new_shape = (int(o0 * scale), int(o1 * scale), o2)
    #画像の縦の大きさの値を保持しているo0, 横の大きさの値を保持しているo1,
    #それぞれをスケールした値をintにキャストしてnew_shape変数に格納する
    #o2は深さなので、スケールする必要はない
    style_target = _get_img(style_path, img_size=new_shape)
    #style_pathを_get_img()で読みこむときに、img_sizeの中身をnew_shapeに設定することでリサイズする
    return style_target
    #リサイズした画像style_targetをreturnする

#srcは読みこむ画像のパス
#img_sizeは基本的にFalseで指定されているが、必要があればサイズを指定することができる
def get_img(src, img_size=False):
#プログラム内に画像を読み込むための関数
   img = scipy.misc.imread(src, mode='RGB') # misc.imresize(, (256, 256, 3))
   #imread関数で画像のソースを読み込み、modeを3x8bitのRGBに設定しておく
   if not (len(img.shape) == 3 and img.shape[2] == 3):
       #要素が高さ、幅、深さの３つもしくは深さのスカラ値が３でなければTrue（深さはピクセル深度のこと）
       img = np.dstack((img,img,img))
       #imgの深さを３にするために、img３つを深さ方向に結合する。（RGBの３つ）
   if img_size != False:
       #引数img_sizeがFalseではないとき
       img = scipy.misc.imresize(img, img_size)
       #画像imgをimg_sizeの大きさにリサイズする
   return img
   #imgをreturnする

#pは存在しているかを調べたいパスが示されている文字列
#msgは表示したいエラーメッセージ
def exists(p, msg):
    #ファイルもしくはディレクトリが存在しなければエラーメッセージを吐く関数exists
    assert os.path.exists(p), msg
    #pが指し示すパスが存在していなければエラーメッセージとしてmsgを表示する

#in_pathはディレクトリを示す文字列で、内包しているファイルを調べたいもの
def list_files(in_path):
    #ディレクトリが内包しているファイルを走査する関数list_files
    files = []
    #空のリストfilesを作成する
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        #現在のディレクトリdirpath,現在のディレクトリが内包するディレクトリdirnames,
        #現在のディレクトリが内包するファイルfilenamesをos.walkでディレクトリを走査して取得する
        files.extend(filenames)
        #filesリストにfilenamesを追加していく
        break
        #ここでbreakすることで内包しているディレクトリの中は走査せず、　現在のディレクトリのみ走査できる

    return files
    #内包するファイル名の文字列を含んだリストfilesをreturnする
