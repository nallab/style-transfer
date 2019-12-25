画像データセット作りメモ
===

動画データから動画を構成するフレームを取り出してくる方法のメモ。忘れちゃうので。

## 

### ffmpeg を install
メインPC が Mac であるため、Homebrew を使って ffmpeg を install します
```
$ brew install ffmpeg
```

### 音声を動画データを分離
音声データは必要ないので、音声と動画を分離します。
また、`.mov`から`mp4`に形式を変換します。

```
$ ffmpeg -i new.mov -an new.mp4
```

`-i`がinput、`-an`がaudio disable。

動画が長いと結構時間がかかる。
ファイル形式については詳しくないが、`mov`から`mp4`の変換でサイズが10分の1ほどになった(かつ音声もなし)。

### 動画の比率の変更
長方形は使いづらいので、正方形に比率を変更する。

```
$ ffmpeg -i new.mp4 -vf crop=956:956 new_resize.mp4
$ ffmpeg -i old.mp4 -vf crop=956:956 old_resize.mp4
```

crop の数値は、キャプチャした画像の縦幅の最大値。
つまり、**横幅の外側部分は部分は切り捨てられる**。

### 動画を画像に分解
動画を画像データに分解していく。
```
$ mkdir new_data
$ mkdir old_data
$ ffmpeg -i new_resize.mp4 -vcodec mjpeg -r 0.2 ./new_data/image_%04d.jpg
$ ffmpeg -i old_resize.mp4 -vcodec mjpeg -r 0.2 ./old_data/image_%04d.jpg
```
`ffmpeg`の`-r`を変更すれば、レートがかわる。
今回は 0.2 にしているため、5秒に1回のペースでキャプチャしている...はず。

### 画像サイズを変更
分割した画像のサイズを一気に変更。
学習時はメモリの関係上大きな画像サイズを用いるのは難しいため、リサイズする。
```
$ cd new_data
$ mogrify -resize 256x256 -quality 100 image_*.jpg  
$ cd ../old_data
$ mogrify -resize 256x256 -quality 100 image_*.jpg  
```

`-path`オプションがあり、指定できるっぽかったが、うまく行かず...。
対象ディレクトリまで移動してしまうことにしてます。

### ひどまずはこんなところ
ひとまずはこんなところ。
あとは、データを実行環境にもっていく。
```
$ scp -r ./new_data USER@IPADRESS:
$ scp -r ./old_data USER@IPADRESS:
```





