Deep Photo Style Transfer
===

## Setup

1. [参照リポジトリ](https://github.com/LouieYang/deep-photo-styletransfer-tf)をcloneしてくる

```
$ git clone https://github.com/LouieYang/deep-photo-styletransfer-tf.git
```

本家とは別のリポジトリ。Tensorflowを使っており、本家よりもコードが読みやすくまたいじりやすいと判断してこちらを採用した。参照先のリポジトリのREADMEにも書いているが、全く本家と同じというわけではない。

2. Dockerで環境を作る

はじめにdockerをインストールし、nvidia-dockerもインストールする。個人的にはここが結構つまった...

あとは、Dockerfileに従い、Dockerイメージをビルドしてあげる。

```
$ docker build -t deep:v1.0 .
```

結構時間がかかるが、その後に

```
$ docker run -v YOUR_HOST_PATH:/neural-style --runtime=nvidia -i -t deep:v1.0 /bin/bash
```

あとは、Dockerコンテナ上で

```
$ pip install numpy==1.16.2
```

```
$ pip install -r /deep/requirements.txt
```

で完了。
numpyをrequirements.txtに書くとクラッシュします。
原因は調査してません。

## How to

1. 環境変数に画像のパスを登録する

```
$ source setting.sh
```

2. pair.txt に 変換する画像のペアを書く

3. 出力画像用のディレクトリをつくる

```
$ mkdir output
```

4. 処理スクリプトを走らせる

```
$ perl run.pl
```

また、pair.txtの中身を作るために`chain.pl`も作成した。
加えて、処理時間の関係上、画像サイズを小さくする方法としては
```
$ mogrify -path ./imageData -resize 50% -quality 100 *.png
```


