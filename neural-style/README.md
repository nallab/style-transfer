Neural-Style
===

Gatysらの手法を用いて、画風変換を行うための環境及び実行リポジトリ。

## Setup

1. [参照リポジトリ](https://github.com/anishathalye/neural-style)をクローンしてくる。

```
$ git clone https://github.com/anishathalye/neural-style
```

本家とは別のリポジトリ。Tensorflowを使っており、本家よりもコードが読みやすくまたいじりやすいと判断してこちらを採用した。参照先のリポジトリのREADMEにも書いているが、全く本家と同じというわけではない。

2. Dockerで環境をつくる

はじめにdockerをインストールし、nvidia-dockerもインストールする。個人的にはここが結構つまった...

あとは、`Dockerfile`に従い、Dockerイメージをビルドしてあげる。
```
docker build -t neural-style:v1.0 .
```

結構時間がかかるが、その後に
```
docker run -v YOUR_HOST_PATH:/neural-style --runtime=nvidia -i -t neural-style:v1.0 /bin/bash
```

あとは、Dockerコンテナ上で
```
pip install -r /neural-style/requirements.txt
```

で準備OK。
