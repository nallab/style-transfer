Setting GPU
===

GPUを使う環境を作る

## ホスト環境の構築
手順は[Tensorflow GPU]()のサイトをみたまま。

## 起動
```
$ nvidia-docker run --rm --name コンテナ名 -v $PWD:/home/ -p 8022:22 -itd イメージ名
```

### うまくできない時
```sh
$ sudo systemctl daemon-reload
$ sudo systemctl restart docker
```

[参考](https://github.com/NVIDIA/nvidia-docker/issues/838)

## GPUがつかえているかどうかの確認

[tf.test.is_gpu_available](https://www.tensorflow.org/api_docs/python/tf/test/is_gpu_available?version=stable)を使って確認を行う。

```
$ python
>>> import tensorflow as tf
>>> tf.test.is_gpu_available()
True
```

`True`が返っていたらGPU使えます。GPUを認識していても`False`だと使えていません。

## その他
[](https://qiita.com/bellx2/items/40d01c5b169c6270427a)