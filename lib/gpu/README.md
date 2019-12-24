Setting GPU
===

GPUを使う環境を作る

## 起動
```
$ nvidia-docker run --rm --name コンテナ名 -v $PWD:/home/ -p 8022:22 -itd イメージ名
```

## GPUがつかえているかどうかの確認

[tf.test.is_gpu_available](https://www.tensorflow.org/api_docs/python/tf/test/is_gpu_available?version=stable)を使って確認を行う。

```
$ python
>>> import tensorflow as tf
>>> tf.test.is_gpu_available()
True
```

`True`が返っていたらGPU使えます。GPUを認識していても`False`だと使えていません。