CycleGAN
===

cycleGAN用の実験リポジトリ


`main.py`
```
$ python main.py --help

       USAGE: main.py [flags]
flags:

main.py:
  --batch_size: Batch size
    (default: '1')
    (an integer)
  --buffer_size: Shuffle  size
    (default: '1000')
    (an integer)
  --checkpoint_dir: Path to the data folder
    (default: './checkpoints/train')
  --content_loss_weight: Value of content_loss weight
    (default: '1')
    (an integer)
  --cycle_loss_weight: Value of cycle_loss weight
    (default: '10')
    (an integer)
  --epochs: Number of epochs
    (default: '200')
    (an integer)
  --[no]vgg: Use VGG
    (default: 'true')

Try --helpfull to get a list of all flags.
```

## 再現手順

### 0. 環境構築

GPUの環境構築は頑張ってください...。
なんで動いているのかわからない状況でした...。

```
$ pip install -r requirements.txt
```

### 1. 画像データをTFRecord形式にする

画像データをそのまま使うと、処理速度が遅いので、TFRecord形式にします。

これには、`util/proper.py`を使います。

以下のように、コマンドライン引数の後に対応するデータセットが入ったディレクトリを指定してあげます。

```
$ python util/proper.py \ 
--old_image_path datasets/Domain/doraemon/_old_draemon \
--new_image_path datasets/Domain/doraemon/_new_draemon \
--test_old_image_path datasets/Domain/doraemon/_test_old_doraemon \
--test_new_image_path datasets/Domain/doraemon/_test_new_doraemon
```

この後、`new.tfrec`,`old.tfrec`,`test_new.tfrec`,`test_old.tfrec`というファイルができればOKです。

### 2. 実行する

あとは、コマンドライン引数を指定して実行します。
以下は、コンテンツロスの重みを10にした時に実行例です。

また、実験には時間がかかるため、`nohup`を用いたリモート実行がおすすめです。これで実行後セッションを切ってもバッググランド実行し続けてくれます。

```
nohup python -u main.py --content_loss_weight 10 &
```





