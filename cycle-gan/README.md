CycleGAN
===

cycleGAN用の実験リポジトリ

### Memo
- ディレクトリ`datasets`はクラウドで管理。
- 必要に応じて`scp`で送る
- `lib/proper.py`は基本的に引数なしで利用できるように調整済み

## How to

プログラムの都合上、画像データはPATHのTFRecord形式にする必要がある

```
$ python util/proper.py 
 --old_image_path /root/old_data 
 --new_image_path /root/new_data
```

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
  --epochs: Number of epochs
    (default: '40')
    (an integer)
```

[Unicode Error](https://qiita.com/cclef/items/dc8692d8b5a0c8c7812c)
