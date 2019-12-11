CycleGAN
===

cycleGAN用の実験リポジトリ

プログラムの都合上、画像データはPATHのTFRecord形式にする必要がある

```
$ python util/proper.py  --old_image_path /root/old_data --new_image_path /root/new_data
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
