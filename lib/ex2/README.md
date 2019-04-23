# 実験用スクリプト2

## 背景

実験の段階で解像度やアスペクト比がバラバラであり、これらが変換に何らかの影響を及ぼしているのではないかと考えた。そのため、Mac Book Pro 13のフルスクリーンモードでアニメ動画をスクショし、そのデータをHD画質、FHD画質にカットリサイズするスクリプトを書いた。これにより、解像度・アスペクト比をある程度まで自在にいじることができる。

## 使い方

### 1.変換したいスクリーンショット画像をあるディレクトリに保存する。
### 2.後は適切なパラメータを設定すれば動く。

```
$ sh imageProcessing.sh -hd old_original HD
SUCESS!!:old_original/ 2018-01-26 23.47.41.png
SUCESS!!:old_original/ 2018-01-26 23.43.30.png
SUCESS!!:old_original/ 2018-01-26 23.44.07.png
SUCESS!!:old_original/ 2018-01-26 23.43.26.png
SUCESS!!:old_original/ 2018-01-26 23.45.54.png
SUCESS!!:old_original/ 2018-01-26 23.43.46.png
...
```
