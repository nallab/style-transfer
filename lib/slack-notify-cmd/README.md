Slack通知用のコマンド
===

プログラムを動かしているときに、途中で止まってしまったことに気づかずに時間を消費してしまうことが多かったため、Slackに通知するコマンドを作った。

## How to

1. `.env`ファイルを作成し、中にWebhookのurlと、slackの該当チャンネル名をかいて上げる

```
WEBHOOK=http://-
CHANNEL=nallab
```

2. ビルドする

```
$ go build
```

3. 実行する

```
./slack
```

