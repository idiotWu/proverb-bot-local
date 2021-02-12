# 名言ボット（ローカルバージョン）
入力文章に対し，最大感情を推定し名言を返します．

## 動作環境
`Python >= 3.8`

## 準備
### 必要なライブラリのインストール
```bash
pip install -r requirements.txt
```

### 学習済みベクトルのダウンロード
1. 学習済みベクトルを [WikiEntVec](https://github.com/singletongue/WikiEntVec/releases) からダウンロードします．今回は 300 次元の単語ベクトル `jawiki.word_vectors.300d.txt.bz2` を利用しました．
2. ダウンロードしたファイルを解凍し，ファイル名を `word_vectors.txt` と変更し，ソースコードのフォルダに移動します．

## プログラムの起動
### 感情分析器としての起動
```bash
python analyzer.py
```

### チャットボットとしての起動
```bash
python main.py
```
