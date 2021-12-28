# ImageCaption-SAMPLE
Pytorch tutorialにある画像キャプション生成部分を使い勝手が良いように改良したもの

### 手順

0. Anaconda仮想環境作成
```bash
conda env create -n tutorial -f tutorial.yml
conda activate tutorial
```

1. フォルダ作成
```bash
mkdir models
```
2. 学習済みモデル、データを保存,そして解凍
https://www.dropbox.com/s/ne0ixz5d58ccbbz/pretrained_model.zip?dl=0


3. 実行
```bash
python inference.py --image='png/example.png'
```

### 引用元
https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning