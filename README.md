# assign_datanumber.py  
GT-1st-bounce.py 等で作成したnpzフォルダ内にあるシミュレーションフォルダを指定して,  
データセットを作成する。ナンバリングをしてPytorchで扱いやすい形式にする。
  
  
# dataset_statistics.py  
1st-bounceの割合等を調べるためのスクリプト
  
  
# GT-1st-bounce-clean.py  
GT-1st-bounce.pyのグラフ出力等を省略したバージョン
  

# GT-1st-bounce.py  
距離の真値(GT)をインパルスの1st-bounceから求める。bunnykillerのhdrデータを読み込み、その過程でxytデータを作成する。
  
# xyt_npdata.py
bunnykillerのhdrデータを読み込み、xytデータを作成するだけのスクリプト。  
なお1st-bounceのみのxytデータを作ることができる。

# etcfile
パルス変調やAMCWの信号作成等のスクリプト


---
  
# Note
注意点などがあれば書く
   
  
# License
ライセンスを明示する
 
"hoge" is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).
 
社内向けなら社外秘であることを明示してる
 
"hoge" is Confidential.
