# assign_datanumber.py  
GT-1st-bounce.py 等で作成したnpzフォルダ内にあるシミュレーションフォルダを指定して,  
データセットを作成する。ナンバリングをしてPytorchで扱いやすい形式にする。
```
実行例

npzes_list = ['livingroom-train1','livingroom-train2']
dataset_name = 'livingroom-train'

hdr_tools/  
　├ assign_datanumber.py  
　├ npz/
　│　└ livingroom-train1/
　│　　　└ 00000/
　│　　　└ 00001/
　│　└ livingroom-train2/
　│　　　└ 00000/
　│　　　└ 00001/
　├ datasets/
　│　└ livingroom-train/ ←統合されたデータセットが作成される
　│　　　└ 00000/
　│　　　└ 00001/
　│　　　└ 00002/
　│　　　└ 00003/
　└ etc
```

  
# dataset_statistics.py  
1st-bounceの割合等を調べるためのスクリプト
  
  
# GT-1st-bounce-clean.py  
GT-1st-bounce.pyのグラフ出力等を省略したバージョン
  
```
実行例

dataname = 'livingroom-train1'

folder_path = os.path.join(r'~/bunnykiller_tools/scene-results', dataname)
npz_folder_path = os.path.join('npz', dataname)

~/bunnykiller_tools/  
　├ scene-results/
　│　└ livingroom-train1/
　│　　　└ 00000/
　│　　　└ 00001/
　└ etc

hdr_tools/  
　├ GT-1st-bounce-clean.py 
　├ npz/
　│　└ livingroom-train1/ ←bunnykillerで出力されたhdrからGTを求めたり光を畳み込む。dataname内全てに適用される。
　│　　　└ 00000/
　│　　　　　└ ground-truth.npz
　│　　　　　└ A0.npz
　│　　　　　└ A1.npz
　│　　　　　└ A2.npz
　│　　　└ 00001/
　└ etc
```

# GT-1st-bounce.py  
距離の真値(GT)をインパルスの1st-bounceから求める。bunnykillerのhdrデータを読み込み、その過程でxytデータを作成する。
  

# xyt_npdata.py
bunnykillerのhdrデータを読み込み、xytデータを作成するだけのスクリプト。  
なお1st-bounceのみのxytデータを作ることができる。
  
```
実行例

dataname = 'livingroom-train1'

folder_path = os.path.join(r'~/bunnykiller_tools/scene-results', dataname)
npz_folder_path = os.path.join('xyt-npz', dataname)

~/bunnykiller_tools/  
　├ scene-results/
　│　└ livingroom-train1/
　│　　　└ 00000/
　│　　　└ 00001/
　└ etc

hdr_tools/  
　├ GT-1st-bounce-clean.py 
　├ xyt-npz/
　│　└ livingroom-train1/ ←bunnykillerで出力されたhdrからGTを求めたり光を畳み込む。dataname内全てに適用される。
　│　　　└ 00000/
　│　　　　　└ xyt.npz
　│　　　└ 00001/
　│　　　　　└ xyt.npz
　└ etc
```


# etcfile
パルス変調やAMCWの信号作成等のスクリプト


---
  
# Note
ディレクトリ構造が異なると動かない場合があるので例を参考に実行環境を作成してください。
