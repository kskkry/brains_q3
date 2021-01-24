### brains_q3(5位)
以下コンテストを進める際に実行したこと

#### 特徴量の扱いについて
* 与えられたデータセットはsmiles文字列のみ
* smilesから構造式に変換できるサイト(http://www.cheminfo.org/flavor/malaria/Utilities/SMILES_generator___checker.html)があったので、それらを利用して構造を確認した。
* 小さな分子から見たことないくらい大きな分子が含まれていることが確認できた。(意外だったのは大きな分子ほどtarget(IC50)の値が大きくなるものが多かった。)
* 序盤はrdkitで拡張したデータのみを学習に使っていた。途中からrdkit+mordredで拡張したデータを学習に使うつもりだったが、制限時間オーバーになったので終盤はより多くのデータが得られるmordredのみを用いた（上位者の話によると、両方使っても制限時間内での実行は可能だったらしい）
* mordredはimportの際にエラーにより上手く機能しなかったので、コンペ中盤以降はディレクトリをダウンロードして提出ディレクトリ中に置いたところ解決した。
* mordredは2000個の特徴量が作れるが、約200個くらいは全てNaNの列だった。さらに全体的にNaNを含む特徴量が多いので、mordredだけで拡張したデータを扱うのは適切ではなかったかもしれない。(NNを使うときはrdkitにより得たデータの方がCVは良かった)
* 序盤はQ2で作成した特徴量(水溶性置換基の有無や分子の対称性や芳香族性などを反映したもの)をそのまま用いた。当然ながらあまり効かなかったので、途中からこれらの特徴量は除去した。
* lightgbmで学習して重要度の高い特徴量を取り出し加減乗除した特徴量をいくつか加えたところ、CVとリーダーボードの乖離しており過学習が見られたので後半は行わなかった。
* MACCSやmorganフィンガープリントにより特徴量が生成でき、それの類似度の計算ができるが、CVへの貢献が小さく計算時間が長いことから後半は使わなかった。
* これらの理由から、終盤ではmordredにより得たデータから重要度の高い200個くらいの特徴量を学習に使った
* 

#### 使用したモデル
* 序盤からlightgbmをずっと使っていた。過学習を避けたかったので序盤はnum_leaves=5あたりで小さめにしてlr=0.05, epoch=3000で固定していた(他のパラメータはデフォルト)
* 序盤からboosting_type='rmse'にしており、'dart'も試したところスコアが悪化したので後半は使わなかった。
* 他の学習モデルにはrandom forestやxgboost、sklearnに含まれるニューラルネットワークや線形モデルを使用した。
* 上のモデルのスコアはどれもlightgbmのスコアより悪かったため、加重平均によるアンサンブルに使用したがlightgbm単一の方が良かった
* 種々のモデルを用いて線形モデルまたはlightgbmによるスタッキングも試みたが、過学習によりLBがかなり悪化したため終盤はlightgbmのみを使った。
* seedの値を変えてアンサンブルも試したが、スコアの推移とモデルの容量との兼ね合いから、終盤は単一のseedで8fold lightgbmのモデルを提出していた
* 
![終盤での進め方](./asset/process.png)

#### その他
* lightgbmやxgboostを使用する際に欠損値を埋めない方がCVは良かった


####  反省