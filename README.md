### brains_q3(5位)
[第5回fujifilm brains コンテスト](https://fujifilmdatasciencechallnge.mystrikingly.com/)を進める際に実行したことのまとめです。

<br>

#### 特徴量の扱いについて
* 与えられたデータセットはsmiles文字列のみ
* smilesから構造式に変換できる[サイト](http://www.cheminfo.org/flavor/malaria/Utilities/SMILES_generator___checker.html) があったので、それらを利用して構造を確認した。
* 小さな分子からかなり大きな分子が含まれていることが確認できた。(意外だったのは大きな分子ほどtarget(IC50)の値が大きくなるものが多かった。)
* 正直なところ、構造式を確認してもtargetの値に寄与する特徴は自分ではわからなかった。特定の官能基や部分構造の寄与というより、むしろ分子の形や部分構造同士の相互作用などが重要になる？と感じた。
* 上の通りに考えた結果、新しい特徴量を自分で考えるのは難しいと感じたため中盤以降はモデル部分に注力した。
* 序盤はrdkitで拡張したデータのみを学習に使っていた。途中からrdkit+mordredで拡張したデータを学習に使うつもりだったが、制限時間オーバーになったので終盤はより多くのデータが得られるmordredのみを用いた（上位の方の話によると、両方使っても制限時間内での実行は可能だったらしい）
* mordredはimportの際にエラーにより上手く機能しなかったので、コンペ中盤以降はディレクトリをダウンロードして提出ディレクトリ中に置いたところ解決した。
* mordredは2000個の特徴量が作れるが、約200個くらいは全てNaNの列だった。さらに全体的にNaNを含む特徴量が多いので、mordredだけで拡張したデータを扱うのは適切ではなかったかもしれない。(NNを使うときはrdkitにより得たデータの方がCVは良かった)
* 序盤はQ2で作成した特徴量(水溶性置換基の有無や分子の対称性や芳香族性などを反映したもの)をそのまま用いた。当然ながらあまり効かなかったので、途中からこれらの特徴量は除去した。
* lightgbmで学習して重要度の高い特徴量を取り出し、それをもとに加減乗除した特徴量をいくつか加えた。(xfeat？なるライブラリが便利らしい？)
* MACCSやmorganフィンガープリントにより特徴量が生成でき、それの類似度の計算ができるが、CVへの貢献が小さく計算時間が長いことから後半は使わなかった。
* これらの理由から、終盤ではmordredにより得たデータとその一部を加減乗除したデータから重要度の高い200個くらいの特徴量を学習に使った
* 

<br><br>

#### 使用したモデル
* 序盤からlightgbmをずっと使っていた。過学習を避けたかったので序盤はnum_leaves=5あたりで小さめにしてlr=0.05, epoch=3000で固定していた(他のパラメータはデフォルト)
* 序盤からboosting_type='rmse'にしていた。'dart'も試したところスコアが悪化したので後半は使わなかった。
* 他の学習モデルにはrandom forestやxgboost、sklearnに含まれるニューラルネットワークや線形モデルを使用した。
* 上のモデルのスコアはどれもlightgbmのスコアより悪かったため、加重平均によるアンサンブルに使用したがlightgbm単一の方が良かった
* 種々のモデルを用いて線形モデルまたはlightgbmによるスタッキングも試みたが、過学習によりLBがかなり悪化したため終盤はlightgbmのみを使った。
* seedの値を変えてアンサンブルも試したが、スコアの推移とモデルの容量との兼ね合いから、終盤は単一のseedで8fold lightgbmのモデルを提出していた
* 中盤以降、スコアを確認しながらnum_leavesを少しずつ大きくした。最終的にはnum_leaves=12で探索していた。
* 筆者がパラメータについてよく理解していないこともあり終盤でのlightgbmのパラメータ探索では、以上のことに加えて過学習を防ぐ目的でmax_binを大きくしただけのモデルを提出していた。その他のパラメータは変更するとCVが悪化したころからデフォルトのままにしておいた。またlearning_rate=0.01にしてepoch=10000くらいにすると、モデルの容量が制約を大幅に超えてしまった。そのため制約を超えない程度にlearning_rateを変更して試したところ、learning_rate=0.015に落ち着いた。結果としてはこれらのパラメータの変更だけで終盤のスコアは大きく伸びた。
* deepchemも使いたかったのでsrcディレクトリに置くなどして色々試してみた。ローカルでは上手く使えたが提出したところエラーが止まらなかったのでdeepchemは泣く泣くあきらめた。

<br>

![終盤での進め方](./asset/process.png)

<br><br>

#### その他
* lightgbmやxgboostを使用する際に欠損値を埋めない方がCVとtrack上のスコアは良かった
* Kaggle本にも書いてある調和平均や幾何平均を試したところtrack上のスコアが0.01程度改善する場合が多々あったので試してみる価値はある。
* サンプルコードではtargetの値にnp.log1pを取ったものを学習に使っていたが、np.log1pを取ったものを正規分布状へ変形させることやnp.sqrt()やnp.sqrt(np.log1p())などにより変形させることを試してみた。しかし悉くCVが悪化したので以降使わなかった。

<br><br>


####  感想と反省
* 制限時間の兼ね合いからmordredのみのデータを用いていたが、1位の方はrdkit+mordred+Fingerprintによる特徴量が使えていたので自分ももっといろいろ試してみるべきだった。
* Q2含めてだがtanimoto係数による類似度の測定を全ての分子同士で計算させていなかった。これも勝手に自分で計算時間を見積もって判断してしまったので大きな反省点
* やはり色々試してみるのが一番良いというのが改めて感じた。
* 序盤はすぐ最高スコアを更新できるだろうの精神で、良いスコアを取った特徴量作成用のファイルやモデルをすぐ捨ててしまった。結果的に序盤の自分のベストスコアが超えれず、かなり苦労したのでファイルの管理は適切に行いたい（自戒）
* 訓練データ量が小さく、またCVとLBの乖離が小さかった点で様々な手法を試すことができたコンペでした。非常に勉強になりました。運営の皆様ありがとうございました。

