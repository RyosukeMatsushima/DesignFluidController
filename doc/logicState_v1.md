# 活動報告書
<div style="text-align: right;">
松島亮輔
</div>

## 1.導入
　物体の運動は位相空間上の状態点と呼ばれる粒子の運動に写像できる。状態点を一般化座標$\boldsymbol x$と一般化運動量$\boldsymbol p$を用いて

$$
\boldsymbol X = (\boldsymbol x \ \boldsymbol p) \tag1
$$

と定義する。そして、状態点の位置の時間変化は以下のようになる。

$$
\frac{d \boldsymbol X}{dt} = \boldsymbol f(\boldsymbol X, \boldsymbol u_i) \tag2
$$

ここで、$\boldsymbol u_i$は制御入力であり、制御則は制御対象が出力可能な制御入力の集合$\left\{{\boldsymbol u_1, \boldsymbol u_2, ..., \boldsymbol u_n}\right\}$から選択する。式(2)は位相空間での状態点の運動を表す流れ場と考えられる。そして、制御入力を選択することはどの流れに乗るか選択する事である。

```
# つぶやき
自由について考えるとそれは何かとても大きなもので、自由であるということはどんな事でも出来るように思えてしまう。しかし、制御則に与えられている自由は出力可能な入力の中から選択する事のみである。人間の体を制御対象と考えれば、私に与えられている自由も実は自身の体で出力可能な制御入力から実際に出力する制御入力を選択する事のみであると考えられる。人間に与えられた自由とは制約の中から選択する事と言える。それでも自然界の自由度の高さと選択するタイミングもその選択肢に入ることから自由がとても複雑なものに感じられる。東西線でふと周りを見渡すと、画面にのめり込み自ら選択肢という自由を画面から与えられるもののみに狭めている人々の姿が写った。辟易した。そして、私は自らの自由を行使するために浦安で電車を降り、地図も見ずに西船橋までの長い道のりを歩いて帰った。後から考えると当たり前のことのように聞こえるが、私には出力可能な制御入力の選択肢がとても多くあり、調子に乗って自由を乱用すると日が暮れる。しかし、自らの人生に評価場を定め、それを最大化または最小化することだけを考え、行動を決めることも傲慢に思える。
```

## 2.状態点の存在確立分布とその時間変化
　制御則が制御入力の集合$\left\{{\boldsymbol u_1, \boldsymbol u_2, ..., \boldsymbol u_n}\right\}$から無作為に入力$\boldsymbol u_i$を選択する場合を考える。ここで、制御入力$\boldsymbol u_i$を選択する確立$P(\boldsymbol u_i) = 1/n$で全て等しいとする。この条件下で状態点の分布(濃度)を$W(\boldsymbol X, t)$として状態点の移動を表す式を求める。まず、式(2)を離散化し以下の式を得る。

$$
\Delta \boldsymbol X(\boldsymbol X, \boldsymbol u_i) \approx \boldsymbol f(\boldsymbol X, \boldsymbol u) \Delta t \tag3
$$

ここで、位相空間における位置$\boldsymbol X$、時間$t + \Delta t$における状態点の存在確率分布は

$$
W(\boldsymbol X, t + \Delta t) = \sum^i P(u_i) \boldsymbol W(\boldsymbol X - \Delta X(\boldsymbol X, \boldsymbol u_i), t) \tag4
$$

となる。ここで、$\Delta X << 1$とし、以下の近似を用いた。

$$
\Delta X(\boldsymbol X, \boldsymbol u_i) \approx \Delta X(\boldsymbol X - \Delta X, \boldsymbol u_i) \tag5
$$

ここで、確立分布$W(\boldsymbol X, t)$について、テイラー展開を行う。

$$
\boldsymbol W(\boldsymbol X, t + \Delta t) =
\boldsymbol W(\boldsymbol X, t)　+ \frac{\partial \boldsymbol W}{\partial t}(\boldsymbol X, t) \Delta t　+ \frac{1}{2} \frac{\partial^2 \boldsymbol W}{\partial^2 t}(\boldsymbol X, t) \Delta t^2 + ...
$$

$$
\boldsymbol W(\boldsymbol X - \Delta \boldsymbol X, t) =
\boldsymbol W(\boldsymbol X, t)　- \frac{\partial \boldsymbol W}{\partial t}(\boldsymbol X, t) \Delta \boldsymbol X　+ \frac{1}{2} \frac{\partial^2 \boldsymbol W}{\partial^2 t}(\boldsymbol X, t) \Delta \boldsymbol X^2 + ...\tag6
$$

これらを式(4)へ代入すると以下のような移流拡散方程式が得られる。

$$
\frac{\partial \boldsymbol W}{\partial t} =　- \frac{\partial \boldsymbol W}{\partial t} \sum^i P(u_i) \frac{\Delta \boldsymbol X(\boldsymbol X, \boldsymbol u_i)}{\Delta t}　+ \frac{1}{2} \frac{\partial^2 \boldsymbol W}{\partial^2 t} \sum^i P(u_i) \frac{\Delta \boldsymbol X^2(\boldsymbol X, \boldsymbol u_i)}{\Delta t}
$$

$$
\frac{\partial \boldsymbol W}{\partial t} =　- \boldsymbol c \frac{\partial \boldsymbol W}{\partial t}　+ \boldsymbol D \frac{\partial^2 \boldsymbol W}{\partial^2 t}
$$

$$
\boldsymbol c = \sum^i P(u_i) \frac{\Delta \boldsymbol X(\boldsymbol X, \boldsymbol u_i)}{\Delta t}
$$

$$
\boldsymbol D = \frac{1}{2}\sum^i P(u_i) \frac{\Delta \boldsymbol X^2(\boldsymbol X, \boldsymbol u_i)}{\Delta t} \tag7
$$

//TODO: ランダムウォークと移流拡散方程式の参考文献

## 3.提案する制御則
### 発想
　日常生活の中で匂いを発生するものをある場所に置くと、その匂い物質は周りの空気の流れにより拡散し、空間に分布する。一般的に匂い物質の強度は匂いを発生させているものに近いほど強くなる。多くの動物はこの匂い物質の分布を嗅覚で感じ、匂いの発生源まで到達することができる。位相空間上に匂いのような目標状態までの距離を表す分布を作り、それをもとに制御入力を選択することが出来るのではないかと考え、本研究を始めた。

### 提案手法１　（上記の発想を忠実に再現した手法）
位相空間上のある初期位置$\boldsymbol X_0$における状態点の存在確立を$P(\boldsymbol X_0, t_0) = 1$とすると時刻$t$における状態点の存在確立分布$P(\boldsymbol X_0, t_0|\boldsymbol X, t)$は初期値$\boldsymbol X$から流れ場(2)に基づき、移流拡散シミュレーションをすることで求めることが出来る。ここで、全ての時刻$t$において$W(\boldsymbol X_0) = 1$として、位相空間上の位置$X_0$に匂いの発生源がある状態を再現する。そして、流れ場に基づき無限時間移流拡散シミュレーションを回した結果として得られる各位置における状態点の濃度は

$$
W(\boldsymbol X) = \int_{t_0}^{\infty} P(\boldsymbol X, t | \boldsymbol X_0, t_0) dt \tag{3.1}
$$

と表すことが出来る。また、上の式は時間を並行にずらし、

$$
W(\boldsymbol X) = \int_{t_0}^{\infty} P(\boldsymbol X, t_0 | \boldsymbol X_0, 2t_0 - t) dt \tag{3.2}
$$

と書き換えられる。ここで、濃度$W(\boldsymbol X)$は時間的概念を失ってしまう。実生活においても匂い物質を観測した場合、その物質が匂いの発生源から放出された時間を特定することは不可能である。

#### 時間対称性
時間を反転させた位相空間における流れ場を考え、制御則が制御入力を選択する場合に指針にする濃度分布を求める。以下、反転させた時間を$t' = t_0 - t$と定義する。力学系の時間対称性より、

//TODO: 時間対称性参考文献

$$
\frac{d \boldsymbol X}{dt'} = -\boldsymbol f(\boldsymbol X, \boldsymbol u_i) \tag{3.3}
$$

となる。そして、目標状態$\boldsymbol X_c$での濃度を$W(\boldsymbol X) = 1$と固定し、式(3.1)と同様に時間反転させた流れ場により無限時間拡散させると以下の式を得る。

$$
W(\boldsymbol X) = \int_{t_0}^{\infty} P(\boldsymbol X, t' | \boldsymbol X_c, t_0) dt' \tag{3.4}
$$

$$
W(\boldsymbol X) = \int_{t_0}^{\infty} P(\boldsymbol X, t_0 | \boldsymbol X_c, 2t_0 - t') dt' \tag{3.5}
$$

ここで時間$t$を用いて変形すると、

$$
W(\boldsymbol X) = \int_{0}^{\infty} P(\boldsymbol X, t_0 | \boldsymbol X_c, t_0 + t) dt \tag{3.6}
$$

と解釈することができる。

$$
u^*(\boldsymbol X) = \argmax_{u_i}(\nabla W(\boldsymbol X) \cdot f(\boldsymbol X, \boldsymbol u_i)) \tag 8
$$

## 数値計算による状態点分布のシミュレーション
