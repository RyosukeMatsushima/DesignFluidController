## 活動報告書
<div style="text-align: right;">
2020/5/13
</div>

### 導入
　制御とは選択の連続である。エージェント（以降、制御入力を決定する部分をエージェント、制御対象をロボットと呼ぶ）は限られた制御入力から最適と思われる入力を選択する。この世界の捉え方として、エージェントが選択を迫られる度に、選択可能な入力の数だけ異なる状態のロボットの確率的な存在の数が増えていく。このような思想をもとに新たな制御則（要調査）を模索している。

### 位相空間における表現
　物体の運動は位相空間上の粒子の運動に写像できる。ここではロボットの状態を表す位相空間上の粒子を以下のように表す。

$$
\xi = (x_i, w) \tag1
$$

ここで$x_i$は位相空間内の微小な開集合でロボットの状態、$w$は粒子の重みをそれぞれ表す。エージェントが無作為に入力を選択する世界で、ロボットを初期状態$x_0$に置いた場合の粒子は図.1のように表される。ここでは、エージェントは毎時$\Delta t$ごとに選択を迫られている。そして、エージェントに$n$個の選択肢が与えられている場合粒子の重みは、

$$
w(t + \Delta t) = w(t)\frac { 1 } { n } \tag 2
$$

となる。初期状態$x_0$からこのような粒子を十分な量と時間放出した状態を考えると,
ある微小状態空間$x_i$内にあるもっとも重い粒子が、初期状態$x_0$から最短経路で遷移してきた粒子となる。このように、式(1)の性質を用いると二つの状態量間の入力制限を考慮した最短経路を求めることができる。

### 連続時間での表現
　力学系におけるロボットの運動は、一般化座標$\boldsymbol p$と一般化運動量$\boldsymbol q$、制御入力$\boldsymbol u$を用いて表される。また、それぞれの時間変化は以下のように表される。

$$
\dot {\boldsymbol p} = \boldsymbol f(\boldsymbol p, \boldsymbol q, \boldsymbol u) \\
\dot {\boldsymbol q} = \boldsymbol g(\boldsymbol p, \boldsymbol q) \tag 3
$$

そして、位相空間上の粒子が流れる場の速度は

$$
\boldsymbol v(u_i) = \left[\begin{array}{c}
            \dot {\boldsymbol p} (u_i)  \\
            \dot {\boldsymbol q} \\
        \end{array}\right] \quad　\tag 4
$$

と表される。ここで、エージェントが入力を選択する行為は、流れ場$\boldsymbol v(u_i)$を選択する行為と同義である。以上を踏まえて、図.1で示した位相空間内での粒子の拡散は以下の式で表すことができる。

$$
\frac{\partial \rho}{\partial t} = \frac {1} {n}\sum_{i=1}^n -\boldsymbol v(u_i) \nabla \rho \tag 5
$$

ここで

$$
\rho = \sum_{\xi \in \mathrm{d}V} w \tag 6
$$

とし以降、$\rho$を密度と呼ぶ。また、式(5)はリウヴィルの定理を拡張した形になている。

### 仮説
　力学系の時間対称性を利用し、時間を反転させた流れ場$-\boldsymbol v(u_i)$の中で、ロボットの目標状態量$\boldsymbol x_c$に一定密度$\rho_0$を常に設定し、十分に

$$
\frac{\partial \rho}{\partial t} = \frac {1} {n}\sum_{i=1}^n -(-\boldsymbol v(u_i)) \nabla \rho \tag 7
$$

ただし、$\rho(\boldsymbol x_c)=\rho_0$で
