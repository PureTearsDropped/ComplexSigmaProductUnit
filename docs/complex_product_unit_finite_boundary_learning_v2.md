# 複素 Product Unit と有限境界学習系
## 4チャネル完全展開版

## 1. 概要

本設計では、複素 Product Unit を以下の考え方で構成する。

1. 複素入力を実部・虚部に分け、さらに各成分を正負チャネルへ分解する。
2. 複素重みも実部・虚部、および正負チャネルで表現する。
3. Product Unit は複素対数を用いて
   \[
   P_{ik}
   =
   \exp\left(
   \sum_j W_{ijk}\Log_0(x_{ijk})
   \right)
   \]
   と定義する。
4. \(\Log_0\) は通常の複素対数をそのまま使うのではなく、ゼロや最小正状態 \(\varepsilon\) を含む独自の拡張演算として定義する。
5. 非ゼロ値・非ゼロ勾配には最小絶対値 \(\varepsilon\) を設ける。
6. overflow / underflow は符号や状態を保持しながら境界値へ写す。
7. forward の飽和演算と backward の微分規則は必要に応じて分離し、代理勾配として明示する。

---

# 2. 複素入力の4チャネル表現

複素入力を

\[
x_{ijk}
=
x^{\mathrm{Re}}_{ijk}
+
i x^{\mathrm{Im}}_{ijk}
\]

とする。

実部・虚部をそれぞれ正負分解する。

\[
\begin{aligned}
x^{\mathrm{Re}+}_{ijk}
&=
\max(x^{\mathrm{Re}}_{ijk},0),
\\
x^{\mathrm{Re}-}_{ijk}
&=
\max(-x^{\mathrm{Re}}_{ijk},0),
\\
x^{\mathrm{Im}+}_{ijk}
&=
\max(x^{\mathrm{Im}}_{ijk},0),
\\
x^{\mathrm{Im}-}_{ijk}
&=
\max(-x^{\mathrm{Im}}_{ijk},0).
\end{aligned}
\]

したがって、

\[
\boxed{
x_{ijk}
=
\left(
x^{\mathrm{Re}+}_{ijk}
-
x^{\mathrm{Re}-}_{ijk}
\right)
+
i
\left(
x^{\mathrm{Im}+}_{ijk}
-
x^{\mathrm{Im}-}_{ijk}
\right)
}
\]

となる。

以下、簡潔化のため

\[
a_{ijk}
=
x^{\mathrm{Re}+}_{ijk}
-
x^{\mathrm{Re}-}_{ijk},
\]

\[
b_{ijk}
=
x^{\mathrm{Im}+}_{ijk}
-
x^{\mathrm{Im}-}_{ijk}
\]

と置く。

すると

\[
\boxed{
x_{ijk}=a_{ijk}+ib_{ijk}
}
\]

である。

---

# 3. ゼロと \(\varepsilon\) は別状態

本設計では、

\[
\boxed{0\neq+\varepsilon}
\]

であり、

\[
\boxed{0\neq-\varepsilon}
\]

である。

\(0\) は「\(\varepsilon\) よりさらに小さい数」としてではなく、独立した特殊状態として扱う。

したがって、実数値の基本表現空間は概念的に

\[
\boxed{
[-\mathrm{MAX},-\varepsilon]
\cup
\{0\}
\cup
[\varepsilon,\mathrm{MAX}]
}
\]

とする。

非ゼロ値について

\[
0<|x|<\varepsilon
\]

なら、

\[
x>0
\Rightarrow
x\mapsto+\varepsilon,
\]

\[
x<0
\Rightarrow
x\mapsto-\varepsilon.
\]

一方、

\[
x=0
\Rightarrow
x\mapsto0.
\]

---

# 4. 独自対数 \(\log_0\)

通常の実対数では

\[
\log 0
\]

は未定義である。

しかし本設計では、演算規則として

\[
\boxed{
\log_0(0):=0
}
\]

と定義する。

さらに最小正状態 \(+\varepsilon\) に対して、

\[
\boxed{
\log_0(\varepsilon):=-\infty
}
\]

と定義する。

ここでの \(\varepsilon\) は単なる任意の有限小数ではなく、本数値体系における「最小正状態」を表す。

したがって、

\[
\boxed{
0
\quad\text{と}\quad
\varepsilon
}
\]

は対数空間でも全く異なる意味を持つ。

\[
\log_0(0)=0,
\qquad
\log_0(\varepsilon)=-\infty.
\]

実装上 `Inf` を実際の浮動小数値として使用しない場合は、数学上の \(-\infty\) を

\[
\mathrm{LOG\_MIN}
\]

のような専用状態またはタグ付き境界値として表現できる。

---

# 5. 複素数の偏角

複素数

\[
z=a+ib
\]

について、偏角は

\[
\arg z
\]

である。

正の実軸上にある条件は

\[
\boxed{
b=0,\qquad a>0
}
\]

であり、このとき

\[
\arg z=0.
\]

例えば \(+\varepsilon\) は

\[
+\varepsilon=\varepsilon+0i
\]

なので、

\[
\arg(+\varepsilon)=0
\]

である。

これは \(\varepsilon\) が特別だからではなく、

\[
b=0,\qquad a>0
\]

だからである。

主値偏角を用いる場合、

\[
\operatorname{Arg}(a+ib)
=
\operatorname{atan2}(b,a)
\]

として、

\[
\operatorname{Arg}z\in(-\pi,\pi]
\]

などの範囲を採用できる。

代表例は

\[
\begin{array}{c|c}
z & \operatorname{Arg}(z)\\
\hline
a+0i,\ a>0 & 0\\
0+bi,\ b>0 & \pi/2\\
a+0i,\ a<0 & \pi\ \text{または branch により }-\pi\\
0+bi,\ b<0 & -\pi/2
\end{array}
\]

である。

---

# 6. \(\arg(0)\) の独自定義

通常、

\[
\arg(0)
\]

は未定義である。

しかし、本設計で

\[
\Log_0(0)=0
\]

を成立させるため、偏角についても

\[
\boxed{
\operatorname{Arg}_0(0,0):=0
}
\]

と定義できる。

したがって、

\[
\boxed{
\operatorname{Arg}_0(a,b)
=
\begin{cases}
0,
&
a=0,\ b=0,
\\[4pt]
\operatorname{atan2}(b,a),
&
\text{otherwise}
\end{cases}
}
\]

とする。

---

# 7. 複素拡張対数 \(\Log_0\)

複素数

\[
z=a+ib
\]

について、

\[
r=|z|
=
\sqrt{a^2+b^2}
\]

とする。

本設計では

\[
\boxed{
\Log_0(z)
=
L_0(r)
+
i\operatorname{Arg}_0(a,b)
}
\]

と定義する。

ここで \(L_0\) は大きさに対する独自対数演算であり、少なくとも

\[
L_0(0)=0,
\]

\[
L_0(\varepsilon)=-\infty
\]

を満たす。

\(z=0\) のとき、

\[
r=0,
\qquad
\operatorname{Arg}_0(0,0)=0
\]

なので、

\[
\boxed{
\Log_0(0)=0+i0=0
}
\]

となる。

---

# 8. 4チャネルから絶対値を直接計算

入力の4チャネルから

\[
a_{ijk}
=
x^{\mathrm{Re}+}_{ijk}
-
x^{\mathrm{Re}-}_{ijk},
\]

\[
b_{ijk}
=
x^{\mathrm{Im}+}_{ijk}
-
x^{\mathrm{Im}-}_{ijk}
\]

なので、

\[
r_{ijk}
=
|x_{ijk}|
=
\sqrt{a_{ijk}^2+b_{ijk}^2}.
\]

したがって完全に展開すると、

\[
\boxed{
r_{ijk}
=
\sqrt{
\left(
x^{\mathrm{Re}+}_{ijk}
-
x^{\mathrm{Re}-}_{ijk}
\right)^2
+
\left(
x^{\mathrm{Im}+}_{ijk}
-
x^{\mathrm{Im}-}_{ijk}
\right)^2
}
}
\]

となる。

---

# 9. 4チャネルから偏角を直接計算

偏角は

\[
\theta_{ijk}
=
\operatorname{Arg}_0(a_{ijk},b_{ijk})
\]

なので、

\[
\boxed{
\theta_{ijk}
=
\operatorname{Arg}_0
\left(
x^{\mathrm{Re}+}_{ijk}
-
x^{\mathrm{Re}-}_{ijk},
\;
x^{\mathrm{Im}+}_{ijk}
-
x^{\mathrm{Im}-}_{ijk}
\right)
}
\]

となる。

非ゼロ入力では、

\[
\theta_{ijk}
=
\operatorname{atan2}
\left(
x^{\mathrm{Im}+}_{ijk}
-
x^{\mathrm{Im}-}_{ijk},
\;
x^{\mathrm{Re}+}_{ijk}
-
x^{\mathrm{Re}-}_{ijk}
\right).
\]

ゼロ入力では、

\[
a_{ijk}=0,
\qquad
b_{ijk}=0
\]

なので、

\[
\theta_{ijk}=0
\]

とする。

---

# 10. 4チャネルから \(\Log_0(x_{ijk})\) を完全展開

まず

\[
u_{ijk}
=
L_0(r_{ijk}),
\]

\[
v_{ijk}
=
\theta_{ijk}
\]

と置く。

すると

\[
\Log_0(x_{ijk})
=
u_{ijk}
+
iv_{ijk}.
\]

4チャネルを直接代入すると、

\[
\boxed{
\begin{aligned}
\Log_0(x_{ijk})
={}&
L_0
\left(
\sqrt{
\left(
x^{\mathrm{Re}+}_{ijk}
-
x^{\mathrm{Re}-}_{ijk}
\right)^2
+
\left(
x^{\mathrm{Im}+}_{ijk}
-
x^{\mathrm{Im}-}_{ijk}
\right)^2
}
\right)
\\
&+
i\,
\operatorname{Arg}_0
\left(
x^{\mathrm{Re}+}_{ijk}
-
x^{\mathrm{Re}-}_{ijk},
\;
x^{\mathrm{Im}+}_{ijk}
-
x^{\mathrm{Im}-}_{ijk}
\right).
\end{aligned}
}
\]

これが入力4チャネルから複素対数までの完全な実数演算表現である。

---

# 11. 複素指数重み

複素指数重みを

\[
W_{ijk}
=
A_{ijk}
+
iB_{ijk}
\]

とする。

実部・虚部について正負分解を使うなら、

\[
\boxed{
A_{ijk}
=
w^{\mathrm{Re}+}_{ijk}
-
w^{\mathrm{Re}-}_{ijk}
}
\]

\[
\boxed{
B_{ijk}
=
w^{\mathrm{Im}+}_{ijk}
-
w^{\mathrm{Im}-}_{ijk}
}
\]

とする。

例えば各パラメータを

\[
w^{\mathrm{Re}+}_{ijk}>0,
\qquad
w^{\mathrm{Re}-}_{ijk}>0,
\]

\[
w^{\mathrm{Im}+}_{ijk}>0,
\qquad
w^{\mathrm{Im}-}_{ijk}>0
\]

としておけば、任意の複素重みを差として表現できる。

---

# 12. 複素 Product Unit の基本定義

Product Unit を

\[
\boxed{
P_{ik}
=
\exp
\left(
\sum_j
W_{ijk}
\Log_0(x_{ijk})
\right)
}
\]

と定義する。

ここで

\[
W_{ijk}
=
A_{ijk}
+
iB_{ijk},
\]

\[
\Log_0(x_{ijk})
=
u_{ijk}
+
iv_{ijk}.
\]

したがって、

\[
W_{ijk}\Log_0(x_{ijk})
=
(A_{ijk}+iB_{ijk})
(u_{ijk}+iv_{ijk}).
\]

これを展開すると、

\[
\boxed{
W_{ijk}\Log_0(x_{ijk})
=
(A_{ijk}u_{ijk}-B_{ijk}v_{ijk})
+
i
(A_{ijk}v_{ijk}+B_{ijk}u_{ijk})
}
\]

となる。

---

# 13. \(U_{ik}\) と \(V_{ik}\)

実部側を

\[
\boxed{
U_{ik}
=
\sum_j
\left(
A_{ijk}u_{ijk}
-
B_{ijk}v_{ijk}
\right)
}
\]

とする。

虚部側を

\[
\boxed{
V_{ik}
=
\sum_j
\left(
A_{ijk}v_{ijk}
+
B_{ijk}u_{ijk}
\right)
}
\]

とする。

すると

\[
\boxed{
P_{ik}
=
\exp(U_{ik}+iV_{ik})
}
\]

である。

オイラーの公式から、

\[
\boxed{
P_{ik}
=
e^{U_{ik}}
\left(
\cos V_{ik}
+
i\sin V_{ik}
\right)
}
\]

となる。

---

# 14. \(U_{ik}\) の4チャネル完全展開

\[
A_{ijk}
=
w^{\mathrm{Re}+}_{ijk}
-
w^{\mathrm{Re}-}_{ijk}
\]

および

\[
B_{ijk}
=
w^{\mathrm{Im}+}_{ijk}
-
w^{\mathrm{Im}-}_{ijk}
\]

を代入すると、

\[
\boxed{
\begin{aligned}
U_{ik}
=
\sum_j
\Bigg[
&
\left(
w^{\mathrm{Re}+}_{ijk}
-
w^{\mathrm{Re}-}_{ijk}
\right)
L_0
\left(
\sqrt{
\left(
x^{\mathrm{Re}+}_{ijk}
-
x^{\mathrm{Re}-}_{ijk}
\right)^2
+
\left(
x^{\mathrm{Im}+}_{ijk}
-
x^{\mathrm{Im}-}_{ijk}
\right)^2
}
\right)
\\
&-
\left(
w^{\mathrm{Im}+}_{ijk}
-
w^{\mathrm{Im}-}_{ijk}
\right)
\operatorname{Arg}_0
\left(
x^{\mathrm{Re}+}_{ijk}
-
x^{\mathrm{Re}-}_{ijk},
\;
x^{\mathrm{Im}+}_{ijk}
-
x^{\mathrm{Im}-}_{ijk}
\right)
\Bigg].
\end{aligned}
}
\]

---

# 15. \(V_{ik}\) の4チャネル完全展開

同様に、

\[
\boxed{
\begin{aligned}
V_{ik}
=
\sum_j
\Bigg[
&
\left(
w^{\mathrm{Re}+}_{ijk}
-
w^{\mathrm{Re}-}_{ijk}
\right)
\operatorname{Arg}_0
\left(
x^{\mathrm{Re}+}_{ijk}
-
x^{\mathrm{Re}-}_{ijk},
\;
x^{\mathrm{Im}+}_{ijk}
-
x^{\mathrm{Im}-}_{ijk}
\right)
\\
&+
\left(
w^{\mathrm{Im}+}_{ijk}
-
w^{\mathrm{Im}-}_{ijk}
\right)
L_0
\left(
\sqrt{
\left(
x^{\mathrm{Re}+}_{ijk}
-
x^{\mathrm{Re}-}_{ijk}
\right)^2
+
\left(
x^{\mathrm{Im}+}_{ijk}
-
x^{\mathrm{Im}-}_{ijk}
\right)^2
}
\right)
\Bigg].
\end{aligned}
}
\]

したがって、複素 Product Unit は4チャネル入力と4チャネル重みだけから実数演算で計算できる。

---

# 16. Product Unit の実部・虚部

\[
P_{ik}
=
e^{U_{ik}}
\left(
\cos V_{ik}
+
i\sin V_{ik}
\right)
\]

なので、

\[
\boxed{
\Re P_{ik}
=
e^{U_{ik}}\cos V_{ik}
}
\]

\[
\boxed{
\Im P_{ik}
=
e^{U_{ik}}\sin V_{ik}
}
\]

となる。

---

# 17. Product Unit 出力の4チャネル化

Product Unit の出力も正負分解する。

\[
\boxed{
P^{\mathrm{Re}+}_{ik}
=
\max
\left(
e^{U_{ik}}\cos V_{ik},
0
\right)
}
\]

\[
\boxed{
P^{\mathrm{Re}-}_{ik}
=
\max
\left(
-e^{U_{ik}}\cos V_{ik},
0
\right)
}
\]

\[
\boxed{
P^{\mathrm{Im}+}_{ik}
=
\max
\left(
e^{U_{ik}}\sin V_{ik},
0
\right)
}
\]

\[
\boxed{
P^{\mathrm{Im}-}_{ik}
=
\max
\left(
-e^{U_{ik}}\sin V_{ik},
0
\right)
}
\]

したがって、

\[
P_{ik}
=
\left(
P^{\mathrm{Re}+}_{ik}
-
P^{\mathrm{Re}-}_{ik}
\right)
+
i
\left(
P^{\mathrm{Im}+}_{ik}
-
P^{\mathrm{Im}-}_{ik}
\right).
\]

---

# 18. 複素係数

最終加算に用いる複素係数を

\[
a_{ik}
=
a^{\mathrm{Re}}_{ik}
+
i a^{\mathrm{Im}}_{ik}
\]

とする。

正負チャネルを使えば、

\[
a^{\mathrm{Re}}_{ik}
=
a^{\mathrm{Re}+}_{ik}
-
a^{\mathrm{Re}-}_{ik},
\]

\[
a^{\mathrm{Im}}_{ik}
=
a^{\mathrm{Im}+}_{ik}
-
a^{\mathrm{Im}-}_{ik}.
\]

したがって、

\[
\boxed{
a_{ik}
=
\left(
a^{\mathrm{Re}+}_{ik}
-
a^{\mathrm{Re}-}_{ik}
\right)
+
i
\left(
a^{\mathrm{Im}+}_{ik}
-
a^{\mathrm{Im}-}_{ik}
\right)
}
\]

となる。

---

# 19. 最終出力

最終出力は

\[
\boxed{
y_k
=
\sum_i
a_{ik}P_{ik}
}
\]

である。

したがって、

\[
\boxed{
\begin{aligned}
y_k
=
\sum_i
&
\left[
\left(
a^{\mathrm{Re}+}_{ik}
-
a^{\mathrm{Re}-}_{ik}
\right)
+
i
\left(
a^{\mathrm{Im}+}_{ik}
-
a^{\mathrm{Im}-}_{ik}
\right)
\right]
\\
&\times
\exp
\left[
\sum_j
\left(
A_{ijk}+iB_{ijk}
\right)
\Log_0(x_{ijk})
\right].
\end{aligned}
}
\]

さらに \(A_{ijk},B_{ijk},\Log_0(x_{ijk})\) を前節の4チャネル式で完全に展開できる。

---

# 20. 4チャネル化の重要な点

4チャネル化したからといって、

\[
\log x^{\mathrm{Re}+},
\qquad
\log x^{\mathrm{Re}-},
\qquad
\log x^{\mathrm{Im}+},
\qquad
\log x^{\mathrm{Im}-}
\]

を個別に計算するわけではない。

まず、

\[
a
=
x^{\mathrm{Re}+}
-
x^{\mathrm{Re}-},
\]

\[
b
=
x^{\mathrm{Im}+}
-
x^{\mathrm{Im}-}
\]

として複素数の実部・虚部を再構成する。

その後、

\[
r
=
\sqrt{a^2+b^2},
\]

\[
\theta
=
\operatorname{Arg}_0(a,b)
\]

を計算し、

\[
\boxed{
\Log_0(x)
=
L_0(r)
+
i\theta
}
\]

とする。

つまり、4チャネルは複素数の情報を保持する表現であり、複素対数は再構成された大きさと位相に対して定義される。

---

# 21. Branch の問題

主値偏角

\[
\operatorname{Arg}z\in(-\pi,\pi]
\]

を使う場合、負の実軸付近で

\[
+\pi
\longleftrightarrow
-\pi
\]

というジャンプが起こる。

これは同じ方向を表すが、

\[
W\Log z
\]

の内部では数値的に異なる値として現れる。

特に

\[
W=A+iB
\]

の場合、

\[
W\Log z
=
(A\log|z|-B\arg z)
+
i(A\arg z+B\log|z|)
\]

なので、偏角の branch jump は位相だけでなく振幅側にも影響しうる。

したがって、モデル仕様として

\[
\boxed{
\operatorname{Arg}_0
\text{ の branch を固定する}
}
\]

必要がある。

---

# 22. Gradient Floor

通常の勾配を

\[
g
=
\frac{\partial L}{\partial\theta}
\]

とする。

非ゼロ勾配について絶対値を最低 \(\varepsilon\) に保つなら、

\[
\boxed{
\tilde g
=
\begin{cases}
0,
&
g=0,
\\[4pt]
\operatorname{sgn}(g)
\max(|g|,\varepsilon),
&
g\neq0
\end{cases}
}
\]

とする。

これにより、

\[
0,
\qquad
+\varepsilon,
\qquad
-\varepsilon
\]

を別状態として扱える。

---

# 23. Gradient Floor と更新量

更新則を

\[
\theta_{t+1}
=
\theta_t
-
\eta_t\tilde g_t
\]

とすると、非ゼロ勾配では

\[
|\tilde g_t|
\ge
\varepsilon
\]

なので、

\[
|\Delta\theta_t|
\ge
\eta_t\varepsilon.
\]

固定学習率では最適点付近で振動する可能性があるため、

\[
\eta_t\to0
\]

とする learning-rate decay と組み合わせることが考えられる。

さらに必要なら、

\[
\varepsilon_t\to0
\]

とする動的 gradient floor も考えられる。

---

# 24. Gradient Floor の適用タイミング

重要なのは、勾配が数値的 underflow によって完全に

\[
g=0
\]

になった後では、

\[
\tilde g=0
\]

のままになることである。

したがって gradient floor は、

\[
\boxed{
\text{勾配が数値的に0へ潰れる前}
}
\]

に適用する必要がある。

---

# 25. Overflow の有限境界化

正方向に表現範囲を超えた場合、

\[
x\mapsto+\mathrm{MAX}
\]

とする。

負方向なら、

\[
x\mapsto-\mathrm{MAX}.
\]

したがって、

\[
\boxed{
+\infty
\Rightarrow
+\mathrm{MAX}
}
\]

\[
\boxed{
-\infty
\Rightarrow
-\mathrm{MAX}
}
\]

という有限境界状態へ写す。

中心となる考え方は、

\[
\boxed{
\text{「範囲外」}
\neq
\text{「計算不能」}
}
\]

である。

---

# 26. Product Unit における log-domain 飽和

Product Unit は

\[
P_{ik}
=
e^{U_{ik}}
(\cos V_{ik}+i\sin V_{ik})
\]

なので、

\[
|P_{ik}|=e^{U_{ik}}.
\]

したがって、実際に \(e^{U_{ik}}\) を overflow させてから処理するより、

\[
U_{\max}
=
\log(\mathrm{MAX})
\]

を用いて事前に境界化する方が自然である。

例えば、

\[
\hat U_{ik}
=
\min(U_{ik},U_{\max})
\]

とし、

\[
P_{ik}
=
e^{\hat U_{ik}}
(\cos V_{ik}+i\sin V_{ik})
\]

とすれば、指数関数そのものの overflow を避けられる。

下限側については、本設計では

\[
L_0(\varepsilon)=-\infty
\]

という特殊状態を含むため、\(-\infty\) を数学的状態として保持するか、実装上の \(\mathrm{LOG\_MIN}\) へ写すかを別途仕様化する。

---

# 27. 飽和後の Backward

forward で hard clip を行う場合、その演算そのものの厳密な微分は飽和領域で 0 になる。

しかし、学習を継続するために境界上でも非ゼロの勾配を流したい場合は、forward と backward を分離する。

すなわち、

\[
\boxed{
\text{Forward: bounded / saturated operator}
}
\]

\[
\boxed{
\text{Backward: explicitly defined surrogate derivative}
}
\]

とする。

これは通常の解析微分ではなく、代理勾配あるいは straight-through 型の規則として扱う。

---

# 28. 複素飽和の方法

複素数

\[
z=a+ib
\]

を境界化する方法には少なくとも2種類ある。

## 28.1 成分ごとの飽和

\[
a\mapsto S(a),
\qquad
b\mapsto S(b)
\]

とする。

## 28.2 振幅のみを飽和

\[
|z|>\mathrm{MAX}
\]

なら、

\[
\boxed{
z
\mapsto
\mathrm{MAX}\frac{z}{|z|}
}
\]

とする。

この方式では位相を保持できる。

Product Unit では

\[
P=e^{U+iV}
\]

という振幅・位相分離が明示的なので、内部表現については振幅側 \(U\) を境界化し、位相 \(V\) を保持する方式が自然である。

---

# 29. 演算順序と飽和

途中で飽和演算を入れると、通常の実数・複素数演算と同じ代数法則が必ずしも成立しない。

例えば一般に、

\[
S(S(a)+S(b))
\neq
S(a+b)
\]

となりうる。

そのため、

\[
\prod_j x_j^{W_j}
\]

を逐次積で実装する場合と、

\[
\exp\left(
\sum_j W_j\Log_0(x_j)
\right)
\]

を実装する場合で、途中の飽和規則によって結果が異なる可能性がある。

したがって本設計では、

\[
\boxed{
P_{ik}
:=
\exp\left(
\sum_j
W_{ijk}\Log_0(x_{ijk})
\right)
}
\]

を Product Unit の正規定義とし、

\[
\prod_j x_{ijk}^{W_{ijk}}
\]

は説明上の対応式として扱う方が明確である。

---

# 30. 全体の正規形

入力：

\[
\boxed{
x_{ijk}
=
(x^{\mathrm{Re}+}_{ijk}-x^{\mathrm{Re}-}_{ijk})
+
i(x^{\mathrm{Im}+}_{ijk}-x^{\mathrm{Im}-}_{ijk})
}
\]

重み：

\[
\boxed{
W_{ijk}
=
(w^{\mathrm{Re}+}_{ijk}-w^{\mathrm{Re}-}_{ijk})
+
i(w^{\mathrm{Im}+}_{ijk}-w^{\mathrm{Im}-}_{ijk})
}
\]

大きさ：

\[
\boxed{
r_{ijk}
=
\sqrt{
(x^{\mathrm{Re}+}_{ijk}-x^{\mathrm{Re}-}_{ijk})^2
+
(x^{\mathrm{Im}+}_{ijk}-x^{\mathrm{Im}-}_{ijk})^2
}
}
\]

偏角：

\[
\boxed{
v_{ijk}
=
\operatorname{Arg}_0
\left(
x^{\mathrm{Re}+}_{ijk}-x^{\mathrm{Re}-}_{ijk},
x^{\mathrm{Im}+}_{ijk}-x^{\mathrm{Im}-}_{ijk}
\right)
}
\]

対数振幅：

\[
\boxed{
u_{ijk}
=
L_0(r_{ijk})
}
\]

複素対数：

\[
\boxed{
\Log_0(x_{ijk})
=
u_{ijk}+iv_{ijk}
}
\]

Product Unit 内部実部：

\[
\boxed{
U_{ik}
=
\sum_j
(A_{ijk}u_{ijk}-B_{ijk}v_{ijk})
}
\]

Product Unit 内部虚部：

\[
\boxed{
V_{ik}
=
\sum_j
(A_{ijk}v_{ijk}+B_{ijk}u_{ijk})
}
\]

Product Unit：

\[
\boxed{
P_{ik}
=
e^{U_{ik}}
(\cos V_{ik}+i\sin V_{ik})
}
\]

最終出力：

\[
\boxed{
y_k
=
\sum_i a_{ik}P_{ik}
}
\]

となる。

---

# 31. 設計思想のまとめ

本方式では、複素 Product Unit を単なる複素数演算としてではなく、

\[
\boxed{
\text{4チャネル有限状態表現}
+
\text{拡張複素対数}
+
\text{log-domain Product Unit}
+
\text{境界付き学習則}
}
\]

として構成する。

特に、

\[
\boxed{
0
}
\]

と

\[
\boxed{
\pm\varepsilon
}
\]

を明確に別状態として扱う。

また、

\[
\log_0(0)=0
\]

と

\[
\log_0(\varepsilon)=-\infty
\]

を同時に採用することで、ゼロ状態と最小非ゼロ状態を対数空間でも区別する。

複素入力については、

\[
x^{\mathrm{Re}+},
\quad
x^{\mathrm{Re}-},
\quad
x^{\mathrm{Im}+},
\quad
x^{\mathrm{Im}-}
\]

から一度

\[
a+ib
\]

を再構成し、

\[
r=\sqrt{a^2+b^2}
\]

と

\[
\theta=\operatorname{Arg}_0(a,b)
\]

を求める。

したがって、4チャネル化しても複素数としての振幅・位相情報を保持できる。

最終的に Product Unit 全体は、

\[
\boxed{
P_{ik}
=
\exp\left(
\sum_j
W_{ijk}\Log_0(x_{ijk})
\right)
}
\]

を正規形とし、その内部をすべて実数演算へ展開可能である。

---

# 32. 実装時に固定すべき仕様

実装前に、少なくとも以下を明示的に固定する必要がある。

1. \(\operatorname{Arg}_0\) の branch 範囲。
2. \(\operatorname{Arg}_0(0,0)=0\) を採用するか。
3. \(L_0(0)=0\) の定義。
4. \(L_0(\varepsilon)=-\infty\) の数学的状態を実装上どう表現するか。
5. \(-\varepsilon\) や複素最小振幅状態に対する \(L_0\) と偏角の扱い。
6. \(U\) の上限・下限飽和規則。
7. 飽和後の backward に用いる代理微分。
8. gradient floor を optimizer のどの段階で適用するか。
9. \(\varepsilon\) を固定するか、学習中に変化させるか。
10. 複素勾配を実部・虚部独立の実微分として扱うか、Wirtinger 微分として扱うか。
11. 4チャネルパラメータの冗長性を許容するか、正則化や制約を追加するか。
12. `MAX`, `LOG_MIN`, \(\varepsilon\) をデータ型ごとにどう定義するか。

これらを固定すれば、数学的な演算規則から具体的な複素 Product Unit 実装へ落とし込める。
