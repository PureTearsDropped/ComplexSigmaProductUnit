# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
# cpe_total_demo.jl — Experiment 3 of complex_product_unit_physics: a *fractional* power of a *complex*
# variable that exists in the laboratory — the constant-phase element (CPE; Warburg at α = ½):
#
#       Z(ω) = R + 1/(Q·(iω)^α)        ⟺   z = r + a·s^W,   s = iw,   W = −α + 0i,   a = 1
#
# (dimensionless: w = ω/ω₀, z = Z·Q·ω₀^α; r = R·Q·ω₀^α).  A complex Product Unit in the normal form of
# 「複素 Product Unit と有限境界学習系」§29,  P = exp(W·Log₀ s),  with a complex INPUT s = iw and a complex
# exponent W = A + iB from four positive channels (A = w⁺−w⁻, B = v⁺−v⁻, all 1 at start ⟹ W = 0, P = 1).
# Log₀ is ScalarTotComplex's log: Log₀(0) = 0 with Arg₀(0) = 0 (§4, §6, §7), L₀(ε) = the tagged boundary
# log MIN⟦≥⟧ (§4's LOG_MIN), the amplitude saturates and the phase is kept (§26, §28.2 — that is the polar
# form), and §22's gradient floor is not an optimizer step here but the arithmetic itself: a gradient that
# would underflow comes back as ±MIN⟦≤⟧ with its sign, and the flag says so.
# The CPE is the element whose phase is constant, −απ/2: **arg reads the exponent**.  The staged schedule
# of the RLC / Schrödinger demos (exponent first with the coefficient cancelled by ratios, freeze, then the
# coefficients), hand-written gradients checked by finite differences, run twice — IEEE Complex{Float64}
# and total arithmetic — then pushed through the boundaries the earlier experiments avoided: ω = 0 and
# ω = ε, an absurd frequency range, and the two sides of the branch cut of s^α in the Laplace half-plane.
include(joinpath(get(ENV, "TOTAL_ARITH_CUDA", joinpath(@__DIR__, "..", "..", "total-arith-cuda")), "julia", "ScalarTotComplex.jl"))   # total-arith-cuda ≥ v1.2.0
using .ScalarTot, .ScalarTotComplex
using Random, Printf
const C = TotComplex
const α_true = 0.8; const r_true = 0.002; const a_true = 1.0 + 0.0im        # R = 20 Ω, Q = 1e-4, ω₀ = 1 rad/s

# ---- Part A: the physics face of the reserved word ------------------------------------------------
println("A. the constant-phase element in total arithmetic (α = $α_true)")
for w in (1e-2, 1.0, 1e4)
    p = (IM * w)^(-α_true)
    @printf("   ω = %-6g  (iω)^(−α) = %s     arg/π = %.4f  (= −α/2 = %.4f: the phase reads the exponent)\n", w, p, p.t, -α_true / 2)
end
PU(s, W) = exp(W * log(s))                                                # the normal form of §29: exp(W·Log₀ s)
Wα = C(-α_true, 0.0)
zε = PU(IM * C(MINF, 0.0, LE), Wα); zM = PU(IM * C(MAXF), Wα)
println("   ω = 0 (exact):  Log₀(i·0) = ", log(IM * C(0.0)), " (§7: L₀(0) = 0, Arg₀(0) = 0) → P = exp(W·0) = ", PU(IM * C(0.0), Wα),
        "  ← a zero input drops out of the product (§29's normal form)")
println("                   the reserved-word power says (i·0)^(−α) = ", (IM * C(0.0))^(-α_true), " — both are total; the document fixes exp∘Log₀ and so does this demo")
println("   ω = ε (MIN⟦≤⟧): Log₀(iε) = ", log(IM * C(MINF, 0.0, LE)), " (L₀(ε) = LOG_MIN, tagged ≥) → P = ", zε, "  ← the limit: |P| ≥ 3e246, phase −α/2 kept: a blocking electrode")
println("   ω = MAX:        P = ", zM)
# the series capacitor of Experiment 1, same distinction: C = 0 vs C = ε
lc(w, l, c) = IM * w * l - IM / (w * c)                                   # i wl − i/(wc)
println("   RLC of Experiment 1 at w = 1, l = 1:  C = 0 → z_LC = ", lc(C(1.0), C(1.0), C(0.0)),
        "  (1/0 = 0: the term is absent — the arithmetic reads C = 0 as \"no capacitor term\", i.e. a short);")
println("                                         C = ε → z_LC = ", lc(C(1.0), C(1.0), C(MINF, 0.0, LE)),
        "  (an open circuit: the limit, flagged).  Physics wants the open at C → 0: the reserved word says")
println("                                         the natural variable of a series capacitor is the elastance S = 1/C, whose 0 IS the absent element.")

# ---- Part B: the staged learning, two arithmetics --------------------------------------------------
# model: z(s) = r + a·s^W,  θ = [u₁ u₂ u₃ u₄ | a_re a_im | r],  W = (e^{u₁} − e^{u₂}) + i(e^{u₃} − e^{u₄})
# loss: mean |ẑ − z|² / |z|²  (relative — |z| spans decades).  Gradients by hand; a flagged contribution
# (total arithmetic) is excluded from the sum and counted.
struct IEEE end; struct TOT end
cnum(::Type{IEEE}, re, im) = Complex{Float64}(re, im)
cnum(::Type{TOT}, re, im) = C(re, im)
rnum(::Type{IEEE}, x) = Float64(x)
rnum(::Type{TOT}, x) = TotNum(Float64(x))
bad(::Type{IEEE}, x) = false                      # IEEE has no flags: whatever comes out is used (that is the point)
bad(::Type{TOT}, x::TotNum) = isflagged(x)
val(x::Float64) = x; val(x::TotNum) = x.val
cre(z::Complex) = real(z); cre(z::C) = real(z)
function forward(::Type{B}, θ, s) where {B}
    u1, u2, u3, u4, are, aim, r = θ
    A = exp(u1) - exp(u2); Bi = exp(u3) - exp(u4)
    W = cnum(B, A, Bi); a = cnum(B, are, aim)
    P = pu(B, s, W)
    zhat = a * P + rnum(B, r)
    zhat, P, W, a
end
pu(::Type{IEEE}, s, W) = exp(W * log(s))                                   # IEEE: log 0 = −Inf, and NaN follows
pu(::Type{TOT}, s, W) = exp(W * log(s))                                    # total: Log₀ 0 = 0, ε → LOG_MIN⟦≥⟧, |P| saturates, phase kept
# the gradient policy for a flagged contribution (total arithmetic only):
#   ⟦≤⟧ only — an underflow: the value ±MIN with its sign IS §22's gradient floor → used as is;
#   ⟦≥⟧ (overflow) or ⟦±⟧ (sign unknown) — not a number that can be trusted → excluded and counted.
gpolicy(::Type{IEEE}, x) = (true, 0)                                       # IEEE has no flags: whatever comes out is used (that is the point)
function gpolicy(::Type{TOT}, x::TotNum)
    x.flag == 0x00 && return (true, 0)
    x.flag == LE && return (true, 1)                                       # floored, kept
    (false, 2)                                                             # saturated / unknown, excluded
end
"""loss and gradient over the samples.  `ratio = true` is stage 1: the coefficient cancels in the ratios P(s_n)/P(s_0),
and the fit is in the LOG domain — the Product Unit's own domain, U + iV = W·(Log₀ s_n − Log₀ s_0) against Log₀ of the
data ratio, linear in W (a relative error on the ratios themselves is a 1e12-steep valley over six decades: Adam stalls
in it — measured).  Returns (loss, grad, floored, excluded)."""
function lossgrad(::Type{B}, θ, S, Z; ratio::Bool = false) where {B}
    u1, u2, u3, u4, are, aim, r = θ
    n = length(S); L = rnum(B, 0.0); g = zeros(7); floored = 0; excl = 0; used = 0
    P0 = ratio ? forward(B, θ, S[1])[2] : nothing
    for k in (ratio ? (2:n) : (1:n))
        s = S[k]; z = Z[k]
        zhat, P, W, a = forward(B, θ, s)
        lg = log(s)                                      # Log₀ of the input
        if ratio                                         # log domain: e = W·(Log₀ s − Log₀ s₀) − Log₀(z_n/z_0)
            lg = lg - log(S[1]); res = W * lg - log(z); wgt = rnum(B, 1.0)
            dA = lg; dB = cnum(B, 0.0, 1.0) * lg; P = cnum(B, 0.0, 0.0)
        else
            res = zhat - z
            wgt = cre(rnum(B, 1.0) / (z * conj(z)))      # 1/|z|²  (total: |z|² = 0 → 1/0 = 0: a zero target has no relative error)
            dA = a * P * lg; dB = a * P * (cnum(B, 0.0, 1.0) * lg)
        end
        lk = cre(res * conj(res)) * wgt
        parts = (cre(conj(res) * dA) * (2 * exp(u1)) * wgt, cre(conj(res) * dA) * (-2 * exp(u2)) * wgt,
                 cre(conj(res) * dB) * (2 * exp(u3)) * wgt, cre(conj(res) * dB) * (-2 * exp(u4)) * wgt,
                 cre(conj(res) * P) * 2 * wgt, cre(conj(res) * (cnum(B, 0.0, 1.0) * P)) * 2 * wgt,
                 cre(res) * 2 * wgt)
        okL, _ = gpolicy(B, lk)
        pol = map(p -> gpolicy(B, p), parts)
        if !okL || any(p -> !p[1], pol)
            excl += 1; continue                          # a saturated / sign-unknown contribution is left out
        end
        floored += count(p -> p[2] == 1, pol)
        used += 1
        L = L + lk
        for j in 1:7; g[j] += val(parts[j]); end
    end
    used == 0 && return NaN, fill(NaN, 7), floored, excl
    val(L) / used, g ./ used, floored, excl
end
"""Adam on the parameters in `idx`; stops when the loss is below `tol` (or a NaN appears: IEEE is then dead)."""
function adam!(θ, ::Type{B}, S, Z, idx; steps = 3000, lr = 0.02, ratio = false, tol = 1e-24) where {B}
    m = zeros(7); v = zeros(7); b1 = 0.9; b2 = 0.999; floored = 0; excl = 0; L = NaN
    for t in 1:steps
        L, g, floored, excl = lossgrad(B, θ, S, Z; ratio)
        (isnan(L) || any(isnan, g[idx])) && return L, floored, excl, t
        L < tol && return L, floored, excl, t
        m .= b1 .* m .+ (1 - b1) .* g; v .= b2 .* v .+ (1 - b2) .* g .^ 2
        for j in idx
            θ[j] -= lr * (m[j] / (1 - b1^t)) / (sqrt(v[j] / (1 - b2^t)) + 1e-12)
        end
    end
    L, floored, excl, steps
end
Wof(θ) = (exp(θ[1]) - exp(θ[2])) + im * (exp(θ[3]) - exp(θ[4]))
function truth(::Type{B}, w) where {B}                    # the data come from the same normal form, exp(W·Log₀ s)
    s = cnum(B, 0.0, 1.0) * rnum(B, w)
    s, rnum(B, r_true) + cnum(B, real(a_true), imag(a_true)) * pu(B, s, cnum(B, -α_true, 0.0))
end
θ0() = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]            # four channels at 1 (W = 0, P = 1), a = 1, r = 0

# finite-difference check of the hand gradient (IEEE and total, ordinary operands)
println("\nB. the gradient, checked (finite differences vs the hand-written formula)")
rng = MersenneTwister(1)
ws = exp10.(range(-2, 4, length = 40))
for Bk in (IEEE, TOT)
    S = [truth(Bk, w)[1] for w in ws]; Z = [truth(Bk, w)[2] for w in ws]
    Q = [(Z[k] - rnum(Bk, r_true)) / (Z[1] - rnum(Bk, r_true)) for k in eachindex(Z)]
    θ = θ0() .+ 0.05 .* randn(rng, 7)                    # (a large perturbation puts the loss at 1e12 where finite differences are noise)
    for ratio in (false, true)
        tgt = ratio ? Q : Z
        _, g, _, _ = lossgrad(Bk, θ, S, tgt; ratio)
        worst = 0.0
        for j in (ratio ? (1:4) : (1:7))
            h = 1e-6; θp = copy(θ); θp[j] += h; θm = copy(θ); θm[j] -= h
            fd = (lossgrad(Bk, θp, S, tgt; ratio)[1] - lossgrad(Bk, θm, S, tgt; ratio)[1]) / 2h
            worst = max(worst, abs(fd - g[j]) / max(abs(fd), 1e-12))
        end
        @printf("   %-5s %-12s max relative deviation over the 7 parameters: %.2e\n", Bk, ratio ? "(stage 1)" : "(stage 2)", worst)
    end
end

# the staged runs (the schedule of Experiments 1–2): stage 1 fits W on the ratios (z_n − r)/(z_0 − r) of the
# CPE term — the coefficient cancels; r is the series resistance, read off the high-frequency plateau in the
# laboratory and taken as known here — then W is frozen and a, r are fitted on the full impedance.
function run(label, Bk, ws; noise = 0.0, seed = 3)
    rng = MersenneTwister(seed)
    S = Any[]; Z = Any[]; Q = Any[]
    for w in ws
        s, z = truth(Bk, w)
        noise > 0 && (z = z * cnum(Bk, 1 + noise * randn(rng), noise * randn(rng)))   # multiplicative complex noise
        push!(S, s); push!(Z, z)
    end
    rr = rnum(Bk, r_true)
    Q = [(Z[k] - rr) / (Z[1] - rr) for k in eachindex(Z)]                              # the ratios of the CPE term (Q[1] = 1)
    θ = θ0()
    L1, fl1, ex1, t1 = adam!(θ, Bk, S, Q, 1:4; ratio = true, lr = 0.05)
    W = Wof(θ)
    L2, fl2, ex2, t2 = adam!(θ, Bk, S, Z, 5:7; lr = 0.01, steps = 6000)
    @printf("   %-44s %-5s stage 1: W = %+.6f%+.6fi (true %+.1f) loss %-8.1e %4d steps, floored %d excluded %d\n",
            label, Bk, real(W), imag(W), -α_true, L1, t1, fl1, ex1)
    @printf("   %-44s %-5s stage 2: a = %+.6f%+.6fi (true 1)  r = %.6f (true %.3f)  loss %-8.1e %4d steps, floored %d excluded %d\n",
            "", "", θ[5], θ[6], θ[7], r_true, L2, t2, fl2, ex2)
    θ
end
println("\nC. the staged learning: exponent from the ratios in the log domain (coefficient cancelled), freeze, then a and r on z = r + a·exp(W·Log₀ s)")
run("40 frequencies in [1e-2, 1e4] rad/s, exact", IEEE, ws)
run("40 frequencies in [1e-2, 1e4] rad/s, exact", TOT, ws)
run("same, 3 % multiplicative noise", IEEE, ws; noise = 0.03)
run("same, 3 % multiplicative noise", TOT, ws; noise = 0.03)
println("\nD. the boundaries the earlier experiments avoided (the same schedule, the data set extended)")
ws0 = vcat(ws, 0.0)                                   # a DC datum: ω = 0 exactly (Log₀ 0 in the model; the truth is the same model)
run("+ the DC point ω = 0 (Log₀ 0 = 0 in the model)", IEEE, ws0)
run("+ the DC point ω = 0 (Log₀ 0 = 0 in the model)", TOT, ws0)
wsx = vcat(ws, exp10.(range(-300, -280, length = 6)), exp10.(range(280, 300, length = 6)))   # |z|² overflows at the ends
run("+ 12 frequencies at 1e±280…±300 (|z|² overflows)", IEEE, wsx)
run("+ 12 frequencies at 1e±280…±300 (|z|² overflows)", TOT, wsx)
println("   (a floored contribution is one whose gradient underflowed and came back as ±MIN⟦≤⟧ — §22's floor, applied by the arithmetic;")
println("    an excluded one saturated to MAX⟦≥⟧ or lost its sign — §27's surrogate is 'not a number: leave it out' here)")

# ---- Part E: the branch cut of s^α — Kahan's signed zero, and what ε does instead ---------------------
println("\nE. the branch cut of s^(−α) on the negative real axis of the Laplace plane (s = σ + iω, σ < 0)")
sI = -1.0 + 0.0im; sIm = -1.0 - 0.0im
println("   IEEE keeps the side in the sign of a zero (Kahan 1987):  (−1 + 0i)^(−α) = ", sI^(-α_true), "   (−1 − 0i)^(−α) = ", sIm^(-α_true))
t0 = C(-1.0)^(-α_true); tp = (C(-1.0) + IM * C(MINF, 0.0, LE))^(-α_true); tm = (C(-1.0) - IM * C(MINF, 0.0, LE))^(-α_true)
println("   total arithmetic has one zero: (−1)^(−α) = ", t0, "  (principal: arg = π)")
println("   the sides are values, not a sign bit:  (−1 + iε)^(−α) = ", tp, "   (−1 − iε)^(−α) = ", tm)
sp = C(-1.0) + IM * C(MINF, 0.0, LE); sm = C(-1.0) - IM * C(MINF, 0.0, LE)
println("   … but the polar form stores arg/π ∈ (−1, 1] in a Float64: (−1 − iε) has arg/π = −1 + 7e-309, which rounds to −1 and wraps to +1:")
println("       arg(−1 + iε)/π = ", sp.t, "   arg(−1 − iε)/π = ", sm.t, "   flags ", sp.flag, " / ", sm.flag)
println("   the direction ε carries survives as a *value* only while |Im/Re| ≳ 1e-16 — below that the polar representation, not the")
println("   arithmetic, forgets the side.  Kahan's sign bit is a 1-bit answer to a question the polar form cannot even ask;")
println("   a Cartesian TotComplex (re, im each totalized) would keep im = −MIN⟦≤⟧ and answer it.  Recorded as a limitation.")
for σ in (-1.0, -1e-3)
    for ω in (1e-8, 1e-15, 1e-17)
        zp = C(σ, ω)^(-α_true); zm = C(σ, -ω)^(-α_true)
        @printf("       s = %g ± %gi:  arg/π of s^(−α) = %+.6f / %+.6f %s\n", σ, ω, zp.t, zm.t, zp.t == zm.t ? "← merged" : "")
    end
end
