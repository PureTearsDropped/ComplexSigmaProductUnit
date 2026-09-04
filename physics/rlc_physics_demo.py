import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


torch.manual_seed(20260904)
torch.set_default_dtype(torch.float64)


def hadamard(n: int) -> torch.Tensor:
    if n < 1 or (n & (n - 1)):
        raise ValueError('n must be a power of two')
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0)
    return H


def walsh_row_tensors_2x2() -> torch.Tensor:
    """Rows h_r ⊗ h_s, flattened: shape [4,4]."""
    H2 = hadamard(2)
    return torch.einsum('ri,sj->rsij', H2, H2).reshape(4, 4)


def positive(raw: torch.Tensor, eps: float) -> torch.Tensor:
    return F.softplus(raw) + eps


def raw_for_positive_value(value: torch.Tensor, eps: float) -> torch.Tensor:
    y = value - eps
    if torch.any(y <= 0):
        raise ValueError('target positive values must exceed eps')
    return y + torch.log(-torch.expm1(-y))


class FourChannelExponent(torch.nn.Module):
    """W=A+iB, A=w_Re+ - w_Re-, B=w_Im+ - w_Im-; all channels start at 1."""
    def __init__(self, units: int, features: int, eps: float = 1e-9):
        super().__init__()
        self.eps = eps
        shape = (units, features)
        one = torch.ones(shape)
        raw1 = raw_for_positive_value(one, eps)
        self.rp = torch.nn.Parameter(raw1.clone())
        self.rn = torch.nn.Parameter(raw1.clone())
        self.ip = torch.nn.Parameter(raw1.clone())
        self.in_ = torch.nn.Parameter(raw1.clone())

    def channels(self):
        e = self.eps
        return positive(self.rp,e), positive(self.rn,e), positive(self.ip,e), positive(self.in_,e)

    def effective(self):
        rp,rn,ip,inn = self.channels()
        return rp-rn, ip-inn

    def product(self, x: torch.Tensor) -> torch.Tensor:
        # x [N,F] positive-real physical dimensionless variables
        A,B = self.effective()
        lx = torch.log(x)
        U = torch.einsum('nf,uf->nu', lx, A)
        V = torch.einsum('nf,uf->nu', lx, B)
        return torch.exp(torch.complex(U,V))


class FourChannelComplexCoeff(torch.nn.Module):
    """Arbitrary complex coefficients as differences of four strictly-positive channels."""
    def __init__(self, initial: torch.Tensor, eps: float = 1e-9, base: float = 1.0):
        super().__init__()
        self.eps = eps
        initial = initial.to(torch.complex128)
        re, im = initial.real, initial.imag
        rp = torch.full_like(re, base) + torch.clamp(re, min=0)
        rn = torch.full_like(re, base) + torch.clamp(-re, min=0)
        ip = torch.full_like(im, base) + torch.clamp(im, min=0)
        inn = torch.full_like(im, base) + torch.clamp(-im, min=0)
        self.rp = torch.nn.Parameter(raw_for_positive_value(rp,eps))
        self.rn = torch.nn.Parameter(raw_for_positive_value(rn,eps))
        self.ip = torch.nn.Parameter(raw_for_positive_value(ip,eps))
        self.in_ = torch.nn.Parameter(raw_for_positive_value(inn,eps))

    def channels(self):
        e=self.eps
        return positive(self.rp,e), positive(self.rn,e), positive(self.ip,e), positive(self.in_,e)

    def effective(self):
        rp,rn,ip,inn=self.channels()
        return torch.complex(rp-rn, ip-inn)


def cmse(a,b):
    return (a-b).abs().square().mean()


def make_rlc_data(n: int):
    """Dimensionless series-RLC dataset with physical reference scales.

    R0 = 10 ohm, omega0=1000 rad/s, L0=10 mH, C0=100 uF.
    All three impedance scales are then Z0=10 ohm:
      R0 = omega0 L0 = 1/(omega0 C0) = 10 ohm.

    x=[r,w,l,c] are ratios to reference values and
      z=Z/Z0 = r + i*w*l - i/(w*c).
    """
    lo,hi = math.log(0.5), math.log(2.0)
    x = torch.exp(torch.empty(n,4).uniform_(lo,hi))
    r,w,l,c = x.unbind(dim=1)
    qR = r
    qL = w*l
    qC = 1.0/(w*c)
    terms = torch.stack([
        torch.complex(qR, torch.zeros_like(qR)),
        torch.complex(torch.zeros_like(qL), qL),
        torch.complex(torch.zeros_like(qC), -qC),
    ], dim=1)
    z = terms.sum(dim=1)
    q = torch.stack([qR,qL,qC],dim=1)
    return x, q, terms, z


def train_exponents(x_train, terms_train, x_test, q_test):
    units, features = 3, 4
    model = FourChannelExponent(units,features)
    A0,B0=model.effective()
    init_channels=model.channels()

    # Coefficients cancel exactly in term-wise ratios Z_n/Z_ref.
    # This lets us infer W first without knowing {1,+i,-i}.
    target_ratio = terms_train / terms_train[0:1,:]

    opt=torch.optim.Adam(model.parameters(), lr=0.025)
    log=[]
    for step in range(2600):
        opt.zero_grad(set_to_none=True)
        P=model.product(x_train)
        pred_ratio=P/P[0:1,:]
        loss=cmse(pred_ratio,target_ratio)
        loss.backward(); opt.step()
        if step in (0,49,99,199,399,799,1299,1999,2599):
            A,B=model.effective()
            true_A=torch.tensor([
                [1.,0.,0.,0.],
                [0.,1.,1.,0.],
                [0.,-1.,0.,-1.],
            ])
            true_B=torch.zeros_like(true_A)
            log.append((step+1,loss.item(),(A-true_A).abs().max().item(),(B-true_B).abs().max().item()))

    A,B=model.effective()
    with torch.no_grad():
        Ptest=model.product(x_test)
        # Scale each learned product to physical magnitude using one reference-free best scalar,
        # only for checking the exponent shape. Exact exponents imply P=q directly because no constant feature.
        qhat=Ptest.real
        rel=((qhat-q_test).abs()/(q_test.abs()+1e-12)).mean().item()
    return model, A.detach(), B.detach(), init_channels, A0.detach(), B0.detach(), log, rel


def train_coefficients(P_train, z_train, P_test, z_test, init: torch.Tensor, steps=2200, lr=0.035):
    coeff=FourChannelComplexCoeff(init)
    opt=torch.optim.Adam(coeff.parameters(),lr=lr)
    first_below=None
    log=[]
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        a=coeff.effective()
        pred=P_train @ a
        loss=cmse(pred,z_train)
        loss.backward(); opt.step()
        val=loss.item()
        if first_below is None and val < 1e-10:
            first_below=step+1
        if step in (0,9,49,99,199,399,799,1399,2199):
            with torch.no_grad():
                at=coeff.effective()
                tl=cmse(P_test@at,z_test).item()
            log.append((step+1,val,tl))
    with torch.no_grad():
        a=coeff.effective().detach()
        train_loss=cmse(P_train@a,z_train).item()
        test_loss=cmse(P_test@a,z_test).item()
    return a, train_loss, test_loss, first_below, log


def main():
    # Data
    x,q,terms,z=make_rlc_data(5000)
    ntr=4000
    xtr,xte=x[:ntr],x[ntr:]
    qtr,qte=q[:ntr],q[ntr:]
    termtr,termte=terms[:ntr],terms[ntr:]
    ztr,zte=z[:ntr],z[ntr:]

    print('PHYSICS: dimensionless series RLC')
    print('z = Z/Z0 = r + i*w*l - i/(w*c)')
    print('references: R0=10 ohm, omega0=1000 rad/s, L0=10 mH, C0=100 uF, Z0=10 ohm')

    # Phase 1
    model,A,B,init_channels,A0,B0,elog,rel=train_exponents(xtr,termtr,xte,qte)
    print('\nPHASE 1: exponent-only, component coefficients cancelled by ratios')
    print('all four exponent channels exactly 1 at init:', all(torch.equal(c,torch.ones_like(c)) for c in init_channels))
    print('initial max |A|=',A0.abs().max().item(),'initial max |B|=',B0.abs().max().item())
    for row in elog:
        print('step=%4d loss=%.3e maxAerr=%.3e maxBerr=%.3e'%row)
    print('learned A=')
    print(A)
    print('learned B=')
    print(B)
    print('held-out mean relative magnitude error:',rel)

    # Freeze exponents and build three physical product magnitudes + one constant dummy PU.
    for p in model.parameters(): p.requires_grad_(False)
    with torch.no_grad():
        Ptr3=model.product(xtr)
        Pte3=model.product(xte)
        ones_tr=torch.ones(Ptr3.shape[0],1,dtype=torch.complex128)
        ones_te=torch.ones(Pte3.shape[0],1,dtype=torch.complex128)
        Ptr=torch.cat([Ptr3,ones_tr],dim=1)
        Pte=torch.cat([Pte3,ones_te],dim=1)

    true_a=torch.tensor([1+0j,0+1j,0-1j,0+0j],dtype=torch.complex128)
    print('\nPHASE 2: freeze exponents, learn complex coefficients from total Z only')
    print('true coefficients [R,L,C,dummy]=',true_a)

    Wrows=walsh_row_tensors_2x2().to(torch.complex128)
    initializers={
        'all_ones': torch.ones(4,dtype=torch.complex128),
        'all_zero': torch.zeros(4,dtype=torch.complex128),
    }
    for k,row in enumerate(Wrows):
        initializers[f'walsh_tensor_row_{k}']=row

    results={}
    for name,init in initializers.items():
        a,trl,tel,hit,clog=train_coefficients(Ptr,ztr,Pte,zte,init)
        results[name]=(a,trl,tel,hit,clog)
        print(f'\n{name}: init={init.tolist()}')
        print(' learned a=',a.tolist())
        print(' train loss=%.3e test loss=%.3e first step loss<1e-10=%s'%(trl,tel,str(hit)))
        print(' max |a-true|=%.3e'%((a-true_a).abs().max().item()))

    # Direct physical held-out sanity check using best learned coefficients.
    # Pick Walsh row with smallest held-out loss.
    best=min(results.items(), key=lambda kv: kv[1][2])
    name,(a,trl,tel,hit,clog)=best
    with torch.no_grad():
        pred=Pte@a
        relZ=((pred-zte).abs()/(zte.abs()+1e-12)).mean().item()
        maxrel=((pred-zte).abs()/(zte.abs()+1e-12)).max().item()
    print('\nBEST coefficient init:',name)
    print('held-out mean relative |Z| complex error=%.3e max=%.3e'%(relZ,maxrel))

    # Save machine-readable-ish summary
    out=Path(__file__).with_name('rlc_physics_result.txt')
    lines=[]
    lines.append('Series RLC: z = r + i*w*l - i/(w*c)')
    lines.append('Phase1 learned A:\n'+str(A.tolist()))
    lines.append('Phase1 learned B:\n'+str(B.tolist()))
    lines.append(f'Phase1 held-out mean relative magnitude error: {rel:.6e}')
    for name,(a,trl,tel,hit,_) in results.items():
        lines.append(f'{name}: coeff={a.tolist()} train={trl:.6e} test={tel:.6e} first_below_1e-10={hit}')
    lines.append(f'best={best[0]} mean_relative_complex_Z_error={relZ:.6e} max_relative_complex_Z_error={maxrel:.6e}')
    out.write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':
    main()
