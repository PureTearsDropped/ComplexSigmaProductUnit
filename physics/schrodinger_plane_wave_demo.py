import math
from pathlib import Path
import torch
import torch.nn.functional as F

torch.manual_seed(20260904)
torch.set_default_dtype(torch.float64)


def positive(raw, eps):
    return F.softplus(raw)+eps

def raw_for_positive_value(value, eps):
    y=value-eps
    return y+torch.log(-torch.expm1(-y))

class FourChannelExponent(torch.nn.Module):
    def __init__(self, features=2, eps=1e-9):
        super().__init__(); self.eps=eps
        one=torch.ones(features)
        r=raw_for_positive_value(one,eps)
        self.rp=torch.nn.Parameter(r.clone()); self.rn=torch.nn.Parameter(r.clone())
        self.ip=torch.nn.Parameter(r.clone()); self.in_=torch.nn.Parameter(r.clone())
    def channels(self):
        e=self.eps
        return positive(self.rp,e),positive(self.rn,e),positive(self.ip,e),positive(self.in_,e)
    def effective(self):
        rp,rn,ip,inn=self.channels(); return rp-rn,ip-inn
    def product(self,X):
        A,B=self.effective(); lx=torch.log(X)
        U=lx@A; V=lx@B
        return torch.exp(torch.complex(U,V))

class FourChannelCoeff(torch.nn.Module):
    def __init__(self, init=1+0j, eps=1e-9, base=1.0):
        super().__init__(); self.eps=eps
        z=torch.tensor(init,dtype=torch.complex128)
        re,im=z.real,z.imag
        vals=[base+torch.clamp(re,min=0),base+torch.clamp(-re,min=0),base+torch.clamp(im,min=0),base+torch.clamp(-im,min=0)]
        raws=[raw_for_positive_value(v.reshape(1),eps) for v in vals]
        self.rp=torch.nn.Parameter(raws[0]); self.rn=torch.nn.Parameter(raws[1]); self.ip=torch.nn.Parameter(raws[2]); self.in_=torch.nn.Parameter(raws[3])
    def effective(self):
        e=self.eps
        re=positive(self.rp,e)-positive(self.rn,e)
        im=positive(self.ip,e)-positive(self.in_,e)
        return torch.complex(re,im).squeeze(0)

def cmse(a,b): return (a-b).abs().square().mean()

def main():
    # Dimensionless free-particle plane wave. Set hbar=1, m=1/2 => omega=k^2.
    k=1.7; omega=k*k
    amp=torch.tensor(0.8-0.6j,dtype=torch.complex128)
    n=4096
    x=torch.empty(n).uniform_(-1.2,1.2)
    t=torch.empty(n).uniform_(-0.8,0.8)
    # Positive Product-Unit inputs: X=e^x, T=e^t
    inp=torch.stack([torch.exp(x),torch.exp(t)],dim=1)
    psi=amp*torch.exp(1j*(k*x-omega*t))
    ntr=3300
    Xtr,Xte=inp[:ntr],inp[ntr:]; ptr,pte=psi[:ntr],psi[ntr:]

    model=FourChannelExponent(2)
    init_channels=model.channels(); A0,B0=model.effective()
    # cancel unknown complex amplitude A by a ratio to one reference sample
    target_ratio=ptr/ptr[0]
    opt=torch.optim.Adam(model.parameters(),lr=0.025)
    trueA=torch.tensor([0.,0.]); trueB=torch.tensor([k,-omega])
    logs=[]
    for step in range(2200):
        opt.zero_grad(set_to_none=True)
        P=model.product(Xtr); pred=P/P[0]
        loss=cmse(pred,target_ratio); loss.backward(); opt.step()
        if step in (0,49,99,199,399,799,1299,2199):
            A,B=model.effective(); logs.append((step+1,loss.item(),(A-trueA).abs().max().item(),(B-trueB).abs().max().item()))
    A,B=model.effective(); A=A.detach(); B=B.detach()

    for p in model.parameters(): p.requires_grad_(False)
    with torch.no_grad(): Ptr=model.product(Xtr); Pte=model.product(Xte)
    coeff=FourChannelCoeff(1+0j)
    opt2=torch.optim.Adam(coeff.parameters(),lr=0.04)
    for step in range(1800):
        opt2.zero_grad(set_to_none=True); a=coeff.effective(); pred=a*Ptr
        l2=cmse(pred,ptr); l2.backward(); opt2.step()
    with torch.no_grad():
        a=coeff.effective(); test=cmse(a*Pte,pte).item()
        rel=((a*Pte-pte).abs()/(pte.abs()+1e-12)).mean().item()

    print('PHYSICS: free-particle Schrodinger plane wave')
    print('psi=A exp(i(k x - omega t)), hbar=1,m=1/2 => omega=k^2')
    print('Product inputs X=e^x, T=e^t => psi=A X^(ik) T^(-i omega)')
    print('all four exponent channels exactly 1:', all(torch.equal(c,torch.ones_like(c)) for c in init_channels))
    print('initial A=',A0.tolist(),'initial B=',B0.tolist(),'initial P=1')
    for r in logs: print('step=%4d loss=%.3e maxAerr=%.3e maxBerr=%.3e'%r)
    print('true effective A=',trueA.tolist(),'learned A=',A.tolist())
    print('true effective B=',trueB.tolist(),'learned B=',B.tolist())
    print('true coefficient=',amp.item(),'learned coefficient=',a.item())
    print('held-out complex MSE=%.3e mean relative error=%.3e'%(test,rel))

    Path(__file__).with_name('schrodinger_plane_wave_result.txt').write_text(
        '\n'.join([
            'psi=A exp(i(kx-omega t)); X=e^x,T=e^t => psi=A X^(ik) T^(-i omega)',
            f'k={k} omega={omega}',
            f'true A={trueA.tolist()} learned A={A.tolist()}',
            f'true B={trueB.tolist()} learned B={B.tolist()}',
            f'true coeff={amp.item()} learned coeff={a.item()}',
            f'test_mse={test:.6e} mean_relative_error={rel:.6e}',
        ]),encoding='utf-8')

if __name__=='__main__': main()
