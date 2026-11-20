import numpy as np, pandas as pd, re, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({"font.family":"serif","font.serif":["DejaVu Serif"],"font.size":9,"axes.spines.top":False,"axes.spines.right":False,
 "axes.edgecolor":"#8a8985","axes.labelcolor":"#33322e","xtick.color":"#55544f","ytick.color":"#55544f","axes.grid":True,
 "grid.color":"#e4e3df","grid.linewidth":0.6,"axes.axisbelow":True,"pdf.fonttype":42})
N=["Qwen2.5-1.5B","Llama-3.2-1B","Gemma-2-2B"]; COL=dict(zip(N,["#2a78d6","#eb6834","#1baf7a"])); MRK=dict(zip(N,["o","s","^"]))
def save(f,n): f.savefig(f"figs/{n}.png",dpi=300,bbox_inches="tight"); plt.close(f)
def toplegend(fig,extra_h=[],extra_l=[],y=1.07,ncol=None):
    h=[plt.Rectangle((0,0),1,1,color=COL[n]) for n in N]+extra_h; l=N+extra_l
    fig.legend(h,l,loc="upper center",ncol=ncol or len(l),frameon=False,fontsize=7.5,bbox_to_anchor=(0.5,y))
# ---- Fig 1: Exp1 (stage3 verified)
fert={"Latin":[(3.091,3.072,3.112),(3.153,3.132,3.175),(2.748,2.731,2.767)],"Cyrillic":[(4.122,4.105,4.139),(4.022,4.005,4.040),(3.423,3.410,3.436)]}
frag={"Latin":[(0.002,0.001,0.004),(8.429,8.215,8.622),(0,0,0)],"Cyrillic":[(3.285,3.202,3.378),(14.320,14.158,14.489),(0,0,0)]}
fig,axes=plt.subplots(1,2,figsize=(7.2,3.0))
for ax,data,yl,fmt,title in [(axes[0],fert,"Fertility (tokens per word)","{:.2f}","(a) Fertility"),(axes[1],frag,"Byte-fragment tokens (%)","{:.1f}","(b) Byte-fragment rate")]:
    x=np.arange(2); w=0.26; top=max(v[2] for s in data for v in data[s])
    for i,n in enumerate(N):
        e=[data[s][i] for s in ["Latin","Cyrillic"]]; xp=x+(i-1)*(w+0.02); est=np.array([a[0] for a in e])
        ax.bar(xp,est,w,color=COL[n],edgecolor="white",linewidth=0.6)
        ax.errorbar(xp,est,yerr=[[a[0]-a[1] for a in e],[a[2]-a[0] for a in e]],fmt="none",ecolor="#33322e",elinewidth=0.8,capsize=2)
        for xi,a in zip(xp,e): ax.text(xi,a[2]+0.015*top,fmt.format(a[0]),ha="center",va="bottom",fontsize=6.5,color="#33322e")
    ax.set_xticks(x); ax.set_xticklabels(["Latin","Cyrillic"]); ax.set_ylabel(yl); ax.set_ylim(0,top*1.13); ax.grid(axis="x",visible=False); ax.set_title(title,loc="left")
toplegend(fig); fig.tight_layout(); save(fig,"fig1_fertility_fragments")
# ---- Fig 2: script ratio (stage4b verified)
obs=[1.333,1.276,1.245]; c1=[(1.298,1.295,1.300),(1.234,1.231,1.237),(1.209,1.206,1.211)]; c2=[(1.307,1.303,1.311),(1.245,1.241,1.250),(1.219,1.215,1.223)]
fig,ax=plt.subplots(figsize=(6.4,3.0)); cats=["Observational\n(different documents)","Controlled:\nCyrillic docs → Latin","Controlled:\nLatin docs → Cyrillic"]
for i,n in enumerate(N):
    x=np.arange(3)+(i-1)*0.2; y=[obs[i],c1[i][0],c2[i][0]]; lo=[0,c1[i][0]-c1[i][1],c2[i][0]-c2[i][1]]; hi=[0,c1[i][2]-c1[i][0],c2[i][2]-c2[i][0]]
    ax.errorbar(x,y,yerr=[lo,hi],fmt=MRK[n],color=COL[n],ms=5.5,capsize=2,elinewidth=0.8)
    for xi,yi in zip(x,y): ax.text(xi+0.05,yi,f"{yi:.2f}",fontsize=6.5,va="center",color="#33322e")
ax.axhline(1,color="#8a8985",lw=0.8,ls="--"); ax.set_xticks(range(3)); ax.set_xticklabels(cats,fontsize=8); ax.set_ylim(0.98,1.36)
ax.set_ylabel("Cyrillic / Latin fertility"); ax.grid(axis="x",visible=False)
fig.legend([Line2D([],[],marker=MRK[n],color=COL[n],ls="none") for n in N],N,loc="upper center",ncol=3,frameon=False,fontsize=7.5,bbox_to_anchor=(0.5,1.07))
fig.tight_layout(); save(fig,"fig2_script_ratio")
# ---- Fig 3: apostrophes (stage4b verified, Latin docs)
V=["ʻ ʼ\nofficial","'\nASCII","‘ ’\ntypographic","’\nright quote","`\nbacktick"]
ap={"Qwen2.5-1.5B":[(0,0,0),(-1.16,-1.22,-1.10),(-0.09,-0.11,-0.08),(-0.65,-0.70,-0.61),(-0.15,-0.17,-0.13)],
    "Llama-3.2-1B":[(0,0,0),(-5.30,-5.44,-5.16),(-4.33,-4.42,-4.22),(-5.16,-5.29,-5.02),(-4.37,-4.47,-4.26)],
    "Gemma-2-2B":[(0,0,0)]*5}
fig,ax=plt.subplots(figsize=(6.4,3.0))
for i,n in enumerate(N):
    x=np.arange(5)+(i-1)*0.2; y=np.array([a[0] for a in ap[n]])
    ax.errorbar(x,y,yerr=[[a[0]-a[1] for a in ap[n]],[a[2]-a[0] for a in ap[n]]],fmt=MRK[n],color=COL[n],ms=5.5,capsize=2,elinewidth=0.8)
    for xi,yi in zip(x[1:],y[1:]):
        if abs(yi)>=0.5: ax.text(xi+0.06,yi,f"{yi:.1f}",fontsize=6.5,va="center",color="#33322e")
ax.axhline(0,color="#8a8985",lw=0.8,ls="--"); ax.set_xticks(range(5)); ax.set_xticklabels(V,fontsize=7.5)
ax.set_ylabel("Change in fertility vs official mark (%)"); ax.grid(axis="x",visible=False)
fig.legend([Line2D([],[],marker=MRK[n],color=COL[n],ls="none") for n in N],N,loc="upper center",ncol=3,frameon=False,fontsize=7.5,bbox_to_anchor=(0.5,1.07))
fig.tight_layout(); save(fig,"fig3_apostrophes")
# ---- Fig 4: morphology (stage5c v2 verified)
f1={"Latin":[(35.9,32.9,38.4,25.0),(36.8,33.8,39.5,24.8),(37.8,35.2,40.5,23.8)],"Cyrillic":[(32.7,30.4,34.6,29.7),(33.8,31.5,36.1,29.0),(35.8,33.1,38.3,27.9)]}
tok={"Latin":{"Qwen2.5-1.5B":[2.58,3.29,4.03,5.07,5.67],"Llama-3.2-1B":[2.54,3.26,4.09,5.10,5.94],"Gemma-2-2B":[2.26,2.94,3.59,4.52,5.11]},
     "Cyrillic":{"Qwen2.5-1.5B":[3.19,4.33,5.17,6.64,7.65],"Llama-3.2-1B":[3.08,4.16,4.94,6.52,7.24],"Gemma-2-2B":[2.67,3.45,4.29,5.48,5.88]}}
fig,axes=plt.subplots(1,2,figsize=(7.2,3.0)); ax=axes[0]; x=np.arange(2); w=0.26
for i,n in enumerate(N):
    e=[f1[s][i] for s in ["Latin","Cyrillic"]]; xp=x+(i-1)*(w+0.02); est=np.array([a[0] for a in e])
    ax.bar(xp,est,w,color=COL[n],edgecolor="white",linewidth=0.6)
    ax.errorbar(xp,est,yerr=[[a[0]-a[1] for a in e],[a[2]-a[0] for a in e]],fmt="none",ecolor="#33322e",elinewidth=0.8,capsize=2)
    ax.scatter(xp,[a[3] for a in e],marker="_",s=150,color="#33322e",linewidths=1.6,zorder=3)
    for xi,a in zip(xp,e): ax.text(xi,a[2]+1.2,f"{a[0]:.1f}",ha="center",fontsize=6.5,color="#33322e")
ax.set_xticks(x); ax.set_xticklabels(["Latin","Cyrillic"]); ax.set_ylabel("Boundary F1 (%)"); ax.set_ylim(0,50); ax.grid(axis="x",visible=False)
ax.set_title("(a) Morpheme-boundary F1",loc="left")
ax=axes[1]
for s,ls in [("Latin","-"),("Cyrillic","--")]:
    for n in N: ax.plot(range(1,6),tok[s][n],ls=ls,marker=MRK[n],color=COL[n],ms=4,lw=1.1,mfc="white" if s=="Cyrillic" else COL[n])
ax.plot([1,5],[1,5],ls=":",color="#8a8985",lw=0.9); ax.text(2.0,1.3,"1 token per morpheme",fontsize=6.5,color="#8a8985")
ax.set_xticks(range(1,6)); ax.set_xlabel("Morphemes per word (gold)"); ax.set_ylabel("Mean tokens per word"); ax.set_title("(b) Fragmentation vs morphology",loc="left")
toplegend(fig,[Line2D([],[],color="#33322e",marker="_",ls="none",ms=10,mew=1.6),Line2D([],[],color="#55544f",ls="-"),Line2D([],[],color="#55544f",ls="--")],["chance level","Latin","Cyrillic"],ncol=6)
fig.tight_layout(); save(fig,"fig4_morphology")
# ---- Fig 5 & 6: domains / registers (stage3 cell-8 output, means)
txt=open("../results/stage3_exp4_domain_register_log.txt").read()
def parse(block):
    rows=[]
    for line in block.strip().splitlines()[2:]:
        m=re.match(r"\s*(.+?)\s+(Latin|Cyrillic)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s",line)
        if m: rows.append(dict(group=m.group(1).strip(),script=m.group(2),n=int(m.group(3)),**{N[k]:float(m.group(4+k)) for k in range(3)}))
    return pd.DataFrame(rows)
dom=parse(txt.split("=== DOMAIN")[1].split("DOMAIN —")[0]); reg=parse(txt.split("=== REGISTER")[1].split("REGISTER —")[0])
assert len(dom)==44 and len(reg)==16, (len(dom),len(reg))
dom.to_csv("figs/domain_means.csv",index=False); reg.to_csv("figs/register_means.csv",index=False)
def dotplot(d,name,h):
    order=d.groupby("group")[N[0]].mean().sort_values().index.tolist()
    fig,axes=plt.subplots(1,2,figsize=(7.2,h),sharey=True)
    for ax,s in zip(axes,["Latin","Cyrillic"]):
        x=d[d.script==s].set_index("group").reindex(order)
        for i,n in enumerate(N): ax.plot(x[n],np.arange(len(order))+(1-i)*0.22,MRK[n],color=COL[n],ms=4)
        ax.set_yticks(np.arange(len(order))); ax.set_yticklabels(order); ax.set_title(s,loc="left"); ax.set_xlabel("Fertility (tokens per word)"); ax.grid(axis="y",visible=False)
    fig.legend([Line2D([],[],marker=MRK[n],color=COL[n],ls="none") for n in N],N,loc="upper center",ncol=3,frameon=False,fontsize=7.5,bbox_to_anchor=(0.5,1.03))
    fig.tight_layout(); save(fig,name)
dotplot(dom,"fig5_domains",6.3); dotplot(reg,"fig6_registers",3.0)
print("ok")
