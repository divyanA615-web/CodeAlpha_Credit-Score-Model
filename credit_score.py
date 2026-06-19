
import os, gc, json, time, warnings
from pathlib import Path
from datetime import datetime
warnings.filterwarnings('ignore')

import numpy  as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing   import LabelEncoder
from sklearn.metrics         import f1_score, accuracy_score, classification_report, confusion_matrix

import lightgbm  as lgb
import xgboost   as xgb
from catboost   import CatBoostClassifier
import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
from optuna.samplers import TPESampler
import shap, joblib

SEED = 42; np.random.seed(SEED)
OUT  = Path("D:/data science related/Internships/M_L/CodeAlpha_Credit-Score-Model/train_output")
for s in ["models","charts","reports"]: (OUT/s).mkdir(parents=True, exist_ok=True)
LABEL_MAP = {0:"Poor", 1:"Standard", 2:"Good"}
COLORS    = ["#e74c3c","#f39c12","#2ecc71"]

def log(m, l="INFO"):
    icons={"INFO":"📋","OK":"✅","WARN":"⚠️","MODEL":"🤖","CHART":"📊","SAVE":"💾"}
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {icons.get(l,'•')} {m}", flush=True)

# ── LOAD ─────────────────────────────────────────────────────────────────────
log("Loading data...")
df = pd.read_csv("D:/data science related/Internships/M_L/CodeAlpha_Credit-Score-Model/train_data.csv")
log(f"Shape: {df.shape}", "OK")

# ── FEATURE ENGINEERING ──────────────────────────────────────────────────────
log("Engineering features...")
def sdiv(a,b): return np.where(np.abs(b)>1e-9, a/b, 0.0)

def engineer(df):
    df = df.copy()
    month_map={m:i for i,m in enumerate(["January","February","March","April","May","June",
               "July","August","September","October","November","December"],1)}
    if "month" in df.columns:
        mn=df["month"].map(month_map).fillna(0).astype(float)
        df["month_sin"]=np.sin(2*np.pi*mn/12); df["month_cos"]=np.cos(2*np.pi*mn/12)
        df.drop(columns=["month"], inplace=True)
    
    loan_kw={"loan_auto":"auto loan","loan_cb":"credit-builder loan",
             "loan_home":"home equity loan","loan_mort":"mortgage loan",
             "loan_pers":"personal loan","loan_pay":"payday loan","loan_stu":"student loan"}
    if "type_of_loan" in df.columns:
        tl=df["type_of_loan"].fillna("").astype(str).str.lower()
        for feat,kw in loan_kw.items(): df[feat]=tl.str.contains(kw).astype(float)
        df["num_loan_types"]=df[list(loan_kw.keys())].sum(axis=1)
        df.drop(columns=["type_of_loan"], inplace=True)
    
    df["debt_to_income"]     = sdiv(df["outstanding_debt"],    df["annual_income"])
    df["emi_burden"]         = sdiv(df["total_emi_per_month"], df["monthly_inhand_salary"])
    df["savings_rate"]       = sdiv(df["amount_invested_monthly"], df["monthly_inhand_salary"])
    df["balance_ratio"]      = sdiv(df["monthly_balance"],     df["monthly_inhand_salary"])
    df["available_cash"]     = df["monthly_inhand_salary"] - df["total_emi_per_month"]
    df["net_surplus"]        = df["monthly_balance"]       - df["total_emi_per_month"]
    df["income_per_loan"]    = sdiv(df["annual_income"],   df["num_of_loan"]+1)
    df["debt_per_card"]      = sdiv(df["outstanding_debt"],df["num_credit_card"]+1)
    df["credit_age_yrs"]     = df["credit_history_age"] / 12.0
    df["card_per_account"]   = sdiv(df["num_credit_card"],  df["num_bank_accounts"]+1)
    df["loan_per_account"]   = sdiv(df["num_of_loan"],      df["num_bank_accounts"]+1)
    df["total_credit_lines"] = df["num_credit_card"]+df["num_bank_accounts"]+df["num_of_loan"]
    df["delay_severity"]     = df["delay_from_due_date"] * df["num_of_delayed_payment"]
    df["inquiry_rate"]       = sdiv(df["num_credit_inquiries"], df["credit_age_yrs"]+0.1)
    df["util_x_debt"]        = df["credit_utilization_ratio"]*df["outstanding_debt"]/1e4
    df["delay_per_loan"]     = sdiv(df["num_of_delayed_payment"],df["num_of_loan"]+1)
    df["payment_missed_r"]   = sdiv(df["num_of_delayed_payment"],df["credit_age_yrs"]+1)
    df["high_risk_flag"]     = ((df["credit_utilization_ratio"]>70)&(df["num_of_delayed_payment"]>10)).astype(float)
    df["age_x_credit_age"]   = df["age"]*df["credit_age_yrs"]
    df["util_x_delay"]       = df["credit_utilization_ratio"]*df["delay_from_due_date"]
    df["emi_x_debt"]         = df["emi_burden"]*df["outstanding_debt"]/1e3
    df["debt_income_sq"]     = df["debt_to_income"]**2
    df["emi_burden_sq"]      = df["emi_burden"]**2
    df["util_ratio_sq"]      = df["credit_utilization_ratio"]**2
    df["outstanding_debt_log"]= np.log1p(df["outstanding_debt"])
    df["annual_income_log"]  = np.log1p(df["annual_income"])
    df["balance_log"]        = np.log1p(df["monthly_balance"].clip(0))
    beh_risk={"High_spent_Large_value_payments":5,"High_spent_Medium_value_payments":4,
              "High_spent_Small_value_payments":3,"Low_spent_Large_value_payments":2,
              "Low_spent_Medium_value_payments":1,"Low_spent_Small_value_payments":0}
    if "payment_behaviour" in df.columns:
        df["payment_risk_score"]=df["payment_behaviour"].map(beh_risk).fillna(2).astype(float)
    return df

df_fe = engineer(df)
log(f"Feature shape: {df_fe.shape}", "OK")

# ── PREPROCESS ───────────────────────────────────────────────────────────────
log("Preprocessing...")
drop_cols=["id","customer_id","name","ssn"]
TARGET="credit_score"
df_fe.drop(columns=[c for c in drop_cols if c in df_fe.columns], inplace=True)

obj_cols=[c for c in df_fe.select_dtypes(include="object").columns if c!=TARGET]
for col in obj_cols:
    le=LabelEncoder()
    le.fit(df_fe[col].fillna("__unk__").astype(str))
    df_fe[col]=le.transform(df_fe[col].fillna("__unk__").astype(str))

num_cols=[c for c in df_fe.select_dtypes(include="number").columns if c!=TARGET]
med=df_fe[num_cols].median()
df_fe[num_cols]=df_fe[num_cols].fillna(med)
for col in num_cols:
    lo,hi=df_fe[col].quantile(0.01),df_fe[col].quantile(0.99)
    df_fe[col]=df_fe[col].clip(lo,hi)

y = df_fe[TARGET]
X = df_fe.drop(columns=[TARGET])
feat_cols=X.columns.tolist()
log(f"Features: {len(feat_cols)} | Classes: {dict(y.value_counts().sort_index())}", "OK")

# ── EDA DASHBOARD ─────────────────────────────────────────────────────────────
log("Building EDA dashboard...", "CHART")
fig=plt.figure(figsize=(22,14)); fig.patch.set_facecolor("#0f1117")
gs=gridspec.GridSpec(2,4,figure=fig,hspace=0.45,wspace=0.40)

raw_df=pd.read_csv("D:/data science related/Internships/M_L/CodeAlpha_Credit-Score-Model/train_data.csv")  # use raw for EDA

def dax(ax):
    ax.set_facecolor("#1a1d27"); ax.tick_params(colors="#cccccc",labelsize=9)
    [s.set_edgecolor("#333344") for s in ax.spines.values()]
    ax.title.set_color("#e0e0e0"); ax.xaxis.label.set_color("#aaa"); ax.yaxis.label.set_color("#aaa")
    return ax

ax1=dax(fig.add_subplot(gs[0,0]))
vc=raw_df[TARGET].value_counts().sort_index()
bars=ax1.bar([LABEL_MAP[k] for k in vc.index],vc.values,color=COLORS,width=0.55)
for bar,v in zip(bars,vc.values):
    ax1.text(bar.get_x()+bar.get_width()/2,bar.get_height()+50,f"{v:,}\n({v/len(raw_df)*100:.1f}%)",
             ha="center",fontsize=9,color="white",fontweight="bold")
ax1.set_title("Target Class Distribution",fontweight="bold")

ax2=dax(fig.add_subplot(gs[0,1]))
for cs,c,l in zip([0,1,2],COLORS,["Poor","Standard","Good"]):
    ax2.hist(raw_df[raw_df[TARGET]==cs]["annual_income"].dropna().clip(0,200000),
             alpha=0.65,label=l,color=c,bins=35); ax2.set_title("Annual Income by Class",fontweight="bold")
ax2.legend(fontsize=8,facecolor="#1a1d27",labelcolor="white")

ax3=dax(fig.add_subplot(gs[0,2]))
for cs,c,l in zip([0,1,2],COLORS,["Poor","Standard","Good"]):
    ax3.hist(raw_df[raw_df[TARGET]==cs]["outstanding_debt"].dropna().clip(0,4500),
             alpha=0.65,label=l,color=c,bins=35); ax3.set_title("Outstanding Debt",fontweight="bold")
ax3.legend(fontsize=8,facecolor="#1a1d27",labelcolor="white")

ax4=dax(fig.add_subplot(gs[0,3]))
for cs,c,l in zip([0,1,2],COLORS,["Poor","Standard","Good"]):
    ax4.hist(raw_df[raw_df[TARGET]==cs]["credit_utilization_ratio"].dropna(),
             alpha=0.65,label=l,color=c,bins=35); ax4.set_title("Credit Utilization Ratio",fontweight="bold")
ax4.legend(fontsize=8,facecolor="#1a1d27",labelcolor="white")

num_all=[c for c in raw_df.select_dtypes(include="number").columns]
top9=(raw_df[num_all].corr()[TARGET].abs().sort_values(ascending=False).head(9).index.tolist())
ax5=dax(fig.add_subplot(gs[1,:2]))
mask=np.triu(np.ones_like(raw_df[top9].corr(),dtype=bool))
sns.heatmap(raw_df[top9].corr(),mask=mask,annot=True,fmt=".2f",cmap="coolwarm",
            center=0,ax=ax5,linewidths=0.3,linecolor="#333344",annot_kws={"size":8})
ax5.set_title("Correlation Heatmap (Top Predictors)",fontweight="bold")
ax5.tick_params(axis="x",rotation=30,colors="#ccc"); ax5.tick_params(axis="y",rotation=0,colors="#ccc")

ax6=dax(fig.add_subplot(gs[1,2]))
for cs,c,l in zip([0,1,2],COLORS,["Poor","Standard","Good"]):
    ax6.hist(raw_df[raw_df[TARGET]==cs]["num_of_delayed_payment"].dropna(),
             alpha=0.65,label=l,color=c,bins=25); ax6.set_title("# Delayed Payments",fontweight="bold")
ax6.legend(fontsize=8,facecolor="#1a1d27",labelcolor="white")

ax7=dax(fig.add_subplot(gs[1,3]))
for cs,c,l in zip([0,1,2],COLORS,["Poor","Standard","Good"]):
    ax7.hist(raw_df[raw_df[TARGET]==cs]["monthly_balance"].dropna().clip(0,8000),
             alpha=0.65,label=l,color=c,bins=35); ax7.set_title("Monthly Balance",fontweight="bold")
ax7.legend(fontsize=8,facecolor="#1a1d27",labelcolor="white")

fig.suptitle("🏆 CREDIT SCORE — GOD-LEVEL EDA DASHBOARD",fontsize=16,fontweight="bold",color="#00d4ff",y=1.01)
plt.savefig(OUT/"charts"/"01_eda_dashboard.png",dpi=150,bbox_inches="tight",facecolor="#0f1117"); plt.close()
log("EDA dashboard saved", "OK")

# ── OPTUNA HPO (fast: 10 trials each) ────────────────────────────────────────
def quick_cv(model, X, y, n=3):
    skf=StratifiedKFold(n_splits=n,shuffle=True,random_state=SEED); scores=[]
    for tr,va in skf.split(X,y):
        m=model(); m.fit(X.iloc[tr],y.iloc[tr])
        scores.append(f1_score(y.iloc[va],m.predict(X.iloc[va]),average="weighted"))
    return float(np.mean(scores))

log("Optuna HPO — LightGBM (12 trials)...", "MODEL")
def lgb_obj(trial):
    p=dict(objective="multiclass",num_class=3,verbosity=-1,random_state=SEED,
           n_estimators=trial.suggest_int("n_estimators",200,800),
           num_leaves=trial.suggest_int("num_leaves",31,128),
           max_depth=trial.suggest_int("max_depth",4,10),
           learning_rate=trial.suggest_float("learning_rate",0.01,0.15,log=True),
           subsample=trial.suggest_float("subsample",0.6,1.0),
           colsample_bytree=trial.suggest_float("colsample_bytree",0.6,1.0),
           reg_alpha=trial.suggest_float("reg_alpha",1e-6,5.0,log=True),
           reg_lambda=trial.suggest_float("reg_lambda",1e-6,5.0,log=True))
    return quick_cv(lambda: lgb.LGBMClassifier(**p), X, y)

s1=optuna.create_study(direction="maximize",sampler=TPESampler(seed=SEED))
s1.optimize(lgb_obj, n_trials=12, show_progress_bar=False)
lgb_p=s1.best_params; log(f"LGB best F1: {s1.best_value:.4f}", "OK")

log("Optuna HPO — XGBoost (10 trials)...", "MODEL")
def xgb_obj(trial):
    p=dict(objective="multi:softprob",num_class=3,verbosity=0,random_state=SEED,tree_method="hist",
           n_estimators=trial.suggest_int("n_estimators",200,700),
           max_depth=trial.suggest_int("max_depth",3,8),
           learning_rate=trial.suggest_float("learning_rate",0.01,0.15,log=True),
           subsample=trial.suggest_float("subsample",0.6,1.0),
           colsample_bytree=trial.suggest_float("colsample_bytree",0.6,1.0),
           gamma=trial.suggest_float("gamma",0.0,3.0),
           reg_alpha=trial.suggest_float("reg_alpha",1e-6,5.0,log=True),
           reg_lambda=trial.suggest_float("reg_lambda",1e-6,5.0,log=True))
    return quick_cv(lambda: xgb.XGBClassifier(**p), X, y)

s2=optuna.create_study(direction="maximize",sampler=TPESampler(seed=SEED))
s2.optimize(xgb_obj, n_trials=10, show_progress_bar=False)
xgb_p=s2.best_params; log(f"XGB best F1: {s2.best_value:.4f}", "OK")

log("Optuna HPO — CatBoost (8 trials)...", "MODEL")
def cat_obj(trial):
    p=dict(loss_function="MultiClass",classes_count=3,random_seed=SEED,verbose=False,
           iterations=trial.suggest_int("iterations",150,500),
           depth=trial.suggest_int("depth",4,8),
           learning_rate=trial.suggest_float("learning_rate",0.02,0.2,log=True),
           l2_leaf_reg=trial.suggest_float("l2_leaf_reg",1e-6,10.0,log=True))
    return quick_cv(lambda: CatBoostClassifier(**p), X, y)

s3=optuna.create_study(direction="maximize",sampler=TPESampler(seed=SEED))
s3.optimize(cat_obj, n_trials=8, show_progress_bar=False)
cat_p=s3.best_params; log(f"CAT best F1: {s3.best_value:.4f}", "OK")

# ── 5-FOLD CV TRAINING ────────────────────────────────────────────────────────
log("="*58); log("5-Fold Stratified CV Training...", "MODEL")
skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
models={"lgb":[],"xgb":[],"cat":[]}
oof={"lgb":np.zeros((len(X),3)),"xgb":np.zeros((len(X),3)),"cat":np.zeros((len(X),3))}
fold_scores=[]

for fold,(tr_idx,va_idx) in enumerate(skf.split(X,y),1):
    X_tr,X_va=X.iloc[tr_idx],X.iloc[va_idx]
    y_tr,y_va=y.iloc[tr_idx],y.iloc[va_idx]
    row={"fold":fold}

    # LGB
    lm=lgb.LGBMClassifier(**lgb_p,objective="multiclass",num_class=3,verbosity=-1,random_state=SEED)
    lm.fit(X_tr,y_tr,eval_set=[(X_va,y_va)],callbacks=[lgb.early_stopping(60,verbose=False),lgb.log_evaluation(-1)])
    oof["lgb"][va_idx]=lm.predict_proba(X_va)
    row["lgb"]=f1_score(y_va,lm.predict(X_va),average="weighted"); models["lgb"].append(lm)

    # XGB
    xm=xgb.XGBClassifier(**xgb_p,objective="multi:softprob",num_class=3,verbosity=0,random_state=SEED,tree_method="hist",early_stopping_rounds=60)
    xm.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],verbose=False)
    oof["xgb"][va_idx]=xm.predict_proba(X_va)
    row["xgb"]=f1_score(y_va,xm.predict(X_va),average="weighted"); models["xgb"].append(xm)

    # CAT
    cm=CatBoostClassifier(**cat_p,loss_function="MultiClass",classes_count=3,random_seed=SEED,verbose=False)
    cm.fit(X_tr,y_tr,eval_set=(X_va,y_va),early_stopping_rounds=60,verbose=False)
    oof["cat"][va_idx]=cm.predict_proba(X_va)
    row["cat"]=f1_score(y_va,cm.predict(X_va),average="weighted"); models["cat"].append(cm)

    fold_scores.append(row)
    log(f"  Fold {fold}/5 → LGB {row['lgb']:.4f} | XGB {row['xgb']:.4f} | CAT {row['cat']:.4f}")

ens_proba=(oof["lgb"]+oof["xgb"]+oof["cat"])/3
ens_preds=ens_proba.argmax(axis=1)
ens_f1=f1_score(y,ens_preds,average="weighted")
ens_acc=accuracy_score(y,ens_preds)
log("="*58)
log(f"LightGBM  CV F1: {np.mean([r['lgb'] for r in fold_scores]):.4f} ± {np.std([r['lgb'] for r in fold_scores]):.4f}", "OK")
log(f"XGBoost   CV F1: {np.mean([r['xgb'] for r in fold_scores]):.4f} ± {np.std([r['xgb'] for r in fold_scores]):.4f}", "OK")
log(f"CatBoost  CV F1: {np.mean([r['cat'] for r in fold_scores]):.4f} ± {np.std([r['cat'] for r in fold_scores]):.4f}", "OK")
log(f"★ Ensemble OOF F1: {ens_f1:.4f}  |  Accuracy: {ens_acc:.4f}", "OK")
log("="*58)
print()
print(classification_report(y, ens_preds, target_names=["Poor","Standard","Good"]))

# ── PERFORMANCE DASHBOARD ─────────────────────────────────────────────────────
log("Building performance dashboard...", "CHART")
labs=["Poor","Standard","Good"]
fig2=plt.figure(figsize=(22,12)); fig2.patch.set_facecolor("#0f1117")
gs2=gridspec.GridSpec(2,4,figure=fig2,hspace=0.45,wspace=0.40)

ax=dax(fig2.add_subplot(gs2[0,0]))
cm_m=confusion_matrix(y,ens_preds)
cm_n=cm_m.astype(float)/cm_m.sum(axis=1,keepdims=True)
sns.heatmap(cm_n,annot=True,fmt=".1%",cmap="Blues",xticklabels=labs,yticklabels=labs,
            ax=ax,linewidths=0.5); ax.set_title("Confusion Matrix (OOF)",fontweight="bold")
ax.set_ylabel("True"); ax.set_xlabel("Predicted")

ax=dax(fig2.add_subplot(gs2[0,1]))
rpt=classification_report(y,ens_preds,target_names=labs,output_dict=True)
x=np.arange(3); w=0.25
ax.bar(x-w,[rpt[l]["f1-score"]  for l in labs],w,label="F1",  color="#3498db")
ax.bar(x,  [rpt[l]["precision"] for l in labs],w,label="Prec",color="#2ecc71")
ax.bar(x+w,[rpt[l]["recall"]    for l in labs],w,label="Rec", color="#e74c3c")
ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_ylim(0,1.15)
ax.set_title("Per-Class Metrics",fontweight="bold")
ax.legend(fontsize=8,facecolor="#1a1d27",labelcolor="white")

ax=dax(fig2.add_subplot(gs2[0,2]))
m_names=["LightGBM","XGBoost","CatBoost","Ensemble"]
m_f1s=[np.mean([r["lgb"] for r in fold_scores]),np.mean([r["xgb"] for r in fold_scores]),
       np.mean([r["cat"] for r in fold_scores]),ens_f1]
bar_clrs=["#3498db","#e67e22","#9b59b6","#00d4ff"]
bars=ax.bar(m_names,m_f1s,color=bar_clrs,width=0.55)
ax.set_ylim(max(0,min(m_f1s)-0.05),1.0); ax.set_title("Model F1 Comparison",fontweight="bold")
for b,v in zip(bars,m_f1s):
    ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.003,f"{v:.4f}",
            ha="center",fontsize=9,color="white",fontweight="bold")

ax=dax(fig2.add_subplot(gs2[0,3]))
folds=[r["fold"] for r in fold_scores]
ax.plot(folds,[r["lgb"] for r in fold_scores],"o-",color="#3498db",label="LGB",lw=2)
ax.plot(folds,[r["xgb"] for r in fold_scores],"s-",color="#e67e22",label="XGB",lw=2)
ax.plot(folds,[r["cat"] for r in fold_scores],"^-",color="#9b59b6",label="CAT",lw=2)
ax.set_title("Per-Fold F1 Scores",fontweight="bold"); ax.set_xlabel("Fold"); ax.set_xticks(folds)
ax.legend(fontsize=8,facecolor="#1a1d27",labelcolor="white")

ax=dax(fig2.add_subplot(gs2[1,0]))
conf=ens_proba.max(axis=1)
ax.hist(conf,bins=60,color="#00d4ff",edgecolor="none",alpha=0.85)
ax.axvline(conf.mean(),color="#e74c3c",ls="--",lw=1.5,label=f"Mean={conf.mean():.3f}")
ax.set_title("Prediction Confidence",fontweight="bold"); ax.set_xlabel("Max Probability")
ax.legend(fontsize=8,facecolor="#1a1d27",labelcolor="white")

ax=dax(fig2.add_subplot(gs2[1,1]))
for cs,c,l in zip([0,1,2],COLORS,labs):
    mask_c=np.array(y)==cs; probs=ens_proba[mask_c,cs]
    ax.hist(probs,bins=40,alpha=0.65,color=c,label=l)
ax.set_title("P(correct class) by True Class",fontweight="bold"); ax.set_xlabel("Probability")
ax.legend(fontsize=8,facecolor="#1a1d27",labelcolor="white")

ax=dax(fig2.add_subplot(gs2[1,2]))
x=np.arange(3); w=0.4
ax.bar(x-0.2,[sum(np.array(y)==cs) for cs in [0,1,2]],w,label="Actual",color="#3498db",alpha=0.85)
ax.bar(x+0.2,[sum(ens_preds==cs)    for cs in [0,1,2]],w,label="Predicted",color="#e74c3c",alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_title("Actual vs Predicted Counts",fontweight="bold")
ax.legend(fontsize=8,facecolor="#1a1d27",labelcolor="white")

ax=dax(fig2.add_subplot(gs2[1,3]))
means=[np.mean([r[k] for r in fold_scores]) for k in ["lgb","xgb","cat"]]
stds =[np.std( [r[k] for r in fold_scores]) for k in ["lgb","xgb","cat"]]
ax.bar(["LightGBM","XGBoost","CatBoost"],means,yerr=stds,capsize=6,
       color=["#3498db","#e67e22","#9b59b6"],error_kw={"color":"white","elinewidth":2})
ax.set_title("CV F1 ± Std Dev (Stability)",fontweight="bold"); ax.set_ylabel("Weighted F1")

fig2.suptitle("🏆 CREDIT SCORE — PERFORMANCE DASHBOARD",fontsize=16,fontweight="bold",color="#00d4ff",y=1.01)
plt.savefig(OUT/"charts"/"02_performance_dashboard.png",dpi=150,bbox_inches="tight",facecolor="#0f1117"); plt.close()
log("Performance dashboard saved", "OK")

# ── FEATURE IMPORTANCE ────────────────────────────────────────────────────────
log("Plotting feature importance...", "CHART")
importances=np.zeros(len(feat_cols))
for m in models["lgb"]: importances+=m.feature_importances_
importances/=len(models["lgb"])
fi=pd.DataFrame({"feature":feat_cols,"importance":importances}).sort_values("importance",ascending=True).tail(28)

fig3,ax=plt.subplots(figsize=(12,14)); fig3.patch.set_facecolor("#0f1117"); ax.set_facecolor("#1a1d27")
grad=plt.cm.plasma(np.linspace(0.2,0.95,len(fi)))
ax.barh(fi["feature"],fi["importance"],color=grad,edgecolor="none")
ax.set_title("Top-28 Feature Importance (LightGBM Gain)",fontsize=13,fontweight="bold",color="#e0e0e0",pad=12)
ax.tick_params(colors="#cccccc"); ax.set_xlabel("Gain",color="#aaaaaa")
[s.set_edgecolor("#333344") for s in ax.spines.values()]
plt.tight_layout()
plt.savefig(OUT/"charts"/"03_feature_importance.png",dpi=150,bbox_inches="tight",facecolor="#0f1117"); plt.close()
log("Feature importance saved", "OK")

# ── SHAP ──────────────────────────────────────────────────────────────────────
log("Running SHAP analysis...", "CHART")
try:
    best_lgb=models["lgb"][-1]
    X_samp=X.sample(min(1500,len(X)),random_state=SEED)
    explainer=shap.TreeExplainer(best_lgb)
    sv=explainer.shap_values(X_samp)
    
    if isinstance(sv,list):
        sv_mean=np.abs(np.array(sv)).mean(axis=0)
    else:
        sv_mean=np.abs(sv)
    
    shap_df=pd.DataFrame({"feature":feat_cols,"shap":sv_mean.mean(axis=0)}).sort_values("shap",ascending=True).tail(20)
    
    fig4,ax=plt.subplots(figsize=(12,10)); fig4.patch.set_facecolor("#0f1117"); ax.set_facecolor("#1a1d27")
    grad=plt.cm.cool(np.linspace(0.1,0.9,len(shap_df)))
    ax.barh(shap_df["feature"],shap_df["shap"],color=grad,edgecolor="none")
    ax.set_title("Top-20 Features — Mean |SHAP| (All Classes)",fontsize=13,fontweight="bold",color="#e0e0e0",pad=12)
    ax.set_xlabel("Mean |SHAP Value|",color="#aaaaaa"); ax.tick_params(colors="#cccccc")
    [s.set_edgecolor("#333344") for s in ax.spines.values()]
    plt.tight_layout()
    plt.savefig(OUT/"charts"/"04_shap_importance.png",dpi=150,bbox_inches="tight",facecolor="#0f1117"); plt.close()
    
    # SHAP beeswarm class=Good
    sv_good=sv[2] if isinstance(sv,list) else sv
    plt.figure(figsize=(12,9)); plt.gcf().patch.set_facecolor("#0f1117")
    shap.summary_plot(sv_good,X_samp,plot_type="dot",max_display=20,show=False)
    plt.title("SHAP Beeswarm — Class: Good (Credit Score=2)",fontsize=13,fontweight="bold",color="#e0e0e0")
    plt.tight_layout()
    plt.savefig(OUT/"charts"/"05_shap_beeswarm.png",dpi=150,bbox_inches="tight",facecolor="#0f1117"); plt.close()
    log("SHAP charts saved", "OK")
except Exception as e:
    log(f"SHAP error: {e}", "WARN")

# ── SAVE ARTIFACTS ────────────────────────────────────────────────────────────
log("Saving artifacts...", "SAVE")
joblib.dump(models["lgb"][-1], OUT/"models"/"lgbm_best.pkl")
joblib.dump(models["xgb"][-1], OUT/"models"/"xgb_best.pkl")
models["cat"][-1].save_model(str(OUT/"models"/"catboost_best.cbm"))
json.dump({"lgb":lgb_p,"xgb":xgb_p,"cat":cat_p},
          open(OUT/"models"/"best_hyperparams.json","w"), indent=2, default=str)
json.dump(feat_cols, open(OUT/"models"/"feature_columns.json","w"), indent=2)

labs=["Poor","Standard","Good"]
rpt_txt=f"""
╔══════════════════════════════════════════════════════════════════╗
║         🏆 CREDIT SCORE PIPELINE — FINAL REPORT        ║
╚══════════════════════════════════════════════════════════════════╝
Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Features  : {len(feat_cols)} engineered features
CV        : 5-Fold Stratified KFold

══════════════ CROSS-VALIDATION RESULTS ════════════════════════════
LightGBM  F1: {np.mean([r['lgb'] for r in fold_scores]):.4f} ± {np.std([r['lgb'] for r in fold_scores]):.4f}
XGBoost   F1: {np.mean([r['xgb'] for r in fold_scores]):.4f} ± {np.std([r['xgb'] for r in fold_scores]):.4f}
CatBoost  F1: {np.mean([r['cat'] for r in fold_scores]):.4f} ± {np.std([r['cat'] for r in fold_scores]):.4f}

★ Ensemble OOF Weighted-F1 : {ens_f1:.4f}
★ Ensemble OOF Accuracy    : {ens_acc:.4f}

══════════════ CLASSIFICATION REPORT ══════════════════════════════
{classification_report(y,ens_preds,target_names=labs)}

══════════════ CONFUSION MATRIX ════════════════════════════════════
{confusion_matrix(y,ens_preds)}
(rows=True Label, cols=Predicted | 0=Poor 1=Standard 2=Good)

══════════════ TOP-10 FEATURES BY IMPORTANCE ════════════════════════
{fi.sort_values('importance',ascending=False).head(10)[['feature','importance']].to_string(index=False)}

══════════════ ARTIFACTS ════════════════════════════════════════════
models/lgbm_best.pkl   | models/xgb_best.pkl  | models/catboost_best.cbm
models/best_hyperparams.json | models/feature_columns.json
charts/01_eda_dashboard.png  | charts/02_performance_dashboard.png
charts/03_feature_importance.png | charts/04_shap_importance.png | charts/05_shap_beeswarm.png
reports/pipeline_report.txt

"""
(OUT / "reports" / "pipeline_report.txt").write_text(rpt_txt, encoding="utf-8")
log("All artifacts saved!", "OK")

print("OUTPUT FILES:")
for f in sorted((OUT).rglob("*")):
    if f.is_file(): print(f"  ✅ {f.relative_to(OUT)}") 