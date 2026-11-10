# ===============================================================
# 🤖 군집 모델 평가 및 선택
# ===============================================================
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

st.divider()
st.header("🤖 군집 모델 평가 및 선택")
st.caption("선수-시즌 기록을 비슷한 유형끼리 묶는 군집 모델 3가지를 비교하고, 가장 적합한 모델을 선택해 보세요.")


@st.cache_data
def build_cluster_data(player_type):
    """선수 유형(타자/투수)별로 '선수-시즌' 한 행짜리 데이터프레임을 만든다."""
    frames = []
    for n, info in PLAYERS.items():
        if info["유형"] != player_type:
            continue
        d = info["시즌기록"].copy()
        d.insert(0, "선수", n)
        frames.append(d)
    data = pd.concat(frames, ignore_index=True)
    data["라벨"] = data["선수"] + " " + data["시즌"].astype(str)
    return data


def make_model(model_name, k, linkage="ward", cov_type="full"):
    if model_name == "K-Means":
        return KMeans(n_clusters=k, n_init=10, random_state=42)
    if model_name == "계층적 군집":
        return AgglomerativeClustering(n_clusters=k, linkage=linkage)
    return GaussianMixture(n_components=k, covariance_type=cov_type, random_state=42)


def fit_labels(model_name, X, k, linkage="ward", cov_type="full"):
    model = make_model(model_name, k, linkage, cov_type)
    if model_name == "가우시안 혼합(GMM)":
        return model.fit(X).predict(X)
    return model.fit_predict(X)


MODEL_NAMES = ["K-Means", "계층적 군집", "가우시안 혼합(GMM)"]

# make_model 은 "가우시안 혼합(GMM)"도 GMM으로 처리 (else 분기)

left, right = st.columns([1, 2])
with left:
    ptype = st.radio("분석 대상", ["타자", "투수"], horizontal=True, key="cl_type")
    cdata = build_cluster_data(ptype)

    # 0으로 채워진 결측 의심 데이터 제외 옵션
    drop_zero = st.checkbox("WAR·wRC+ 등이 0인(원본 결측 의심) 행 제외", value=True)
    if drop_zero:
        if ptype == "타자":
            cdata = cdata[(cdata["wRC+"] > 0) & (cdata["경기수"] >= 30)]
        else:
            cdata = cdata[(cdata["WAR"] != 0) & (cdata["K/9"] > 0)]
    cdata = cdata.reset_index(drop=True)

    num_cols = [c_ for c_ in cdata.columns if c_ not in ("선수", "시즌", "라벨")]
    default_feat = (["타율", "출루율", "장타율", "홈런", "도루", "WAR"]
                    if ptype == "타자"
                    else ["평균자책", "WHIP", "K/9", "BB/9", "WAR"])
    features = st.multiselect("군집에 사용할 특성", num_cols, default=default_feat,
                              key=f"cl_feat_{ptype}")
    k_range = st.slider("탐색할 군집 수(K) 범위", 2, 8, (2, 6))
    k_final = st.slider("최종 확인할 군집 수(K)", 2, 8, 3)

if len(features) < 2:
    st.info("특성을 2개 이상 선택하세요.")
    st.stop()

if len(cdata) <= k_range[1]:
    st.warning("데이터 수가 너무 적어요. K 범위를 줄이거나 제외 옵션을 해제해 보세요.")
    st.stop()

X = StandardScaler().fit_transform(cdata[features])
st.write(f"사용 데이터: **{len(cdata)}행** (선수-시즌), 특성 **{len(features)}개** (표준화 적용)")

# ---------- 1) 모델별 평가 ----------
st.subheader("1️⃣ 모델별 성능 비교")
rows = []
for mname in MODEL_NAMES:
    for k in range(k_range[0], k_range[1] + 1):
        try:
            lab = fit_labels(mname, X, k)
            if len(set(lab)) < 2:
                continue
            rows.append({
                "모델": mname, "K": k,
                "실루엣(↑)": silhouette_score(X, lab),
                "Davies-Bouldin(↓)": davies_bouldin_score(X, lab),
                "Calinski-Harabasz(↑)": calinski_harabasz_score(X, lab),
            })
        except Exception:
            continue
res = pd.DataFrame(rows)

if res.empty:
    st.error("평가 결과를 계산하지 못했어요. 설정을 바꿔 보세요.")
    st.stop()

st.dataframe(
    res.style.format({"실루엣(↑)": "{:.3f}", "Davies-Bouldin(↓)": "{:.3f}",
                      "Calinski-Harabasz(↑)": "{:.1f}"}),
    use_container_width=True,
)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, metric in zip(axes, ["실루엣(↑)", "Davies-Bouldin(↓)", "Calinski-Harabasz(↑)"]):
    for mname in MODEL_NAMES:
        sub = res[res["모델"] == mname]
        ax.plot(sub["K"], sub[metric], marker="o", label=mname)
    ax.set_title(metric)
    ax.set_xlabel("군집 수 K")
    ax.grid(alpha=0.3)
axes[0].legend()
fig.tight_layout()
st.pyplot(fig)

# 실루엣 기준 추천
best = res.sort_values("실루엣(↑)", ascending=False).iloc[0]
st.success(f"📌 실루엣 계수 기준 추천: **{best['모델']} (K={int(best['K'])})**, "
           f"실루엣 = {best['실루엣(↑)']:.3f}")
st.caption("※ 지표마다 '좋다'고 하는 모델이 다를 수 있어요. 한 지표만 보지 말고 "
           "아래 시각화와 군집의 의미까지 함께 판단해 보세요.")

# ---------- 2) 모델 선택 & 시각화 ----------
st.subheader("2️⃣ 모델 선택 및 군집 확인")
chosen_model = st.selectbox("사용할 모델 선택", MODEL_NAMES, key="cl_model")

linkage, cov_type = "ward", "full"
if chosen_model == "계층적 군집":
    linkage = st.selectbox("연결 방식(linkage)", ["ward", "complete", "average"])
if chosen_model == "가우시안 혼합(GMM)":
    cov_type = st.selectbox("공분산 유형", ["full", "tied", "diag", "spherical"])

labels = fit_labels(chosen_model, X, k_final, linkage, cov_type)
cdata["군집"] = labels

if len(set(labels)) > 1:
    m1, m2, m3 = st.columns(3)
    m1.metric("실루엣", f"{silhouette_score(X, labels):.3f}")
    m2.metric("Davies-Bouldin", f"{davies_bouldin_score(X, labels):.3f}")
    m3.metric("Calinski-Harabasz", f"{calinski_harabasz_score(X, labels):.1f}")

# PCA 2차원 시각화
pca = PCA(n_components=2, random_state=42)
pts = pca.fit_transform(X)
fig2, ax2 = plt.subplots(figsize=(8, 6))
sc = ax2.scatter(pts[:, 0], pts[:, 1], c=labels, cmap="tab10", s=70, edgecolor="white")
if len(cdata) <= 70:
    for (px, py), t in zip(pts, cdata["라벨"]):
        ax2.annotate(t, (px, py), fontsize=7, alpha=0.8)
ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
ax2.set_title(f"{chosen_model} (K={k_final}) 군집 결과 - PCA 2차원")
ax2.grid(alpha=0.3)
ax2.legend(*sc.legend_elements(), title="군집")
fig2.tight_layout()
st.pyplot(fig2)

# 군집별 평균 프로필
st.markdown("**군집별 특성 평균** (군집의 의미를 해석해 보세요)")
profile = cdata.groupby("군집")[features].mean()
profile["데이터 수"] = cdata.groupby("군집").size()
st.dataframe(profile.style.format("{:.3f}", subset=features), use_container_width=True)

# 군집별 구성원
with st.expander("군집별 구성원 보기"):
    for g in sorted(cdata["군집"].unique()):
        members = cdata.loc[cdata["군집"] == g, "라벨"].tolist()
        st.write(f"**군집 {g}** ({len(members)}개): " + ", ".join(members))

# 덴드로그램 (계층적 군집일 때)
if chosen_model == "계층적 군집":
    from scipy.cluster.hierarchy import dendrogram, linkage as sch_linkage
    Z_mat = sch_linkage(X, method=linkage)
    fig3, ax3 = plt.subplots(figsize=(12, 4))
    dendrogram(Z_mat, labels=cdata["라벨"].tolist(), leaf_rotation=90,
               leaf_font_size=7, ax=ax3)
    ax3.set_title("덴드로그램")
    fig3.tight_layout()
    st.pyplot(fig3)

# ---------- 3) 최종 선택 ----------
st.subheader("3️⃣ 최종 모델 확정")
if st.button("✅ 이 모델로 확정하기"):
    st.session_state["final_cluster_model"] = {
        "대상": ptype, "모델": chosen_model, "K": k_final, "특성": features,
    }
if "final_cluster_model" in st.session_state:
    f = st.session_state["final_cluster_model"]
    st.success(f"확정된 모델: [{f['대상']}] {f['모델']} / K={f['K']} / 특성: {', '.join(f['특성'])}")

st.download_button("⬇️ 군집 결과 CSV", cdata.to_csv(index=False).encode("utf-8-sig"),
                   file_name=f"{ptype}_군집결과.csv", mime="text/csv")
