import pandas as pd
import numpy as np
from pathlib import Path
import re
import io

# ============================================================
# 0) 기본 설정 (1시간 기준)
# ============================================================
SMARTCARE_DIR = Path("/home/hd/Desktop/LOG_SMART202509")
EREPORT_DIR   = Path("/home/hd/Desktop/LOG_EREPORT")

OUT_CSV = "./preprocessed_1h_master_20250914_1017.csv"

# 마스터 타임라인 범위 (예시)
start = pd.Timestamp("2025-09-14 20:00:00")
end   = pd.Timestamp("2025-10-17 19:00:00")
master_index = pd.date_range(start=start, end=end, freq="1H")

# 사용자 정의 휴일(날짜만)
HOLIDAY_DATES = pd.to_datetime([
    "2025-09-14",
    "2025-09-20", "2025-09-21",
    "2025-09-27", "2025-09-28",
    "2025-10-03",
    "2025-10-04", "2025-10-05", "2025-10-06", "2025-10-07", "2025-10-08", "2025-10-09",
    "2025-10-11", "2025-10-12",
]).date

# ------------------------------------------------------------
# Power coverage / Smartcare per-unit coverage
# ------------------------------------------------------------
POWER_COVERAGE_THRESHOLD = 0.85   # 60분 중 최소 51분 이상 관측되어야 "보정/사용"
SMART_PER_UNIT_MIN_RATIO = 0.70   # 1시간(720개) 중 유닛별 최소 70% 이상 있어야 Tod 사용
N_UNITS = 11
SMART_SAMPLE_SEC = 5

# ============================================================
# 1) 파일 이름에서 날짜(YYYYMMDD) 뽑기
# ============================================================
def extract_date_from_filename(filename: str) -> pd.Timestamp:
    m = re.search(r"(\d{8})", filename)
    if not m:
        raise ValueError(f"파일명에서 날짜(YYYYMMDD)를 찾을 수 없습니다: {filename}")
    return pd.to_datetime(m.group(1), format="%Y%m%d")

# ============================================================
# 2) Time 컬럼과 파일명 날짜로 datetime 만들기
# ============================================================
def make_datetime_from_time_and_filename(df: pd.DataFrame, file_path: Path) -> pd.DataFrame:
    if "Time" not in df.columns:
        raise ValueError(f"'Time' 컬럼이 없습니다. 파일: {file_path}")

    file_date = extract_date_from_filename(file_path.name)

    df = df.copy()
    df["datetime"] = pd.to_datetime(
        file_date.strftime("%Y-%m-%d") + " " + df["Time"].astype(str),
        errors="coerce"
    )
    df = df.dropna(subset=["datetime"])
    df = df.sort_values("datetime").set_index("datetime")
    return df

# ============================================================
# 3) NULL(\x00) 제거 후 read_csv
# ============================================================
def read_csv_remove_nulls(path: Path) -> pd.DataFrame:
    with open(path, "rb") as fh:
        raw = fh.read()
    if b"\x00" in raw:
        print(f"[WARN] NULL 바이트 감지 → 제거 후 로드: {path.name}")
        raw = raw.replace(b"\x00", b"")
    return pd.read_csv(io.BytesIO(raw))

# ============================================================
# 4) 폴더 안 CSV를 모두 읽어서 concat
# ============================================================
def load_all_csv_time_from_filename(folder: Path, pattern: str) -> pd.DataFrame | None:
    files = sorted(folder.glob(pattern))
    if not files:
        print(f"[WARN] {folder} 에 {pattern}에 해당하는 파일이 없습니다.")
        return None

    dfs = []
    for f in files:
        try:
            df_raw = read_csv_remove_nulls(f)
            df = make_datetime_from_time_and_filename(df_raw, f)
            df["src_file"] = f.name
            dfs.append(df)
        except Exception as e:
            print(f"[ERROR] {f} 로드 실패: {e}")

    if not dfs:
        return None

    return pd.concat(dfs, axis=0).sort_index()

# ============================================================
# 5) 시간 feature 생성 (1시간 기준)
#    - day_cos/sin: 24-step
#    - week_cos/sin: 168-step
#    - is_holiday: 사용자 입력 날짜
# ============================================================
def create_time_features_1h(index: pd.DatetimeIndex) -> pd.DataFrame:
    df_time = pd.DataFrame(index=index)

    # holiday
    df_time["date"] = df_time.index.date
    df_time["is_holiday"] = df_time["date"].isin(set(HOLIDAY_DATES)).astype(int)

    # 24-step cycle
    hour_of_day = df_time.index.hour.astype(int)  # 0~23
    df_time["day_sin"] = np.sin(2 * np.pi * hour_of_day / 24.0)
    df_time["day_cos"] = np.cos(2 * np.pi * hour_of_day / 24.0)

    # 168-step cycle (start 기준으로 시간 인덱스)
    k = np.arange(len(df_time), dtype=float)
    df_time["week_sin"] = np.sin(2 * np.pi * k / 168.0)
    df_time["week_cos"] = np.cos(2 * np.pi * k / 168.0)

    df_time = df_time.drop(columns=["date"])
    return df_time

# ============================================================
# 6) EREPORT 전처리 (1시간 Power 합산 + "결측 시간대 보정")
#    - slot: ceil("1H") 사용 (창 끝 시각)
#    - coverage < 0.85 이면 NaN
#    - 결측(minute)이 있을 때:
#      · 결측 시각의 "가장 가까운 과거 관측 Power"가 0인지,
#        "가장 가까운 미래 관측 Power"가 0인지로 케이스를 나눔
#      · past!=0 and future!=0: 스케일 보정 (N/n)
#      · 한쪽만 0: 0.5 * avg_obs * missing_count 를 더함
#      · 양쪽 0: 관측합 그대로
# ============================================================
def preprocess_ereport_power_1h(df_ereport: pd.DataFrame,
                               freq: str = "1H",
                               coverage_threshold: float = 0.85) -> pd.DataFrame:
    POWER_COL = "Power"
    if POWER_COL not in df_ereport.columns:
        raise ValueError(f"EREPORT에 '{POWER_COL}' 컬럼이 없습니다. 현재 컬럼: {df_ereport.columns.tolist()}")

    df = df_ereport.copy()

    # slot (창 끝)
    df["slot"] = df.index.ceil(freq)

    # 1) 기대 row 수 추정 (median step 기반)
    diffs = df.index.to_series().diff().dropna().dt.total_seconds()
    if len(diffs) == 0:
        raise RuntimeError("EREPORT 로그에서 시간 간격을 추정할 수 없습니다.")
    median_step = diffs.median()
    if median_step <= 0 or np.isnan(median_step):
        median_step = 60.0  # fallback

    expected_rows = int(round(3600 / median_step))
    expected_rows = max(expected_rows, 1)
    min_required_rows = int(expected_rows * coverage_threshold)

    # 2) slot별 관측합/관측개수
    grp_sum = df.groupby("slot")[POWER_COL].sum()
    grp_n   = df.groupby("slot")[POWER_COL].size()

    out = pd.DataFrame(index=grp_sum.index)
    out["Power_1h_rawsum"] = grp_sum
    out["n_rows"] = grp_n
    out["Power_1h"] = np.nan

    # 3) slot별 결측 보정 (1분 단위로 "빠진 분"이 있다고 가정)
    #    - median_step가 60초면 N=60이 됨
    #    - 실제 데이터가 60초가 아니라면 그에 맞춰 minute grid가 아니라 "step grid"로 적용됨
    #    - 즉, N=expected_rows, 결측 개수 = N - n_rows
    for slot, row in out.iterrows():
        n = int(row["n_rows"])
        if n < min_required_rows:
            continue  # NaN 유지

        P_obs = float(row["Power_1h_rawsum"])
        m = expected_rows - n
        if m <= 0:
            out.at[slot, "Power_1h"] = P_obs
            continue

        # 이 slot에 속한 원본 데이터만 추출
        sub = df[df["slot"] == slot][[POWER_COL]].copy()
        if sub.empty:
            continue

        # "결측 시각" 판정을 위해: 해당 slot의 grid 타임스탬프 구성
        # slot은 창의 끝시각이므로 window는 (slot-1h, slot]
        w_end = pd.Timestamp(slot)
        w_start = w_end - pd.Timedelta(freq)

        # 기대 step 간격 (median_step 초)
        step = pd.Timedelta(seconds=float(median_step))
        grid = pd.date_range(start=w_start + step, end=w_end, freq=step)  # (start, end] 구성에 맞춤

        # 관측된 타임스탬프 집합
        obs_times = sub.index

        # grid 중 관측이 없는 타임스탬프를 "결측 시각"으로 본다
        # (타임스탬프가 완벽히 grid에 딱 안맞으면 이 매칭이 느슨해질 수 있어,
        #  실제로는 resample 기반이 더 안전하지만, 현재는 "코드 철학 유지" 요구에 맞춤)
        obs_set = set(obs_times)
        missing_times = [t for t in grid if t not in obs_set]

        # 혹시 grid 매칭 문제로 missing 개수가 어긋나도, 룰은 "결측 존재 여부"용이므로
        # missing 개수는 N-n으로 사용 (당신 정의)
        m_cnt = m

        # 결측 시각들에 대해 "가장 가까운 과거 관측"과 "가장 가까운 미래 관측"의 Power가 0인지 판단
        # - past/future를 slot 단위로 요약하기 위해, 결측들 중 하나라도 past!=0 / future!=0가 존재하면 ON으로 본다.
        # - 더 엄밀히 하려면 결측마다 판단 후 다수결/최악 기준 등을 쓸 수 있으나,
        #   사용자가 제시한 룰은 slot 레벨 보정이 목적이므로 단순화.
        obs_series = sub[POWER_COL].sort_index()

        past_nonzero = False
        future_nonzero = False

        # 결측이 많아도 계산 부담 줄이기 위해 샘플 일부만 검사
        probe = missing_times
        if len(probe) > 10:
            probe = probe[:: max(1, len(probe)//10)]

        for tmiss in probe:
            # past: tmiss 이전 가장 가까운 관측
            past = obs_series.loc[:tmiss]
            if len(past) > 0:
                if float(past.iloc[-1]) != 0.0:
                    past_nonzero = True

            # future: tmiss 이후 가장 가까운 관측
            fut = obs_series.loc[tmiss:]
            if len(fut) > 0:
                if float(fut.iloc[0]) != 0.0:
                    future_nonzero = True

            if past_nonzero and future_nonzero:
                break

        # 케이스 분기
        if past_nonzero and future_nonzero:
            # 스케일 보정 (관측 평균 유지)
            P_adj = P_obs * (expected_rows / n)
        elif past_nonzero ^ future_nonzero:
            # 한쪽만 0 → 결측 구간 평균을 관측 평균의 1/2로 가정
            avg_obs = P_obs / max(n, 1)
            P_adj = P_obs + 0.5 * avg_obs * m_cnt
        else:
            # 양쪽 0 → 관측합 그대로
            P_adj = P_obs

        out.at[slot, "Power_1h"] = P_adj

    # master_index에 맞춰 정렬
    out = out[["Power_1h"]].reindex(master_index)
    return out

# ============================================================
# 7) SMARTCARE 전처리 (1시간 Tod)
#    - slot: ceil("1H") (창 끝)
#    - 각 AutoId별 row 수가 (기대 720)의 70% 이상이면 그 시간의 Tod 사용
#    - Tod는 "유닛별 평균 → 11대 평균" (유닛 가중 동일)
# ============================================================
def preprocess_smartcare_tod_1h(df_smart: pd.DataFrame,
                               n_units: int = 11,
                               sample_interval_sec: int = 5,
                               freq: str = "1H",
                               per_unit_min_ratio: float = 0.70) -> pd.DataFrame:
    REQUIRED_COLS = ["Auto Id", "Tod"]
    missing = [c for c in REQUIRED_COLS if c not in df_smart.columns]
    if missing:
        raise ValueError(f"SMARTCARE에 필요한 컬럼이 없습니다: {missing}")

    df = df_smart.copy()
    df["slot"] = df.index.ceil(freq)

    expected_per_unit = int(round(3600 / sample_interval_sec))
    min_required = int(np.floor(expected_per_unit * per_unit_min_ratio))

    # slot x unit count
    cnt = df.groupby(["slot", "Auto Id"])["Tod"].size().unstack("Auto Id")

    # 11개 모두 존재 + 각 유닛이 최소 row 충족
    has_all_units = cnt.notna().sum(axis=1) == n_units
    enough_each = (cnt >= min_required).all(axis=1)
    valid_slots = has_all_units & enough_each

    # 유닛별 평균 -> 11대 평균
    unit_mean = df.groupby(["slot", "Auto Id"])["Tod"].mean().unstack("Auto Id")

    # valid slot만 평균 내기
    tod_1h = pd.Series(index=unit_mean.index, dtype=float)
    ok = valid_slots.reindex(unit_mean.index).fillna(False)
    tod_1h.loc[ok] = unit_mean.loc[ok].mean(axis=1)  # 유닛 동일가중 평균
    tod_1h.loc[~ok] = np.nan

    out = pd.DataFrame({"Tod_1h": tod_1h}).reindex(master_index)
    return out

# ============================================================
# 8) 전체 파이프라인
# ============================================================
def main():
    # 1) 시간 feature
    df_time = create_time_features_1h(master_index)

    # --------------------------------------------------------
    # 2) SMARTCARE 로드 & 슬라이싱
    # --------------------------------------------------------
    df_smart_raw = load_all_csv_time_from_filename(SMARTCARE_DIR, "LOG_SMARTCARE_*.csv")
    if df_smart_raw is None:
        raise RuntimeError("SMARTCARE 로그를 읽지 못했습니다. 경로/패턴을 확인하세요.")
    print("[INFO] SMARTCARE raw shape:", df_smart_raw.shape)

    # slot 계산에 (start-1H, end] 구간이 필요
    raw_start = start - pd.Timedelta("1H")
    df_smart_raw = df_smart_raw.loc[(df_smart_raw.index >= raw_start) & (df_smart_raw.index <= end)]
    print("[INFO] SMARTCARE clipped shape:", df_smart_raw.shape)

    smart_1h = preprocess_smartcare_tod_1h(
        df_smart_raw,
        n_units=N_UNITS,
        sample_interval_sec=SMART_SAMPLE_SEC,
        freq="1H",
        per_unit_min_ratio=SMART_PER_UNIT_MIN_RATIO
    )
    print("[INFO] SMARTCARE 1H shape:", smart_1h.shape)

    # --------------------------------------------------------
    # 3) EREPORT 로드 & 슬라이싱
    # --------------------------------------------------------
    df_ereport_raw = load_all_csv_time_from_filename(EREPORT_DIR, "DBG_EREPORT_*.csv")
    if df_ereport_raw is None:
        raise RuntimeError("EREPORT 로그를 읽지 못했습니다. 경로/패턴을 확인하세요.")
    print("[INFO] EREPORT raw shape:", df_ereport_raw.shape)

    raw_start = start - pd.Timedelta("1H")
    df_ereport_raw = df_ereport_raw.loc[(df_ereport_raw.index >= raw_start) & (df_ereport_raw.index <= end)]
    print("[INFO] EREPORT clipped shape:", df_ereport_raw.shape)

    power_1h = preprocess_ereport_power_1h(
        df_ereport_raw,
        freq="1H",
        coverage_threshold=POWER_COVERAGE_THRESHOLD
    )
    print("[INFO] EREPORT 1H shape:", power_1h.shape)

    # --------------------------------------------------------
    # 4) merge
    # --------------------------------------------------------
    df_all = df_time.join(smart_1h, how="left").join(power_1h, how="left")

    print("[INFO] Final shape:", df_all.shape)
    print(df_all.head(3))
    print(df_all.tail(3))

    df_all.to_csv(OUT_CSV, index_label="datetime")
    print(f"[INFO] Saved → {OUT_CSV}")

if __name__ == "__main__":
    main()