import pandas as pd


TARGET_PRIORITY_ORDER = ["Low", "Medium", "High"]
DEFAULT_SAMPLE_SIZE = 15000
DEFAULT_RANDOM_STATE = 42
MAX_NUMERIC_FEATURES = 6
MAX_CATEGORICAL_FEATURES = 6
MAX_CATEGORY_LEVELS = 10
MISSING_LABEL = "Sin dato"
OTHER_LABEL = "Otras categorías"


def prepare_eda_metadata(df: pd.DataFrame, target_column: str | None) -> dict:
    raw_numeric_columns = df.select_dtypes(include="number").columns.tolist()
    raw_categorical_columns = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_columns = [
        column
        for column in raw_numeric_columns
        if column != target_column and not _looks_like_identifier(df, column)
    ]
    categorical_columns = [
        column
        for column in raw_categorical_columns
        if column != target_column and df[column].nunique(dropna=False) > 1
    ]

    target_profile = prepare_target_profile(df, target_column)

    return {
        "target_profile": target_profile,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "selected_numeric_columns": select_top_numeric_features(
            df,
            numeric_columns,
            target_profile,
            limit=MAX_NUMERIC_FEATURES,
        ),
        "selected_categorical_columns": select_top_categorical_features(
            df,
            categorical_columns,
            target_profile,
            limit=MAX_CATEGORICAL_FEATURES,
        ),
    }


def prepare_target_profile(df: pd.DataFrame, target_column: str | None) -> dict | None:
    if not target_column or target_column not in df.columns:
        return None

    raw_series = df[target_column]
    display_series = raw_series.astype("string").fillna(MISSING_LABEL)

    if pd.api.types.is_numeric_dtype(raw_series):
        analysis_series = pd.to_numeric(raw_series, errors="coerce")
        return {
            "column": target_column,
            "display_series": display_series,
            "analysis_series": analysis_series,
            "is_numeric": True,
            "is_encoded": False,
            "encoding_note": None,
            "tick_values": None,
            "tick_text": None,
        }

    analysis_series, mapping = encode_target_series(raw_series)
    ordered_pairs = sorted(mapping.items(), key=lambda item: item[1])

    return {
        "column": target_column,
        "display_series": display_series,
        "analysis_series": analysis_series,
        "is_numeric": False,
        "is_encoded": True,
        "encoding_note": (
            "Para scatter plots, correlaciones y medias por categoría, la variable objetivo se "
            "codifica como ordinal solo para preservar el orden relativo de las clases."
        ),
        "tick_values": [value for _, value in ordered_pairs],
        "tick_text": [label for label, _ in ordered_pairs],
    }


def build_overview_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dtype_table = (
        pd.DataFrame(
            {
                "columna": df.columns,
                "tipo_dato": df.dtypes.astype(str).values,
                "no_nulos": df.notna().sum().values,
                "valores_unicos": df.nunique(dropna=False).values,
            }
        )
        .sort_values("columna")
        .reset_index(drop=True)
    )

    missing_table = (
        pd.DataFrame(
            {
                "columna": df.columns,
                "nulos": df.isna().sum().values,
            }
        )
        .assign(
            porcentaje_nulos=lambda frame: (frame["nulos"] / len(df) * 100).round(2)
        )
        .sort_values(["porcentaje_nulos", "nulos"], ascending=False)
        .reset_index(drop=True)
    )
    missing_table = missing_table[missing_table["nulos"] > 0]

    return dtype_table, missing_table


def build_target_summary(df: pd.DataFrame, target_profile: dict | None) -> dict:
    if target_profile is None:
        return {
            "distribution": pd.DataFrame(),
            "stats": pd.DataFrame(),
            "interpretation": (
                "La variable objetivo no está disponible en este dataset, así que el análisis "
                "específico del target se omite de forma controlada."
            ),
        }

    target_column = target_profile["column"]
    distribution_source = (
        df[target_column]
        .astype("string")
        .fillna(MISSING_LABEL)
        .value_counts(dropna=False)
        .rename_axis(target_column)
        .reset_index(name="recuento")
    )
    distribution_source["porcentaje"] = (
        distribution_source["recuento"] / distribution_source["recuento"].sum() * 100
    ).round(2)

    dominant_row = distribution_source.iloc[0]
    stats_rows = [
        {"métrica": "Filas", "valor": int(len(df))},
        {"métrica": "Valores distintos del objetivo", "valor": int(distribution_source.shape[0])},
        {"métrica": "Valor más frecuente", "valor": str(dominant_row[target_column])},
        {"métrica": "Peso de la clase mayoritaria (%)", "valor": float(dominant_row["porcentaje"])},
        {"métrica": "Valores nulos en el objetivo", "valor": int(df[target_column].isna().sum())},
    ]

    if target_profile["is_encoded"]:
        encoded_target = target_profile["analysis_series"].dropna()
        stats_rows.extend(
            [
                {"métrica": "Media codificada", "valor": round(float(encoded_target.mean()), 3)},
                {"métrica": "Mediana codificada", "valor": round(float(encoded_target.median()), 3)},
            ]
        )

    stats = pd.DataFrame(stats_rows)
    distribution = distribution_source.rename(
        columns={target_column: "valor_objetivo"}
    )

    interpretation = (
        f"La variable objetivo está liderada por `{dominant_row[target_column]}` con "
        f"{dominant_row['porcentaje']:.2f}% de los registros. Esto conviene destacarlo en entrevista "
        "porque anticipa si hará falta vigilar desbalanceo, métricas por clase o ponderación."
    )

    return {
        "distribution": distribution,
        "stats": stats,
        "interpretation": interpretation,
    }


def sample_for_visuals(
    df: pd.DataFrame,
    max_rows: int = DEFAULT_SAMPLE_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df.copy()
    return df.sample(n=max_rows, random_state=random_state)


def select_top_numeric_features(
    df: pd.DataFrame,
    numeric_columns: list[str],
    target_profile: dict | None,
    limit: int = MAX_NUMERIC_FEATURES,
) -> list[str]:
    if not numeric_columns:
        return []

    feature_scores: list[tuple[tuple[float, float, float], str]] = []
    target_series = None if target_profile is None else target_profile["analysis_series"]

    for column in numeric_columns:
        series = pd.to_numeric(df[column], errors="coerce")
        valid_ratio = float(series.notna().mean())
        variability = float(series.std(skipna=True) or 0.0)
        corr_score = 0.0

        if target_series is not None:
            paired = pd.concat([series, target_series], axis=1).dropna()
            if paired[column].nunique() > 1 and paired.iloc[:, 1].nunique() > 1:
                corr_score = abs(float(paired.corr(numeric_only=True).iloc[0, 1]))

        feature_scores.append(((corr_score, valid_ratio, variability), column))

    ranked_columns = sorted(feature_scores, key=lambda item: item[0], reverse=True)
    return [column for _, column in ranked_columns[:limit]]


def select_top_categorical_features(
    df: pd.DataFrame,
    categorical_columns: list[str],
    target_profile: dict | None,
    limit: int = MAX_CATEGORICAL_FEATURES,
) -> list[str]:
    if not categorical_columns:
        return []

    feature_scores: list[tuple[tuple[float, float, float], str]] = []
    target_series = None if target_profile is None else target_profile["analysis_series"]

    for column in categorical_columns:
        collapsed = collapse_categories(df[column], top_n=MAX_CATEGORY_LEVELS)
        valid_ratio = float(df[column].notna().mean())
        cardinality_penalty = -float(collapsed.nunique(dropna=False))
        separation_score = 0.0

        if target_series is not None:
            grouped = (
                pd.DataFrame({"categoria": collapsed, "objetivo": target_series})
                .dropna(subset=["objetivo"])
                .groupby("categoria", dropna=False)["objetivo"]
                .agg(["mean", "size"])
            )
            if not grouped.empty:
                separation_score = float(grouped["mean"].max() - grouped["mean"].min())

        feature_scores.append(((separation_score, valid_ratio, cardinality_penalty), column))

    ranked_columns = sorted(feature_scores, key=lambda item: item[0], reverse=True)
    return [column for _, column in ranked_columns[:limit]]


def build_categorical_frequency_table(
    df: pd.DataFrame,
    category_column: str,
    top_n: int = MAX_CATEGORY_LEVELS,
) -> pd.DataFrame:
    collapsed = collapse_categories(df[category_column], top_n=top_n)
    counts = collapsed.value_counts(dropna=False)

    frequency_table = counts.rename_axis(category_column).reset_index(name="recuento")
    frequency_table["porcentaje"] = (
        frequency_table["recuento"] / frequency_table["recuento"].sum() * 100
    ).round(2)
    return frequency_table


def build_categorical_target_table(
    df: pd.DataFrame,
    category_column: str,
    target_profile: dict | None,
    top_n: int = MAX_CATEGORY_LEVELS,
) -> pd.DataFrame:
    if target_profile is None:
        return pd.DataFrame()

    collapsed = collapse_categories(df[category_column], top_n=top_n)
    target_table = (
        pd.DataFrame(
            {
                category_column: collapsed,
                "media_objetivo": target_profile["analysis_series"],
            }
        )
        .dropna(subset=["media_objetivo"])
        .groupby(category_column, dropna=False)["media_objetivo"]
        .agg(["mean", "size"])
        .rename(columns={"mean": "media_objetivo", "size": "recuento"})
        .reset_index()
        .sort_values(["media_objetivo", "recuento"], ascending=[False, False])
        .reset_index(drop=True)
    )
    target_table["media_objetivo"] = target_table["media_objetivo"].round(3)
    return target_table


def build_correlation_matrix(df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    usable_columns = [
        column for column in numeric_columns if df[column].nunique(dropna=True) > 1
    ]
    if len(usable_columns) < 2:
        return pd.DataFrame()
    return df[usable_columns].corr(numeric_only=True)


def build_target_correlation_ranking(
    df: pd.DataFrame,
    numeric_columns: list[str],
    target_profile: dict | None,
) -> pd.DataFrame:
    if target_profile is None:
        return pd.DataFrame()

    ranking_rows = []
    target_series = target_profile["analysis_series"]

    for column in numeric_columns:
        series = pd.to_numeric(df[column], errors="coerce")
        paired = pd.concat([series, target_series], axis=1).dropna()
        if paired.empty or paired[column].nunique() <= 1 or paired.iloc[:, 1].nunique() <= 1:
            continue

        corr_value = float(paired.corr(numeric_only=True).iloc[0, 1])
        ranking_rows.append(
            {
                "variable": column,
                "correlacion": round(corr_value, 3),
                "correlacion_absoluta": round(abs(corr_value), 3),
            }
        )

    if not ranking_rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(ranking_rows)
        .sort_values(["correlacion_absoluta", "variable"], ascending=[False, True])
        .reset_index(drop=True)
    )


def generate_business_insights(
    df: pd.DataFrame,
    metadata: dict,
) -> list[str]:
    insights: list[str] = []
    target_profile = metadata["target_profile"]
    numeric_columns = metadata["numeric_columns"]
    categorical_columns = metadata["selected_categorical_columns"]

    if target_profile is not None:
        target_summary = build_target_summary(df, target_profile)
        distribution = target_summary["distribution"]
        if not distribution.empty:
            dominant = distribution.iloc[0]
            minority = distribution.iloc[-1]
            insights.append(
                f"`{dominant['valor_objetivo']}` representa el {dominant['porcentaje']:.2f}% del dataset, "
                f"mientras que `{minority['valor_objetivo']}` concentra el {minority['porcentaje']:.2f}%. "
                "Es una señal importante para justificar métricas por clase y vigilancia del desbalanceo."
            )

    correlation_ranking = build_target_correlation_ranking(df, numeric_columns, target_profile)
    if not correlation_ranking.empty:
        top_feature = correlation_ranking.iloc[0]
        insights.append(
            f"`{top_feature['variable']}` es la variable numérica con mayor asociación lineal absoluta "
            f"con el objetivo codificado (|corr| = {top_feature['correlacion_absoluta']:.3f}). "
            "Es una candidata natural para explicar dónde aparece la señal predictiva principal."
        )

        if len(correlation_ranking) > 1:
            second_feature = correlation_ranking.iloc[1]
            insights.append(
                f"`{second_feature['variable']}` aparece como segunda señal numérica más fuerte "
                f"(|corr| = {second_feature['correlacion_absoluta']:.3f}), lo que sugiere que el problema "
                "no depende de un único driver ambiental."
            )

    strongest_category = find_strongest_categorical_difference(df, categorical_columns, target_profile)
    if strongest_category is not None:
        insights.append(
            f"`{strongest_category['column']}` muestra la mayor diferencia en media del objetivo codificado "
            f"entre `{strongest_category['high_category']}` y `{strongest_category['low_category']}` "
            f"(diferencia = {strongest_category['spread']:.3f}). Es una buena variable para defender segmentación o feature engineering."
        )

    outlier_summary = build_outlier_summary(df, numeric_columns)
    if outlier_summary is not None:
        if outlier_summary["outlier_share"] > 0:
            insights.append(
                f"`{outlier_summary['feature']}` presenta aproximadamente un "
                f"{outlier_summary['outlier_share'] * 100:.2f}% de outliers según la regla del IQR. "
                "No implica causalidad ni error de dato, pero sí conviene considerarlo al comparar modelos robustos."
            )
        else:
            insights.append(
                "Las variables numéricas se ven acotadas bajo una revisión por IQR, así que el tratamiento agresivo de outliers no parece una prioridad inicial."
            )

    return insights[:5]


def find_strongest_categorical_difference(
    df: pd.DataFrame,
    categorical_columns: list[str],
    target_profile: dict | None,
) -> dict | None:
    if target_profile is None or not categorical_columns:
        return None

    best_result = None

    for column in categorical_columns:
        grouped = build_categorical_target_table(df, column, target_profile)
        if grouped.shape[0] < 2:
            continue

        high_row = grouped.iloc[0]
        low_row = grouped.iloc[-1]
        spread = float(high_row["media_objetivo"] - low_row["media_objetivo"])

        if best_result is None or spread > best_result["spread"]:
            best_result = {
                "column": column,
                "high_category": str(high_row[column]),
                "low_category": str(low_row[column]),
                "spread": spread,
            }

    return best_result


def build_outlier_summary(df: pd.DataFrame, numeric_columns: list[str]) -> dict | None:
    best_result = None

    for column in numeric_columns:
        outlier_share = calculate_outlier_share(df[column])
        if best_result is None or outlier_share > best_result["outlier_share"]:
            best_result = {
                "feature": column,
                "outlier_share": outlier_share,
            }

    return best_result


def calculate_outlier_share(series: pd.Series) -> float:
    clean_series = pd.to_numeric(series, errors="coerce").dropna()
    if clean_series.nunique() < 4:
        return 0.0

    q1 = clean_series.quantile(0.25)
    q3 = clean_series.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0.0

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return float(((clean_series < lower_bound) | (clean_series > upper_bound)).mean())


def collapse_categories(series: pd.Series, top_n: int = MAX_CATEGORY_LEVELS) -> pd.Series:
    normalized = series.astype("string").fillna(MISSING_LABEL)
    counts = normalized.value_counts(dropna=False)
    if len(counts) <= top_n:
        return normalized

    top_categories = counts.head(top_n).index
    return normalized.where(normalized.isin(top_categories), OTHER_LABEL)


def encode_target_series(series: pd.Series) -> tuple[pd.Series, dict]:
    cleaned = series.dropna().astype(str)
    unique_values = cleaned.unique().tolist()

    if set(unique_values).issubset(set(TARGET_PRIORITY_ORDER)):
        ordered_values = [value for value in TARGET_PRIORITY_ORDER if value in unique_values]
    else:
        ordered_values = sorted(unique_values)

    mapping = {value: index for index, value in enumerate(ordered_values)}
    encoded_series = series.astype("string").map(mapping).astype("float")
    return encoded_series.rename(f"{series.name}_codificada"), mapping


def format_large_number(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def _looks_like_identifier(df: pd.DataFrame, column: str) -> bool:
    lowered = column.lower()
    if lowered == "id" or lowered.endswith("_id"):
        return True
    return df[column].nunique(dropna=False) >= len(df) * 0.95
