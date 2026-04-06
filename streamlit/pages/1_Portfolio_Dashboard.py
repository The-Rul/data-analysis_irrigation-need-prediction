from pathlib import Path
import sys

import pandas as pd
import streamlit as st

st.set_page_config(layout="wide")


CURRENT_FILE = Path(__file__).resolve()
APP_DIR = CURRENT_FILE.parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from utils.analysis import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_SAMPLE_SIZE,
    build_categorical_frequency_table,
    build_categorical_target_table,
    build_correlation_matrix,
    build_overview_tables,
    build_target_correlation_ranking,
    build_target_summary,
    format_large_number,
    generate_business_insights,
    prepare_eda_metadata,
    sample_for_visuals,
)
from utils.charts import (
    make_categorical_frequency_chart,
    make_categorical_target_mean_chart,
    make_correlation_heatmap,
    make_feature_target_scatter,
    make_numeric_boxplot,
    make_numeric_histogram,
    make_target_boxplot,
    make_target_correlation_chart,
    make_target_histogram,
)
from utils.data_loader import get_data_paths, load_dataset, summarize_dataset


st.title("EDA - Analisis exploratorio")
st.caption("Análisis exploratorio orientado a entrevista técnica y decisiones de modelado.")

train_path, _ = get_data_paths()
df = load_dataset(train_path)
summary = summarize_dataset(df)
metadata = prepare_eda_metadata(df, summary["target_column"])
target_profile = metadata["target_profile"]
sampled_df = sample_for_visuals(
    df,
    max_rows=DEFAULT_SAMPLE_SIZE,
    random_state=DEFAULT_RANDOM_STATE,
)
dtype_table, missing_table = build_overview_tables(df)


with st.container(border=True):
    st.subheader("Vista rápida del dataset")

    resumen_tab, objetivo_tab, numericas_tab, categoricas_tab, correlaciones_tab, hallazgos_tab = st.tabs(
        [
            "Resumen",
            "Variable objetivo",
            "Variables numéricas",
            "Variables categóricas",
            "Correlaciones",
            "Hallazgos de negocio",
        ]
    )

    with resumen_tab:
        st.caption("Aquí se valida tamaño, tipología de columnas y calidad básica del dato antes de hablar de modelos.")

        metric_cols = st.columns(4)
        metric_cols[0].metric("Dimensiones", f"{summary['rows']} x {summary['columns']}")
        metric_cols[1].metric("Columnas", summary["columns"])
        metric_cols[2].metric("Variables numéricas", len(summary["numeric_columns"]))
        metric_cols[3].metric("Variables categóricas", len(summary["categorical_columns"]))


        st.markdown("**Vista previa de los datos**")
        st.dataframe(df.head(10), use_container_width=True, height=360)

        st.markdown("**Tipos de datos**")
        st.dataframe(dtype_table, use_container_width=True, height=360, hide_index=True)

        st.markdown("**Porcentaje de nulos por columna**")
        if missing_table.empty:
            st.success("No se detectaron valores nulos en el dataset de entrenamiento.")
        else:
            st.dataframe(missing_table, use_container_width=True, height=320, hide_index=True)

    with objetivo_tab:
        st.caption("Esta pestaña ayuda a explicar si el objetivo tiene desbalanceo o estructura ordinal relevante para el modelado.")

        if target_profile is None:
            st.info("No se encontró `Irrigation_Need`, así que el análisis específico de la variable objetivo se omite sin romper la app.")
        else:
            target_frame = pd.DataFrame(
                {
                    "objetivo_display": target_profile["display_series"],
                    "objetivo_analisis": target_profile["analysis_series"],
                }
            ).dropna(subset=["objetivo_analisis"])
            target_summary = build_target_summary(df, target_profile)

            chart_col1, chart_col2 = st.columns(2, gap="large")
            with chart_col1:
                st.plotly_chart(
                    make_target_histogram(target_frame, target_profile["column"]),
                    use_container_width=True,
                )
            with chart_col2:
                st.plotly_chart(
                    make_target_boxplot(target_frame, target_profile["column"], target_profile),
                    use_container_width=True,
                )

            stats_col, distribution_col = st.columns([1, 1.3], gap="large")
            with stats_col:
                st.markdown("**Estadísticas descriptivas**")
                st.dataframe(target_summary["stats"], use_container_width=True, hide_index=True)

            with distribution_col:
                st.markdown("**Distribución del objetivo**")
                st.dataframe(target_summary["distribution"], use_container_width=True, hide_index=True)

            st.info(target_summary["interpretation"])
            if target_profile["encoding_note"]:
                st.caption(target_profile["encoding_note"])

    with numericas_tab:
        st.caption("Se priorizan hasta 6 variables numéricas relevantes para explicar dispersión, forma y relación con el objetivo.")

        selected_numeric = metadata["selected_numeric_columns"]
        if not selected_numeric:
            st.info("No se detectaron variables numéricas analizables tras excluir columnas con aspecto de identificador.")
        else:
            st.caption(
                f"Los scatter plots usan una muestra aleatoria reproducible de {format_large_number(len(sampled_df))} filas para mantener la app ágil."
            )

            for index, feature_name in enumerate(selected_numeric):
                with st.expander(feature_name, expanded=index == 0):
                    hist_col, box_col = st.columns(2, gap="large")

                    with hist_col:
                        st.plotly_chart(
                            make_numeric_histogram(sampled_df, feature_name),
                            use_container_width=True,
                        )

                    with box_col:
                        st.plotly_chart(
                            make_numeric_boxplot(sampled_df, feature_name),
                            use_container_width=True,
                        )

                    if target_profile is None:
                        st.info("No se puede mostrar la relación con el objetivo porque `Irrigation_Need` no está disponible.")
                    else:
                        scatter_frame = sampled_df[[feature_name]].copy()
                        scatter_frame["objetivo_analisis"] = target_profile["analysis_series"].reindex(sampled_df.index)
                        scatter_frame["objetivo_display"] = target_profile["display_series"].reindex(sampled_df.index)
                        scatter_frame = scatter_frame.dropna(subset=[feature_name, "objetivo_analisis"])

                        st.plotly_chart(
                            make_feature_target_scatter(
                                scatter_frame,
                                feature_name,
                                target_profile["column"],
                                target_profile,
                            ),
                            use_container_width=True,
                        )

    with categoricas_tab:
        st.caption("Las variables categóricas ayudan a defender segmentaciones y diferencias operativas sin sobreinterpretar causalidad.")

        selected_categorical = metadata["selected_categorical_columns"]
        if not selected_categorical:
            st.info("No se detectaron variables categóricas con suficiente variación para esta vista.")
        else:
            st.caption("Cuando una variable tiene muchas categorías, la vista se limita automáticamente al top 10 para mantener legibilidad.")
            if target_profile and target_profile["encoding_note"]:
                st.caption(target_profile["encoding_note"])

            for index, category_column in enumerate(selected_categorical):
                with st.expander(category_column, expanded=index == 0):
                    frequency_table = build_categorical_frequency_table(df, category_column)
                    chart_col1, chart_col2 = st.columns(2, gap="large")

                    with chart_col1:
                        st.plotly_chart(
                            make_categorical_frequency_chart(frequency_table, category_column),
                            use_container_width=True,
                        )

                    with chart_col2:
                        if target_profile is None:
                            st.info("La media del objetivo por categoría no está disponible porque falta `Irrigation_Need`.")
                        else:
                            target_table = build_categorical_target_table(
                                df,
                                category_column,
                                target_profile,
                            )
                            if target_table.empty:
                                st.info("No fue posible calcular la media del objetivo para esta variable categórica.")
                            else:
                                st.plotly_chart(
                                    make_categorical_target_mean_chart(
                                        target_table,
                                        category_column,
                                        target_profile["column"],
                                        target_profile,
                                    ),
                                    use_container_width=True,
                                )

    with correlaciones_tab:
        st.caption("Esta vista resume posibles redundancias y destaca qué señales numéricas merecen más atención en modelado.")

        correlation_matrix = build_correlation_matrix(df, metadata["numeric_columns"])
        correlation_ranking = build_target_correlation_ranking(
            df,
            metadata["numeric_columns"],
            target_profile,
        )

        if correlation_matrix.empty:
            st.info("Se necesitan al menos dos variables numéricas útiles para construir el mapa de calor.")
        else:
            st.plotly_chart(
                make_correlation_heatmap(correlation_matrix),
                use_container_width=True,
            )

        if target_profile is None:
            st.info("El ranking de correlación con la variable objetivo no está disponible porque falta `Irrigation_Need`.")
        elif correlation_ranking.empty:
            st.info("No se pudieron calcular correlaciones estables entre las variables numéricas y el objetivo.")
        else:
            ranking_col, table_col = st.columns([1.35, 1], gap="large")
            with ranking_col:
                st.plotly_chart(
                    make_target_correlation_chart(correlation_ranking, target_profile["column"]),
                    use_container_width=True,
                )
            with table_col:
                st.markdown("**Ranking de correlaciones**")
                st.dataframe(correlation_ranking, use_container_width=True, hide_index=True)

    with hallazgos_tab:
        st.caption("Estos mensajes convierten el EDA en conclusiones breves, profesionales y defendibles en entrevista.")

        insights = generate_business_insights(df, metadata)
        if not insights:
            st.info("No se detectó suficiente estructura para generar hallazgos automáticos útiles.")
        else:
            insight_cols = st.columns(2, gap="large")
            for position, insight in enumerate(insights, start=1):
                target_col = insight_cols[(position - 1) % 2]
                with target_col:
                    with st.container(border=True):
                        st.markdown(f"**Hallazgo {position}.** {insight}")
