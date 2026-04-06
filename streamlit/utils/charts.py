import pandas as pd
import plotly.express as px


PALETTE = [
    "#4C78A8",  # azul (principal)
    "#F58518",  # naranja
    "#54A24B",  # verde
    "#E45756",  # rojo
    "#72B7B2",  # turquesa
    "#B279A2",  # morado
]
PLOTLY_LAYOUT = {
    "template": "plotly_white",
    "legend_title_text": "",
    "margin": {"l": 20, "r": 20, "t": 60, "b": 20},
    "height": 360,
}


def apply_plotly_layout(fig, **overrides):
    layout_config = {**PLOTLY_LAYOUT, **overrides}
    fig.update_layout(**layout_config)
    return fig


def make_target_histogram(
    target_frame: pd.DataFrame,
    target_column: str,
):
    fig = px.histogram(
        target_frame,
        x="objetivo_display",
        color="objetivo_display",
        color_discrete_sequence=PALETTE,
        title=f"Distribución de {target_column}",
        labels={
            "objetivo_display": target_column,
            "count": "Recuento",
        },
    )
    apply_plotly_layout(fig, showlegend=False)
    fig.update_xaxes(title=target_column)
    fig.update_yaxes(title="Recuento")
    return fig


def make_target_boxplot(
    target_frame: pd.DataFrame,
    target_column: str,
    target_profile: dict,
):
    fig = px.box(
        target_frame,
        y="objetivo_analisis",
        points="outliers",
        color_discrete_sequence=[PALETTE[0]],
        title=f"Boxplot de {target_column}",
        labels={"objetivo_analisis": target_column},
    )
    apply_plotly_layout(fig, showlegend=False)
    fig.update_xaxes(title="")
    fig.update_yaxes(title=target_column)

    if target_profile["tick_values"] and target_profile["tick_text"]:
        fig.update_yaxes(
            tickmode="array",
            tickvals=target_profile["tick_values"],
            ticktext=target_profile["tick_text"],
        )

    return fig


def make_numeric_histogram(df: pd.DataFrame, feature_name: str):
    fig = px.histogram(
        df,
        x=feature_name,
        nbins=40,
        color_discrete_sequence=[PALETTE[1]],
        title=f"Distribución de {feature_name}",
        labels={feature_name: feature_name, "count": "Recuento"},
    )
    apply_plotly_layout(fig, showlegend=False)
    fig.update_xaxes(title=feature_name)
    fig.update_yaxes(title="Recuento")
    return fig


def make_numeric_boxplot(df: pd.DataFrame, feature_name: str):
    fig = px.box(
        df,
        y=feature_name,
        points="outliers",
        color_discrete_sequence=[PALETTE[0]],
        title=f"Boxplot de {feature_name}",
        labels={feature_name: feature_name},
    )
    apply_plotly_layout(fig, showlegend=False)
    fig.update_xaxes(title="")
    fig.update_yaxes(title=feature_name)
    return fig


def make_feature_target_scatter(
    df: pd.DataFrame,
    feature_name: str,
    target_column: str,
    target_profile: dict,
):
    fig = px.scatter(
        df,
        x=feature_name,
        y="objetivo_analisis",
        color="objetivo_display",
        opacity=0.55,
        color_discrete_sequence=PALETTE,
        title=f"{feature_name} frente a {target_column}",
        labels={
            feature_name: feature_name,
            "objetivo_analisis": target_column,
            "objetivo_display": target_column,
        },
    )
    apply_plotly_layout(fig)
    fig.update_xaxes(title=feature_name)
    fig.update_yaxes(title=target_column)

    if target_profile["tick_values"] and target_profile["tick_text"]:
        fig.update_yaxes(
            tickmode="array",
            tickvals=target_profile["tick_values"],
            ticktext=target_profile["tick_text"],
        )

    return fig


def make_categorical_frequency_chart(
    frequency_table: pd.DataFrame,
    category_column: str,
):
    fig = px.bar(
        frequency_table,
        x=category_column,
        y="recuento",
        color="recuento",
        color_continuous_scale="Blues",
        title=f"Frecuencia de {category_column}",
        hover_data=["porcentaje"],
        labels={
            category_column: category_column,
            "recuento": "Recuento",
            "porcentaje": "Porcentaje (%)",
        },
    )
    apply_plotly_layout(fig, coloraxis_showscale=False)
    fig.update_xaxes(title=category_column)
    fig.update_yaxes(title="Recuento")
    return fig


def make_categorical_target_mean_chart(
    target_table: pd.DataFrame,
    category_column: str,
    target_column: str,
    target_profile: dict,
):
    fig = px.bar(
        target_table,
        x=category_column,
        y="media_objetivo",
        color="media_objetivo",
        color_continuous_scale="Tealgrn",
        title=f"Media de {target_column} por {category_column}",
        hover_data=["recuento"],
        labels={
            category_column: category_column,
            "media_objetivo": f"Media de {target_column}",
            "recuento": "Recuento",
        },
    )
    apply_plotly_layout(fig, coloraxis_showscale=False)
    fig.update_xaxes(title=category_column)
    fig.update_yaxes(title=f"Media de {target_column}")

    if target_profile["tick_values"] and target_profile["tick_text"]:
        fig.update_yaxes(
            tickmode="array",
            tickvals=target_profile["tick_values"],
            ticktext=target_profile["tick_text"],
        )

    return fig


def make_correlation_heatmap(corr_df: pd.DataFrame):
    fig = px.imshow(
        corr_df,
        text_auto=".2f",
        color_continuous_scale="Blues",
        aspect="auto",
        title="Mapa de calor de correlaciones numéricas",
    )
    apply_plotly_layout(fig, height=460)
    return fig


def make_target_correlation_chart(correlation_ranking: pd.DataFrame, target_column: str):
    fig = px.bar(
        correlation_ranking.sort_values("correlacion_absoluta"),
        x="correlacion_absoluta",
        y="variable",
        orientation="h",
        color="correlacion",
        color_continuous_scale="RdBu",
        title=f"Correlación absoluta con {target_column}",
        labels={
            "correlacion_absoluta": "Correlación absoluta",
            "correlacion": "Correlación",
            "variable": "Variable",
        },
    )
    apply_plotly_layout(fig, coloraxis_showscale=False, height=420)
    fig.update_xaxes(title="Correlación absoluta")
    fig.update_yaxes(title="Variable")
    return fig
