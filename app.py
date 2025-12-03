def compute_progressive_tax(gain):
    tax = 0
    if gain <= 0:
        return 0
    brackets = [
        (6000, 0.19),
        (44000, 0.21),
        (150000, 0.23),
        (float("inf"), 0.26),
    ]
    remaining = gain
    limits = [6000, 50000, 200000]
    prev = 0
    for limit, rate in zip(limits, [0.19, 0.21, 0.23]):
        if remaining <= 0:
            break
        taxable = min(remaining, limit - prev)
        tax += taxable * rate
        remaining -= taxable
        prev = limit
    if remaining > 0:
        tax += remaining * 0.26
    return tax

# === Helper: Cálculo neto desde bruto anual en España (aprox) ===
def compute_salary_net(gross_annual: float):
    """
    Cálculo aproximado de sueldo NETO a partir de BRUTO anual en España.

    - Aplica una cotización de Seguridad Social del trabajador ~6.35% sobre el bruto,
      con un tope de base anual aproximado (por encima de esa base la cuota ya no aumenta).
    - Sobre la base después de SS aplica tramos de IRPF aproximados (tipo combinado estatal + autonómico).
    - NO tiene en cuenta mínimos personales/familiares ni deducciones específicas,
      así que es una estimación orientativa, no una simulación fiscal exacta.
    """
    if gross_annual <= 0:
        return 0.0, 0.0, 0.0, 0.0

    # 1) Seguridad Social trabajador (~6.35% del bruto) con tope de base
    ss_rate = 0.0635

    # Aproximación de base máxima anual de cotización:
    # por encima de esta cantidad, no aumentan las cotizaciones del trabajador.
    SS_MAX_BASE_ANUAL = 60000.0  # aprox; valor orientativo

    ss_base = min(gross_annual, SS_MAX_BASE_ANUAL)
    ss_contrib = ss_base * ss_rate

    # Base para IRPF (simplificada: bruto - SS)
    base_irpf = max(0.0, gross_annual - ss_contrib)

    # 2) Tramos IRPF aproximados (ejemplo genérico España, puede variar por CCAA)
    #    0–12.450€: 19%
    #    12.450–20.200€: 24%
    #    20.200–35.200€: 30%
    #    35.200–60.000€: 37%
    #    60.000–300.000€: 45%
    #    >300.000€: 47%
    limits = [12450, 20200, 35200, 60000, 300000]
    rates = [0.19, 0.24, 0.30, 0.37, 0.45]
    remaining = base_irpf
    prev = 0.0
    irpf = 0.0

    for limit, rate in zip(limits, rates):
        if remaining <= 0:
            break
        tramo = min(remaining, limit - prev)
        if tramo > 0:
            irpf += tramo * rate
            remaining -= tramo
            prev = limit

    # Tramo final > 300.000€
    if remaining > 0:
        irpf += remaining * 0.47

    net_annual = gross_annual - ss_contrib - irpf
    if gross_annual > 0:
        effective_total_rate = 1.0 - (net_annual / gross_annual)
    else:
        effective_total_rate = 0.0

    return net_annual, ss_contrib, irpf, effective_total_rate

# === JSON helpers for cartera/planes ===
import os
import json

PORTFOLIO_FILE = "cartera.json"
PLANS_FILE = "planes.json"
PORTFOLIOS_FILE = "carteras.json"
CUSTOM_ASSETS_FILE = "activos_custom.json"

def load_plans():
    if os.path.exists(PLANS_FILE):
        try:
            with open(PLANS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_plans(plans: dict) -> None:
    with open(PLANS_FILE, "w", encoding="utf-8") as f:
        json.dump(plans, f, ensure_ascii=False, indent=2)


# === Helpers para carteras nombradas ===
def load_portfolios():
    """Carga el diccionario de carteras nombradas desde 'carteras.json'."""
    if os.path.exists(PORTFOLIOS_FILE):
        try:
            with open(PORTFOLIOS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            return {}
    return {}


def save_portfolios(portfolios: dict) -> None:
    """Guarda el diccionario de carteras nombradas en 'carteras.json'."""
    with open(PORTFOLIOS_FILE, "w", encoding="utf-8") as f:
        json.dump(portfolios, f, ensure_ascii=False, indent=2)


# === Helpers para activos personalizados del usuario ===
def load_custom_assets():
    """Carga activos personalizados del usuario desde un JSON local.

    El fichero debe llamarse 'activos_custom.json' y contener una lista de objetos
    con, al menos, la clave 'nombre' (y opcionalmente 'tipo', 'ticker', 'isin').
    """
    if os.path.exists(CUSTOM_ASSETS_FILE):
        try:
            with open(CUSTOM_ASSETS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except Exception:
            return []
    return []


def save_custom_assets(custom_assets: list) -> None:
    """Guarda la lista de activos personalizados del usuario en 'activos_custom.json'."""
    with open(CUSTOM_ASSETS_FILE, "w", encoding="utf-8") as f:
        json.dump(custom_assets, f, ensure_ascii=False, indent=2)



import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# --- Loader del universo de activos (CSV grande) ---
@st.cache_data
def load_universe_csv():
    """
    Carga el universo completo de activos desde el CSV generado
    (ej: 'TradeRepublic_Activos_Completo.csv').

    El CSV debe contener al menos:
    ISIN, Name, Type, Region, Country, Country_Code, ETF_Provider,
    ETF_Subtype, Distribution, Currency_Name, Is_ADR, Page, Search_Key
    """
    try:
        df = pd.read_csv("TradeRepublic_Activos_Completo.csv")
        # Normalizamos algunas columnas clave
        for col in ["ISIN", "Name", "Search_Key"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        # Aseguramos columnas esperadas aunque vengan ausentes
        for col in ["Type", "Region", "Country", "ETF_Provider", "ETF_Subtype", "Currency_Name"]:
            if col not in df.columns:
                df[col] = ""
        return df
    except Exception:
        return pd.DataFrame()

from rebalance_marcos import (
    Portfolio,
    compute_contribution_plan,
    required_constant_monthly_for_goal,
    simulate_constant_plan,
    required_growing_monthlies_for_goal,
    simulate_dca_ramp,
)


st.set_page_config(
    page_title="Planificador de cartera - Marcos",
    page_icon="💶",
    layout="wide",
)


st.title("Planificador de cartera - Marcos Ibáñez")

st.markdown(
    """
Esta aplicación te permite gestionar de forma avanzada tu planificación financiera:

- **Rebalanceo mensual**: reparte tu aportación del mes entre activos para mantener tus pesos objetivo, guardando/cargando carteras y visualizando cómo cambian tus porcentajes antes y después de invertir.  
- **Objetivo a largo plazo**: calcula cuánto deberías aportar (de forma constante o creciente) para alcanzar un patrimonio deseado en X años, con opción de contemplar impuestos, evolución anual, sueldo necesario y gráficos.  
- **Plan de vivienda**: planifica la entrada de un piso incluyendo gastos, rentabilidad, aportaciones constantes o crecientes, impuestos, y simulación de hipoteca.

Usa las pestañas de abajo para navegar por cada módulo.
"""
)

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🔁 Rebalanceo mensual",
        "🎯 Objetivo a largo plazo",
        "🏠 Plan de vivienda",
        "📊 Análisis de cartera",
    ]
)


# ============================
# TAB 1: REBALANCEO MENSUAL
# ============================
with tab1:
    st.header("Rebalanceo con nueva aportación mensual")

    st.markdown(
        "1. Rellena la tabla con tus activos, tipo, valor actual y porcentaje objetivo.\n"
        "2. Indica cuánto vas a aportar el próximo mes y el umbral de rebalanceo.\n"
        "3. Pulsa el botón para ver cómo repartir el dinero."
    )

    # Cargamos listado maestro de activos: universo completo + personalizados
    universo_df = load_universe_csv()
    custom_assets = load_custom_assets()

    # Diccionario de metadatos:
    # - Por nombre de activo: {nombre: {"tipo": ..., "isin": ...}}
    # - Por ISIN: {isin: {"nombre": ..., "tipo": ..., "isin": ...}}
    asset_meta_by_name = {}
    asset_meta_by_isin = {}

    def register_asset(name, tipo=None, isin=None):
        name = str(name).strip()
        isin = (str(isin).strip() if isin is not None else "").upper()
        tipo = tipo or ""
        if not name and not isin:
            return

        # Por nombre
        if name:
            if name not in asset_meta_by_name:
                asset_meta_by_name[name] = {
                    "tipo": tipo or "",
                    "isin": isin or "",
                }
            else:
                if not asset_meta_by_name[name].get("tipo") and tipo:
                    asset_meta_by_name[name]["tipo"] = tipo
                if not asset_meta_by_name[name].get("isin") and isin:
                    asset_meta_by_name[name]["isin"] = isin

        # Por ISIN
        if isin:
            if isin not in asset_meta_by_isin:
                asset_meta_by_isin[isin] = {
                    "nombre": name,
                    "tipo": tipo or "",
                    "isin": isin,
                }
            else:
                if not asset_meta_by_isin[isin].get("nombre") and name:
                    asset_meta_by_isin[isin]["nombre"] = name
                if not asset_meta_by_isin[isin].get("tipo") and tipo:
                    asset_meta_by_isin[isin]["tipo"] = tipo

    # --- Normalización de tipos de activo del CSV ---
    def normalize_asset_type(raw_type: str) -> str:
        """Normaliza el tipo de activo del CSV a las categorías usadas en la app."""
        if not raw_type:
            return ""
        s = str(raw_type).strip().lower()
        if any(x in s for x in ["etf", "index fund", "fund", "fonds"]):
            return "ETF"
        if any(x in s for x in ["stock", "share", "equity", "aktion", "acción", "acciones"]):
            return "Acción"
        if any(x in s for x in ["bond", "renta fija", "obligat"]):
            return "Bono"
        if any(x in s for x in ["crypto", "bitcoin", "btc", "eth"]):
            return "Criptomoneda"
        if any(x in s for x in ["derivative", "option", "future", "warrant"]):
            return "Derivado"
        if any(x in s for x in ["fund", "sicav", "fond"]):
            return "Fondo"
        return "Otro"

    # Añadimos activos personalizados
    for a in custom_assets:
        nombre = a.get("nombre")
        tipo = a.get("tipo") or ""
        isin = a.get("isin") or ""
        register_asset(nombre, tipo=tipo, isin=isin)

    # Añadimos el universo completo (CSV)
    if not universo_df.empty:
        for _, row_uni in universo_df.iterrows():
            nombre = row_uni.get("Name", "")
            tipo_raw = row_uni.get("Type", "")
            tipo_norm = normalize_asset_type(tipo_raw)
            isin = row_uni.get("ISIN", "")
            register_asset(nombre, tipo=tipo_norm, isin=isin)

    # UI para crear activos personalizados locales
    with st.expander("➕ Añadir activo personalizado a tu lista"):
        nombre_custom = st.text_input(
            "Nombre del activo personalizado",
            key="nombre_activo_pers",
        )
        tipo_custom = st.selectbox(
            "Tipo del activo personalizado",
            options=["ETF", "Acción", "Bono", "Derivado", "Criptomoneda", "Fondo", "Otro"],
            key="tipo_activo_pers",
        )
        ticker_custom = st.text_input(
            "Ticker (opcional)",
            key="ticker_activo_pers",
        )
        isin_custom = st.text_input(
            "ISIN (opcional)",
            key="isin_activo_pers",
        )

        if st.button("Añadir activo personalizado", key="btn_add_custom"):
            if not nombre_custom.strip():
                st.error("El nombre del activo no puede estar vacío.")
            else:
                existentes = load_custom_assets()
                existentes.append(
                    {
                        "nombre": nombre_custom.strip(),
                        "tipo": tipo_custom,
                        "ticker": ticker_custom.strip(),
                        "isin": isin_custom.strip(),
                    }
                )
                save_custom_assets(existentes)
                st.success(f"Activo personalizado '{nombre_custom}' añadido correctamente.")
                st.rerun()

    # Tabla vacía por defecto: el usuario añadirá activos mediante el editor
    default_data = pd.DataFrame(
        columns=["Activo", "Tipo", "ISIN", "Valor_actual_€", "Peso_objetivo_%"]
    )

    # Inicializar cartera en sesión cargando de fichero si existe
    if "cartera_df" not in st.session_state:
        if os.path.exists(PORTFOLIO_FILE):
            try:
                st.session_state["cartera_df"] = pd.read_json(PORTFOLIO_FILE)
            except Exception:
                st.session_state["cartera_df"] = default_data.copy()
        else:
            st.session_state["cartera_df"] = default_data.copy()

    # Construimos la lista de opciones de activos para el selector de la tabla
    opciones_activos = []
    vistos = set()

    def _add_nombre(n):
        n = str(n).strip()
        if n and n not in vistos:
            vistos.add(n)
            opciones_activos.append(n)

    # 1) Activos personalizados
    for a in custom_assets:
        _add_nombre(a.get("nombre", ""))

    # 2) Universo completo de Trade Republic (columna Name)
    if not universo_df.empty and "Name" in universo_df.columns:
        for n in universo_df["Name"].dropna().unique().tolist():
            _add_nombre(n)

    # 3) Cualquier activo que ya esté en la cartera actual (para que nunca desaparezca del desplegable)
    cartera_df_current = st.session_state["cartera_df"]
    if "Activo" in cartera_df_current.columns:
        for n in cartera_df_current["Activo"].dropna().tolist():
            _add_nombre(n)

    # Lista de ISINs disponibles para el selector (con buscador)
    opciones_isin_set = set(
        isin
        for isin in asset_meta_by_isin.keys()
        if isinstance(isin, str) and isin.strip()
    )
    if "ISIN" in cartera_df_current.columns:
        for i in cartera_df_current["ISIN"].dropna().tolist():
            s = str(i).strip().upper()
            if s:
                opciones_isin_set.add(s)
    opciones_isin = sorted(opciones_isin_set)

    st.subheader("📋 Activos de la cartera")

    # Partimos de la cartera de sesión (sin columna 'Incluir')
    source_df = st.session_state["cartera_df"].copy()

    columnas_orden = ["Activo", "Tipo", "ISIN", "Valor_actual_€", "Peso_objetivo_%"]
    for col in columnas_orden:
        if col not in source_df.columns:
            if col in ["Valor_actual_€", "Peso_objetivo_%"]:
                source_df[col] = 0.0
            else:
                source_df[col] = ""
    source_df = source_df[columnas_orden]

    # Forzamos un índice limpio (1, 2, 3, ...) para evitar índices None/NaN
    source_df = source_df.reset_index(drop=True)
    source_df.index = source_df.index + 1

    # Editor de cartera (los cambios se quedan en el estado interno del widget hasta pulsar el botón de actualizar)
    df_activos = st.data_editor(
        source_df,
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        column_config={
            "Activo": st.column_config.SelectboxColumn(
                "Activo",
                options=opciones_activos,
                help="Selecciona el activo desde la base de datos/universo.",
            ),
            "Tipo": st.column_config.SelectboxColumn(
                "Tipo",
                options=["ETF", "Acción", "Bono", "Derivado", "Criptomoneda", "Fondo", "Otro"],
                help="Se rellena automáticamente al seleccionar el activo si se conoce, pero puedes ajustarlo.",
            ),
            "ISIN": st.column_config.SelectboxColumn(
                "ISIN",
                options=opciones_isin,
                help=(
                    "Selecciona o busca un ISIN. Si el ISIN existe en la base de datos, "
                    "se autocompletará el nombre del activo y su tipo."
                ),
            ),
            "Valor_actual_€": st.column_config.NumberColumn(
                "Valor actual (€)",
                min_value=0.0,
                step=10.0,
                default=0.0,
            ),
            "Peso_objetivo_%": st.column_config.NumberColumn(
                "Peso objetivo (%)",
                min_value=0.0,
                step=1.0,
                default=0.0,
            ),
        },
        key="cartera_editor",
    )

    # Autocompletar Activo, Tipo e ISIN a partir de lo que haya en la fila
    df_autocomplete = df_activos.copy()
    if {"Activo", "ISIN"}.issubset(df_autocomplete.columns):
        for idx, row in df_autocomplete.iterrows():
            nombre = str(row.get("Activo", "")).strip()
            isin = str(row.get("ISIN", "")).strip().upper()

            meta = None

            # Prioridad 1: si hay ISIN en la fila, usamos ese como referencia
            if isin:
                meta = asset_meta_by_isin.get(isin)

            # Prioridad 2: si no se ha encontrado meta por ISIN, probamos por nombre
            if not meta and nombre:
                meta_name = asset_meta_by_name.get(nombre)
                if meta_name:
                    isin_meta = meta_name.get("isin", "")
                    meta = {
                        "nombre": nombre,
                        "tipo": meta_name.get("tipo", ""),
                        "isin": isin_meta,
                    }

            if meta:
                # Aplicamos la meta a la fila: nombre, tipo e ISIN se sincronizan SIEMPRE
                if meta.get("nombre"):
                    df_autocomplete.at[idx, "Activo"] = meta["nombre"]
                if meta.get("tipo"):
                    df_autocomplete.at[idx, "Tipo"] = meta["tipo"]
                if meta.get("isin"):
                    df_autocomplete.at[idx, "ISIN"] = meta["isin"]

        # Asignar 0 por defecto a Valor_actual_€ y Peso_objetivo_% cuando haya activo/ISIN pero no valores
        for idx, row in df_autocomplete.iterrows():
            nombre = str(row.get("Activo", "")).strip()
            isin = str(row.get("ISIN", "")).strip()
            if not nombre and not isin:
                continue  # fila totalmente vacía

            # Valor actual
            val = row.get("Valor_actual_€")
            if val is None or (isinstance(val, str) and not val.strip()):
                df_autocomplete.at[idx, "Valor_actual_€"] = 0.0

            # Peso objetivo
            peso = row.get("Peso_objetivo_%")
            if peso is None or (isinstance(peso, str) and not peso.strip()):
                df_autocomplete.at[idx, "Peso_objetivo_%"] = 0.0

    # Aseguramos que las columnas numéricas sean numéricas y sin NaN
    if "Valor_actual_€" in df_autocomplete.columns:
        df_autocomplete["Valor_actual_€"] = pd.to_numeric(
            df_autocomplete["Valor_actual_€"], errors="coerce"
        ).fillna(0.0)
    if "Peso_objetivo_%" in df_autocomplete.columns:
        df_autocomplete["Peso_objetivo_%"] = pd.to_numeric(
            df_autocomplete["Peso_objetivo_%"], errors="coerce"
        ).fillna(0.0)

    df_activos = df_autocomplete

    # Guardamos inmediatamente la versión autocompletada en sesión,
    # de forma que cualquier cambio en la tabla se refleje en la cartera
    st.session_state["cartera_df"] = df_activos.copy()

    # Mostrar suma de pesos objetivo justo debajo de la tabla (solo filas con Activo no vacío)
    show_normalize_button = False
    try:
        df_live = df_activos.copy()
        df_live = df_live[df_live["Activo"].astype(str).str.strip().ne("")]
        suma_pesos_live = float(df_live["Peso_objetivo_%"].sum())
        st.markdown(
            f"**Suma de pesos objetivo (filas con activo) en tiempo real: {suma_pesos_live:.2f}%**"
        )
        # Solo mostramos el botón si la suma se pasa o se queda corta fuera del rango 98.5–101.5%
        if not (98.5 <= suma_pesos_live <= 101.5):
            show_normalize_button = True
    except Exception:
        show_normalize_button = False

    # Botón para normalizar pesos objetivo a 100% (solo si la suma está fuera del rango)
    if show_normalize_button and st.button("⚖️ Normalizar pesos objetivo al 100%", key="normalizar_pesos"):
        try:
            df_norm = df_activos.copy()
            # Consideramos solo filas con activo no vacío
            mask_valid = df_norm["Activo"].astype(str).str.strip().ne("")
            suma = df_norm.loc[mask_valid, "Peso_objetivo_%"].sum()

            if suma > 0:
                df_norm.loc[mask_valid, "Peso_objetivo_%"] = (
                    df_norm.loc[mask_valid, "Peso_objetivo_%"] / suma * 100.0
                )
                st.session_state["cartera_df"] = df_norm
                st.success("Pesos normalizados correctamente al 100% sobre las filas con activo.")
                st.rerun()
            else:
                st.error("La suma de pesos objetivo de las filas con activo es 0. No se puede normalizar.")
        except Exception as e:
            st.error(f"No se pudo normalizar los pesos: {e}")

    # Filtrar filas vacías (sin activo) para el resto de cálculos y gráficos
    df_activos = df_activos[df_activos["Activo"].astype(str).str.strip().ne("")].copy()

    # Gráfico de tarta con la distribución actual de la cartera (en tiempo real)
    if not df_activos.empty:
        total_valor = float(df_activos["Valor_actual_€"].sum()) if "Valor_actual_€" in df_activos else 0.0

        # Si no hay valor invertido, no intentamos dibujar el pie chart
        if total_valor <= 0:
            st.info(
                "Introduce algún valor actual (> 0 €) en tus activos para poder mostrar el gráfico de distribución."
            )
        else:
            pesos_actuales = df_activos["Valor_actual_€"] / total_valor

            labels = df_activos["Activo"].tolist()
            tipos = df_activos["Tipo"].tolist()

            # Mapa de colores por tipo de activo (para el pie chart)
            type_colors = {
                "ETF": "#1f77b4",
                "Acción": "#ff7f0e",
                "Bono": "#2ca02c",
                "Derivado": "#d62728",
                "Criptomoneda": "#9467bd",
                "Fondo": "#8c564b",
                "Otro": "#7f7f7f",
            }
            colors = [type_colors.get(t, "#7f7f7f") for t in tipos]

            # Ajustar texto al tema actual de Streamlit (oscuro / claro)
            theme_base = st.get_option("theme.base")
            text_color = st.get_option("theme.textColor")

            # Si no hay color definido o estamos en tema oscuro, forzamos blanco para máxima legibilidad.
            if theme_base == "dark" or not text_color:
                text_color = "#FFFFFF"
            else:
                text_color = text_color or "#000000"

            fig, ax = plt.subplots()
            fig.patch.set_facecolor("none")
            ax.set_facecolor("none")

            wedges, texts, autotexts = ax.pie(
                pesos_actuales,
                labels=labels,
                autopct="%1.1f%%",
                startangle=90,
                colors=colors,
            )
            ax.axis("equal")

            for t in texts + autotexts:
                t.set_color(text_color)

            st.markdown("#### Distribución actual de la cartera (por valor de mercado)")
            st.pyplot(fig)

            unique_tipos = []
            unique_colors = []
            for t, c in zip(tipos, colors):
                if t not in unique_tipos:
                    unique_tipos.append(t)
                    unique_colors.append(c)

            if unique_tipos:
                legend_lines = []
                for t, c in zip(unique_tipos, unique_colors):
                    legend_lines.append(
                        f"<span style='font-size:0.85em;'><span style='color:{c}'>■</span> {t}</span>"
                    )
                st.markdown("<br/>".join(legend_lines), unsafe_allow_html=True)
    else:
        st.info("Añade activos a la tabla y asigna un valor actual para ver el gráfico de distribución.")

    col_left, col_right = st.columns(2)

    with col_left:
        monthly_contribution = st.number_input(
            "¿Cuánto dinero quieres aportar el próximo mes? (€)",
            min_value=0,
            step=10,
            value=150,
        )

        umbral_pct = st.number_input(
            "Umbral de rebalanceo (en puntos porcentuales, ej. 2 = 2%)",
            min_value=0.0,
            step=0.5,
            value=2.0,
        )


    if st.button("📊 Calcular plan de aportación para este mes"):
        if df_activos.empty:
            st.error("Añade al menos un activo en la tabla.")
        elif monthly_contribution <= 0:
            st.error("La aportación mensual debe ser mayor que 0.")
        else:
            # Construir diccionarios para Portfolio
            holdings = {}
            targets = {}
            asset_types = {}

            for _, row in df_activos.iterrows():
                nombre = str(row["Activo"]).strip()
                tipo = str(row["Tipo"]).strip()
                valor = float(row["Valor_actual_€"])
                peso_pct = float(row["Peso_objetivo_%"])

                holdings[nombre] = valor
                targets[nombre] = peso_pct / 100.0
                asset_types[nombre] = tipo

            # Normalizar targets si no suman 1
            suma_targets = sum(targets.values())
            if suma_targets == 0:
                st.error("Los pesos objetivo no pueden ser todos cero.")
            else:
                if abs(suma_targets - 1.0) > 0.01:
                    st.info("Normalizando porcentajes objetivo para que sumen 100%.")
                    targets = {k: v / suma_targets for k, v in targets.items()}

                portfolio = Portfolio(
                    holdings=holdings,
                    targets=targets,
                    asset_types=asset_types,
                )

                rebalance_threshold = umbral_pct / 100.0

                plan = compute_contribution_plan(
                    portfolio=portfolio,
                    monthly_contribution=float(monthly_contribution),
                    rebalance_threshold=rebalance_threshold,
                )

                st.subheader("✅ Plan de aportación sugerido")

                df_plan = pd.DataFrame(
                    {
                        "Activo": list(plan.keys()),
                        "Aportación_mes_€": list(plan.values()),
                    }
                )

                st.dataframe(df_plan)
                st.markdown(
                    "Esta tabla indica **cómo repartir la aportación del próximo mes** entre tus activos "
                    "para acercarte a los porcentajes objetivo, **sin vender nada**, solo añadiendo dinero nuevo."
                )

                # Mostrar situación de la cartera antes y después de aplicar la aportación mensual
                st.subheader("⚖️ Situación de la cartera: antes y después de la aportación")

                total_actual = portfolio.total_value()
                pesos_actuales = portfolio.current_weights()

                # Valores y pesos después de aplicar el plan de aportación
                total_despues = total_actual + float(monthly_contribution)
                valores_despues = {
                    a: holdings[a] + float(plan.get(a, 0.0)) for a in holdings.keys()
                }
                if total_despues > 0:
                    pesos_despues = {
                        a: valores_despues[a] / total_despues for a in holdings.keys()
                    }
                else:
                    pesos_despues = {a: 0.0 for a in holdings.keys()}

                df_pesos = pd.DataFrame(
                    {
                        "Activo": list(holdings.keys()),
                        "Valor_antes_€": [holdings[a] for a in holdings.keys()],
                        "Peso_antes_%": [pesos_actuales[a] * 100 for a in holdings.keys()],
                        "Aportación_mes_€": [float(plan.get(a, 0.0)) for a in holdings.keys()],
                        "Valor_despues_€": [valores_despues[a] for a in holdings.keys()],
                        "Peso_despues_%": [pesos_despues[a] * 100 for a in holdings.keys()],
                        "Peso_objetivo_%": [targets[a] * 100 for a in holdings.keys()],
                    }
                )

                st.dataframe(df_pesos)

                st.markdown(
                    "En esta tabla puedes ver, para cada activo: "
                    "**valor y peso ANTES**, la **aportación del mes**, y el **valor y peso DESPUÉS** de aplicar el plan, "
                    "junto con el peso objetivo que quieres mantener.\n\n"
                    "Esto te ayuda a ver si la cartera se acerca a tus porcentajes objetivo usando solo dinero nuevo, "
                    "sin necesidad de vender posiciones."
                )

                # --- Escenario alternativo: incluir ventas si solo con compras no se entra en los porcentajes objetivo ---
                # Comprobamos si, tras aplicar solo la aportación del mes, alguna posición sigue fuera del umbral
                fuera_umbral = []
                for a in holdings.keys():
                    peso_obj_pp = targets[a] * 100.0
                    peso_desp_pp = pesos_despues[a] * 100.0
                    diff_pp = abs(peso_desp_pp - peso_obj_pp)
                    if diff_pp > umbral_pct + 1e-6:
                        fuera_umbral.append(a)

                if fuera_umbral:
                    st.subheader("💸 Escenario con ventas para llegar exactamente a los porcentajes objetivo")
                    st.markdown(
                        "Con solo la aportación de **este mes** no es posible dejar **todas** las posiciones dentro del "
                        "umbral de rebalanceo definido. A continuación se muestra un escenario en el que, además de "
                        "las compras del plan, se realizan **ventas mínimas necesarias** en los activos sobreponderados "
                        "para llegar exactamente a los pesos objetivo."
                    )

                    # Valor total tras aplicar únicamente la aportación (sin ventas)
                    total_despues_solo_compras = total_despues

                    # Holdings ideales si pudiésemos rebalancear completamente con compras + ventas
                    ideal_holdings = {
                        a: targets[a] * total_despues_solo_compras for a in holdings.keys()
                    }

                    ventas = {}
                    for a in holdings.keys():
                        valor_con_aporte = valores_despues[a]
                        delta = ideal_holdings[a] - valor_con_aporte
                        venta = 0.0
                        if delta < 0:
                            # Necesitamos vender para bajar hasta el nivel ideal
                            venta = -delta
                        # Guardamos la venta redondeada a euros sin decimales
                        ventas[a] = int(round(venta))

                    venta_total = sum(ventas.values())

                    if venta_total <= 1e-6:
                        st.info(
                            "En la práctica, las desviaciones son muy pequeñas y no merece la pena plantear ventas adicionales."
                        )
                    else:
                        # Valores finales después de aplicar ventas
                        valores_final = {
                            a: valores_despues[a] - ventas[a] for a in holdings.keys()
                        }
                        total_final = total_despues_solo_compras - venta_total
                        if total_final <= 0:
                            total_final = 1e-9
                        pesos_final = {
                            a: valores_final[a] / total_final for a in holdings.keys()
                        }

                        df_ventas = pd.DataFrame(
                            {
                                "Activo": list(holdings.keys()),
                                "Tipo": [asset_types.get(a, "") for a in holdings.keys()],
                                "Valor_antes_€": [holdings[a] for a in holdings.keys()],
                                "Aportación_mes_€": [float(plan.get(a, 0.0)) for a in holdings.keys()],
                                "Valor_despues_solo_compras_€": [valores_despues[a] for a in holdings.keys()],
                                "Peso_despues_solo_compras_%": [pesos_despues[a] * 100 for a in holdings.keys()],
                                "Peso_objetivo_%": [targets[a] * 100 for a in holdings.keys()],
                                "Venta_necesaria_€": [ventas[a] for a in holdings.keys()],
                                "Valor_final_post_venta_€": [valores_final[a] for a in holdings.keys()],
                                "Peso_final_%": [pesos_final[a] * 100 for a in holdings.keys()],
                            }
                        )

                        st.markdown(
                            f"**Venta total mínima necesaria para alcanzar exactamente los pesos objetivo:** "
                            f"≈ **{venta_total:,.0f} €**, repartida entre los activos sobreponderados."
                        )

                        # Tabla 1: resumen de ventas por activo (más compacta)
                        st.markdown("##### 🧾 Resumen de ventas por activo")
                        df_resumen_ventas = df_ventas[[
                            "Activo",
                            "Venta_necesaria_€",
                            "Peso_despues_solo_compras_%",
                            "Peso_final_%",
                            "Peso_objetivo_%",
                        ]].copy()
                        st.dataframe(df_resumen_ventas)

                        st.caption(
                            "Las cantidades de venta se calculan como la **venta mínima necesaria** para dejar cada activo "
                            "en su peso objetivo, partiendo de la situación tras aplicar solo la aportación del mes."
                        )


    # Gestión de carteras nombradas (guardado/carga en carteras.json)
    st.markdown("---")
    st.markdown("### 💾 Carteras guardadas")

    portfolios = load_portfolios()
    nombres_carteras = sorted(portfolios.keys()) if isinstance(portfolios, dict) else []

    col_cartera_1, col_cartera_2 = st.columns([2, 2])
    with col_cartera_1:
        nombre_cartera_nueva = st.text_input(
            "Nombre para guardar esta cartera (ej. 'Cartera TR largo plazo')",
            value="",
        )
    with col_cartera_2:
        opciones_carteras = ["(ninguna)"] + nombres_carteras
        cartera_seleccionada = st.selectbox(
            "Cargar cartera existente",
            options=opciones_carteras,
        )

    col_cart_save, col_cart_load = st.columns(2)
    with col_cart_save:
        if st.button("💾 Guardar cartera actual"):
            if not nombre_cartera_nueva:
                st.error("Pon un nombre para la cartera antes de guardarla.")
            else:
                if not isinstance(portfolios, dict):
                    portfolios = {}
                # Guardamos la cartera actual como lista de registros (filas)
                portfolios[nombre_cartera_nueva] = st.session_state["cartera_df"].to_dict(orient="records")
                save_portfolios(portfolios)
                st.success(f"Cartera '{nombre_cartera_nueva}' guardada correctamente en '{PORTFOLIOS_FILE}'.")
    with col_cart_load:
        if st.button("📂 Cargar cartera seleccionada"):
            if cartera_seleccionada == "(ninguna)":
                st.warning("Selecciona una cartera para cargar.")
            else:
                datos = portfolios.get(cartera_seleccionada)
                if not datos:
                    st.error("No se ha podido cargar esa cartera.")
                else:
                    try:
                        st.session_state["cartera_df"] = pd.DataFrame(datos)
                        st.success(f"Cartera '{cartera_seleccionada}' cargada. Revisa/edita la tabla; los cambios se aplican automáticamente.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error al reconstruir la cartera desde '{PORTFOLIOS_FILE}': {e}")


    # --- Reset TAB 1 ---
    st.markdown("---")
    if st.button("🔄 Restablecer", key="reset_tab1"):
        for key in ["cartera_df", "cartera_confirmada"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# ============================
# TAB 2: OBJETIVO A LARGO PLAZO
# ============================
with tab2:
    st.header("Calcular aportación mensual para un objetivo futuro")

    # Si hay un plan pendiente de cargar, volcamos sus valores ANTES de instanciar los widgets
    pending_plan_lp = st.session_state.pop("pending_plan_lp", None)
    if pending_plan_lp:
        st.session_state["Valor actual de tu cartera invertida (€)"] = pending_plan_lp["current_total"]
        st.session_state["Ahorros extra iniciales a considerar (cuentas, colchón, etc.) (€)"] = pending_plan_lp["extra_savings"]
        st.session_state["Objetivo de patrimonio futuro que quieres conseguir (€)"] = pending_plan_lp["objetivo_final"]
        st.session_state["Años hasta el objetivo"] = pending_plan_lp["years"]
        st.session_state["Rentabilidad anual estimada (%)"] = pending_plan_lp["annual_return_input"]
        st.session_state["Tener en cuenta impuestos sobre plusvalías al vender todo al final"] = pending_plan_lp["apply_tax"]
        st.session_state["Modo de aportación"] = pending_plan_lp["modo"]
        st.session_state["¿Con cuánto te gustaría empezar aportando cada mes? (€)"] = pending_plan_lp["initial_monthly"]
        st.session_state["¿Qué porcentaje de tu sueldo quieres que represente la aportación mensual? (%) (opcional)"] = pending_plan_lp["salary_pct_input"]

    st.markdown(
        """
Aquí puedes jugar a:
- Elegir un **objetivo de patrimonio** (ej. 50.000 €),
- Decir en cuántos años lo quieres,
- Suponer una rentabilidad anual (ej. 6–8%),
- Y dejar que la app te diga cuánto debes aportar:

- **O bien una cantidad mensual constante**, o  
- **Una aportación que vaya creciendo (linealmente) con los años**.

Además puedes incluir **ahorros extra** que ya tengas fuera de la cartera.
"""
    )

    colA, colB = st.columns(2)

    with colA:
        current_total = st.number_input(
            "Valor actual de tu cartera invertida (€)",
            min_value=0.0,
            step=100.0,
            value=0.0,
            key="Valor actual de tu cartera invertida (€)",
        )
        extra_savings = st.number_input(
            "Ahorros extra iniciales a considerar (cuentas, colchón, etc.) (€)",
            min_value=0.0,
            step=100.0,
            value=0.0,
            key="Ahorros extra iniciales a considerar (cuentas, colchón, etc.) (€)",
        )
        objetivo_final = st.number_input(
            "Objetivo de patrimonio futuro que quieres conseguir (€)",
            min_value=0.0,
            step=1000.0,
            value=50000.0,
            key="Objetivo de patrimonio futuro que quieres conseguir (€)",
        )
        years = st.number_input(
            "Años hasta el objetivo",
            min_value=1,
            max_value=60,
            step=1,
            value=10,
            key="Años hasta el objetivo",
        )

    with colB:
        annual_return_input = st.number_input(
            "Rentabilidad anual estimada (%)",
            min_value=0.0,
            max_value=20.0,
            step=0.5,
            value=7.0,
            key="Rentabilidad anual estimada (%)",
        )
        annual_return = annual_return_input / 100.0

        apply_tax = st.checkbox(
            "Tener en cuenta impuestos sobre plusvalías al vender todo al final",
            value=False,
            help=(
                "Si lo marcas, la cuota mensual se calculará para que el objetivo sea neto, "
                "después de pagar un tipo efectivo sobre las ganancias según tramos progresivos."
            ),
            key="Tener en cuenta impuestos sobre plusvalías al vender todo al final",
        )

        modo = st.radio(
            "Modo de aportación",
            options=["Constante", "Creciente"],
            index=0,
            help="Constante = mismo importe todos los meses. Creciente = empiezas con una cantidad y vas subiendo cada año.",
            key="Modo de aportación",
        )

        initial_monthly = 0
        if modo == "Creciente":
            initial_monthly = st.number_input(
                "¿Con cuánto te gustaría empezar aportando cada mes? (€)",
                min_value=0,
                step=10,
                value=150,
                key="¿Con cuánto te gustaría empezar aportando cada mes? (€)",
            )

        salary_pct_input = st.number_input(
            "¿Qué porcentaje de tu sueldo quieres que represente la aportación mensual? (%) (opcional)",
            min_value=0.0,
            max_value=100.0,
            step=1.0,
            value=0.0,
            help="Por ejemplo, si quieres que la inversión mensual sea el 20% de tu sueldo, pon 20.",
            key="¿Qué porcentaje de tu sueldo quieres que represente la aportación mensual? (%) (opcional)",
        )

    if st.button("🧮 Calcular plan para llegar al objetivo"):
        if objetivo_final <= 0:
            st.error("El objetivo debe ser mayor que 0.")
        else:
            if modo == "Constante":
                months_total = years * 12

                if apply_tax:
                    # Buscamos la aportación mensual para que el objetivo sea NETO (después de impuestos)
                    def net_final_with_monthly(C: float):
                        C_int = int(round(C))
                        if C_int < 0:
                            C_int = 0
                        final_value_sim, _ = simulate_constant_plan(
                            current_total=current_total,
                            monthly_contribution=C_int,
                            years=years,
                            annual_return=annual_return,
                            extra_savings=extra_savings,
                        )
                        principal_total_sim = current_total + extra_savings + C_int * months_total
                        gain_sim = max(0.0, final_value_sim - principal_total_sim)
                        tax_sim = compute_progressive_tax(gain_sim)
                        net_final_sim = final_value_sim - tax_sim
                        return net_final_sim, final_value_sim, gain_sim, tax_sim

                    net0, _, _, _ = net_final_with_monthly(0.0)
                    if net0 >= objetivo_final:
                        mensual_necesaria = 0
                        final_value = current_total + extra_savings
                        gain = 0.0
                        tax = 0.0
                        net_final = final_value
                        series = [final_value] * months_total
                    else:
                        low = 0.0
                        high = max(objetivo_final / max(months_total, 1) * 3, 5000.0)
                        final_value = 0.0
                        gain = 0.0
                        tax = 0.0
                        net_final = 0.0
                        for _ in range(40):
                            mid = (low + high) / 2
                            net_mid, final_mid, gain_mid, tax_mid = net_final_with_monthly(mid)
                            if net_mid < objetivo_final:
                                low = mid
                            else:
                                high = mid
                                final_value = final_mid
                                gain = gain_mid
                                tax = tax_mid
                                net_final = net_mid
                        mensual_necesaria = int(round(high))

                    # Vuelve a simular para obtener la serie (bruta)
                    final_value, series = simulate_constant_plan(
                        current_total=current_total,
                        monthly_contribution=mensual_necesaria,
                        years=years,
                        annual_return=annual_return,
                        extra_savings=extra_savings,
                    )
                    principal_total = current_total + extra_savings + mensual_necesaria * months_total
                    gain = max(0.0, final_value - principal_total)
                    tax = compute_progressive_tax(gain)
                    net_final = final_value - tax

                else:
                    # Sin impuestos: usamos la función auxiliar original
                    mensual_necesaria = required_constant_monthly_for_goal(
                        current_total=current_total,
                        objetivo_final=objetivo_final,
                        years=years,
                        annual_return=annual_return,
                        extra_savings=extra_savings,
                        tax_rate=0.0,
                    )
                    final_value, series = simulate_constant_plan(
                        current_total=current_total,
                        monthly_contribution=mensual_necesaria,
                        years=years,
                        annual_return=annual_return,
                        extra_savings=extra_savings,
                    )
                    months_total = years * 12
                    principal_total = current_total + extra_savings + mensual_necesaria * months_total
                    gain = max(0.0, final_value - principal_total)
                    tax = 0.0
                    net_final = final_value

                if mensual_necesaria == 0:
                    st.success(
                        "Con lo que ya tienes y la rentabilidad asumida, "
                        "en teoría llegarías al objetivo sin necesidad de aportar más (o con 0 €/mes)."
                    )
                else:
                    st.subheader("📌 Resultado (aportación constante)")
                    st.write(
                        f"Para alcanzar **{objetivo_final:,.0f} € NETOS** en **{years} años** "
                        f"con una rentabilidad anual del **{annual_return_input:.1f}%**, "
                        f"deberías aportar aproximadamente **{mensual_necesaria} € al mes**, "
                        "de forma constante."
                    )

                st.write(
                    f"Patrimonio bruto estimado al final: **{final_value:,.0f} €**"
                )
                st.write(
                    f"Plusvalía (beneficio antes de impuestos): **{gain:,.0f} €**"
                )
                if apply_tax:
                    st.write(
                        f"Impuestos estimados sobre plusvalías (según tramos progresivos): "
                        f"**{tax:,.0f} €**"
                    )
                    st.write(
                        f"Patrimonio neto estimado tras impuestos: **{net_final:,.0f} €**"
                    )

                if salary_pct_input > 0 and mensual_necesaria > 0:
                    pct = salary_pct_input / 100.0
                    sueldo_bruto_anual = mensual_necesaria * 12 / pct
                    sueldo_neto_anual, ss_contrib, irpf, eff_rate = compute_salary_net(sueldo_bruto_anual)

                    st.markdown("#### 💼 Sueldo de referencia para esa aportación")
                    st.write(
                        f"Para que **{mensual_necesaria} € al mes** supongan aproximadamente el **{salary_pct_input:.0f}%** de tu sueldo NETO, "
                        f"necesitarías un sueldo bruto de referencia de unos **{sueldo_bruto_anual:,.0f} € al año**, "
                        f"que se traducirían en ~**{sueldo_neto_anual:,.0f} € netos al año** "
                        f"después de una retención total aproximada del **{eff_rate*100:.1f}%** "
                        f"(Seguridad Social + IRPF por tramos)."
                    )

                    st.caption(
                        "El cálculo de neto es una aproximación: usa tramos genéricos de IRPF y una cotización de "
                        "Seguridad Social del ~6.35%, sin tener en cuenta mínimos personales ni deducciones específicas."
                    )

                st.markdown("#### Evolución estimada del patrimonio (antes de impuestos)")
                df_evol = pd.DataFrame(
                    {
                        "Año": [m / 12 for m in range(1, len(series) + 1)],
                        "Patrimonio_estimado_€": series,
                    }
                )
                st.line_chart(df_evol, x="Año", y="Patrimonio_estimado_€")

                st.caption(
                    "Es una simulación sencilla del **valor bruto de la cartera mes a mes**. "
                    "No tiene en cuenta cambios de fiscalidad en el tiempo, tipos variables, ni "
                    "la volatilidad real del mercado."
                )

            else:  # Creciente
                if initial_monthly <= 0:
                    st.error("La aportación inicial debe ser mayor que 0.")
                else:
                    months_total = years * 12

                    if apply_tax:
                        # Buscamos la aportación mensual final para que el objetivo sea NETO
                        def net_final_with_final_monthly(F: float):
                            F_float = float(F)
                            final_val, _ = simulate_dca_ramp(
                                initial_monthly=initial_monthly,
                                final_monthly=F_float,
                                years=years,
                                annual_return=annual_return,
                                initial_value=current_total + extra_savings,
                            )
                            contrib_total = months_total * (initial_monthly + F_float) / 2.0
                            principal_total_sim = current_total + extra_savings + contrib_total
                            gain_sim = max(0.0, final_val - principal_total_sim)
                            tax_sim = compute_progressive_tax(gain_sim)
                            net_final_sim = final_val - tax_sim
                            return net_final_sim, final_val, gain_sim, tax_sim

                        net0, _, _, _ = net_final_with_final_monthly(initial_monthly)
                        if net0 >= objetivo_final:
                            final_monthly_aprox = initial_monthly
                            final_value_grow, series_grow = simulate_dca_ramp(
                                initial_monthly=initial_monthly,
                                final_monthly=final_monthly_aprox,
                                years=years,
                                annual_return=annual_return,
                                initial_value=current_total + extra_savings,
                            )
                            contrib_total = months_total * (initial_monthly + final_monthly_aprox) / 2.0
                            principal_total = current_total + extra_savings + contrib_total
                            gain = max(0.0, final_value_grow - principal_total)
                            tax = compute_progressive_tax(gain)
                            net_final = final_value_grow - tax
                        else:
                            low = initial_monthly
                            high = max(initial_monthly * 3, 5000.0)
                            final_value_grow = 0.0
                            gain = 0.0
                            tax = 0.0
                            net_final = 0.0
                            series_grow = []
                            for _ in range(40):
                                mid = (low + high) / 2
                                net_mid, final_mid, gain_mid, tax_mid = net_final_with_final_monthly(mid)
                                if net_mid < objetivo_final:
                                    low = mid
                                else:
                                    high = mid
                                    final_value_grow = final_mid
                                    gain = gain_mid
                                    tax = tax_mid
                                    net_final = net_mid
                            final_monthly_aprox = int(round(high))
                            final_value_grow, series_grow = simulate_dca_ramp(
                                initial_monthly=initial_monthly,
                                final_monthly=final_monthly_aprox,
                                years=years,
                                annual_return=annual_return,
                                initial_value=current_total + extra_savings,
                            )
                            contrib_total = months_total * (initial_monthly + final_monthly_aprox) / 2.0
                            principal_total = current_total + extra_savings + contrib_total
                            gain = max(0.0, final_value_grow - principal_total)
                            tax = compute_progressive_tax(gain)
                            net_final = final_value_grow - tax
                    else:
                        final_monthly_aprox, resumen_anual = required_growing_monthlies_for_goal(
                            current_total=current_total,
                            objetivo_final=objetivo_final,
                            years=years,
                            annual_return=annual_return,
                            initial_monthly=initial_monthly,
                            extra_savings=extra_savings,
                            tax_rate=0.0,
                        )
                        final_value_grow, series_grow = simulate_dca_ramp(
                            initial_monthly=initial_monthly,
                            final_monthly=final_monthly_aprox,
                            years=years,
                            annual_return=annual_return,
                            initial_value=current_total + extra_savings,
                        )
                        contrib_total = months_total * (initial_monthly + final_monthly_aprox) / 2.0
                        principal_total = current_total + extra_savings + contrib_total
                        gain = max(0.0, final_value_grow - principal_total)
                        tax = 0.0
                        net_final = final_value_grow

                    # Construimos resumen anual
                    resumen_anual = []
                    for año in range(1, years + 1):
                        start_idx = (año - 1) * 12
                        end_idx = año * 12 - 1
                        if months_total > 1:
                            start_month = int(
                                round(
                                    initial_monthly
                                    + (final_monthly_aprox - initial_monthly) * (start_idx / (months_total - 1))
                                )
                            )
                            end_month = int(
                                round(
                                    initial_monthly
                                    + (final_monthly_aprox - initial_monthly) * (end_idx / (months_total - 1))
                                )
                            )
                        else:
                            start_month = final_monthly_aprox
                            end_month = final_monthly_aprox
                        avg_month = int(round((start_month + end_month) / 2))
                        resumen_anual.append(
                            {
                                "year": año,
                                "start": start_month,
                                "end": end_month,
                                "avg": avg_month,
                            }
                        )

                    st.subheader("📌 Resultado (aportación creciente)")
                    st.write(
                        f"Para alcanzar aproximadamente **{objetivo_final:,.0f} € NETOS** en **{years} años** "
                        f"con una rentabilidad anual del **{annual_return_input:.1f}%** y aportaciones crecientes, "
                        f"deberías empezar aportando **{initial_monthly} € al mes** y terminar aportando "
                        f"aproximadamente **{final_monthly_aprox} € al mes**."
                    )

                    df_resumen = pd.DataFrame(resumen_anual)
                    df_resumen = df_resumen.rename(
                        columns={
                            "year": "Año",
                            "start": "Inicio_€/mes",
                            "end": "Fin_€/mes",
                            "avg": "Media_€/mes",
                        }
                    )

                    # Si el usuario ha indicado un porcentaje de sueldo, añadimos columnas de sueldo BRUTO y NETO necesarios
                    if salary_pct_input > 0:
                        pct = salary_pct_input / 100.0
                        sueldos_brutos = []
                        sueldos_netos = []
                        retenciones_totales = []
                        for _, fila in df_resumen.iterrows():
                            media_mes = fila["Media_€/mes"]
                            if media_mes <= 0:
                                sueldo_bruto_anual = 0.0
                                sueldo_neto_anual = 0.0
                                ret_total_pct = 0.0
                            else:
                                sueldo_bruto_anual = media_mes * 12 / pct
                                sueldo_neto_anual, ss_contrib, irpf, eff_rate = compute_salary_net(sueldo_bruto_anual)
                                ret_total_pct = eff_rate * 100.0
                            sueldos_brutos.append(round(sueldo_bruto_anual))
                            sueldos_netos.append(round(sueldo_neto_anual))
                            retenciones_totales.append(round(ret_total_pct, 1))

                        df_resumen["Sueldo_bruto_necesario_€/año"] = sueldos_brutos
                        df_resumen["Sueldo_neto_estimado_€/año"] = sueldos_netos
                        df_resumen["Retención_total_aprox_%"] = retenciones_totales

                    st.markdown("#### Aportaciones aproximadas por año")
                    st.dataframe(df_resumen)

                    st.markdown(
                        "Cada fila representa un año del plan: \n"
                        "- **Inicio_€/mes**: cuánto aportarías al comienzo de ese año.\n"
                        "- **Fin_€/mes**: cuánto aportarías al final de ese año.\n"
                        "- **Media_€/mes**: aportación mensual media aproximada durante ese año.\n"
                        "- **Sueldo_bruto_necesario_€/año** (si has indicado un % de sueldo): sueldo aproximado para que esa media mensual represente ese porcentaje."
                    )

                    st.write(
                        f"Patrimonio bruto estimado al final: **{final_value_grow:,.0f} €**"
                    )
                    st.write(
                        f"Plusvalía (beneficio antes de impuestos): **{gain:,.0f} €**"
                    )
                    if apply_tax:
                        st.write(
                            f"Impuestos estimados sobre plusvalías (según tramos progresivos): "
                            f"**{tax:,.0f} €**"
                        )
                        st.write(
                            f"Patrimonio neto estimado tras impuestos: **{net_final:,.0f} €**"
                        )

                    st.markdown("#### Evolución estimada del patrimonio (antes de impuestos)")
                    df_evol_grow = pd.DataFrame(
                        {
                            "Año": [m / 12 for m in range(1, len(series_grow) + 1)],
                            "Patrimonio_estimado_€": series_grow,
                        }
                    )
                    st.line_chart(df_evol_grow, x="Año", y="Patrimonio_estimado_€")

                    st.caption(
                        "Es una simulación sencilla del **valor bruto de la cartera mes a mes** con aportaciones crecientes. "
                        "No tiene en cuenta cambios de fiscalidad en el tiempo, tipos variables, ni "
                        "la volatilidad real del mercado."
                    )

                    st.caption(
                        "El crecimiento es lineal entre la aportación inicial y la final. "
                        "No tiene en cuenta escalones salariales reales ni cambios de sueldo, "
                        "pero sirve como referencia para visualizar la tendencia."
                    )

    # Gestión de presets / planes para objetivo a largo plazo
    st.markdown("---")
    st.markdown("### 💾 Planes guardados (largo plazo)")

    plans = load_plans()
    planes_lp = plans.get("largo_plazo", {})

    col_plan_lp_1, col_plan_lp_2 = st.columns([2, 2])
    with col_plan_lp_1:
        nombre_plan_lp = st.text_input(
            "Nombre para guardar este plan (ej. 'Plan Indexa/ETF largo plazo')",
            value="",
        )
    with col_plan_lp_2:
        opciones_planes_lp = ["(ninguno)"] + sorted(planes_lp.keys()) if isinstance(planes_lp, dict) else ["(ninguno)"]
        plan_lp_seleccionado = st.selectbox(
            "Cargar plan existente",
            options=opciones_planes_lp,
        )

    col_plan_lp_save, col_plan_lp_load = st.columns(2)
    with col_plan_lp_save:
        if st.button("💾 Guardar plan de largo plazo"):
            if not nombre_plan_lp:
                st.error("Pon un nombre para el plan antes de guardarlo.")
            else:
                if not isinstance(plans.get("largo_plazo"), dict):
                    plans["largo_plazo"] = {}
                plans["largo_plazo"][nombre_plan_lp] = {
                    "current_total": current_total,
                    "extra_savings": extra_savings,
                    "objetivo_final": objetivo_final,
                    "years": int(years),
                    "annual_return_input": annual_return_input,
                    "apply_tax": apply_tax,
                    "modo": modo,
                    "initial_monthly": initial_monthly,
                    "salary_pct_input": salary_pct_input,
                }
                save_plans(plans)
                st.success(f"Plan '{nombre_plan_lp}' guardado correctamente.")
    with col_plan_lp_load:
        if st.button("📂 Cargar plan de largo plazo"):
            if plan_lp_seleccionado == "(ninguno)":
                st.warning("Selecciona un plan para cargar.")
            else:
                plan = planes_lp.get(plan_lp_seleccionado)
                if not plan:
                    st.error("No se ha podido cargar ese plan.")
                else:
                    # Guardamos el plan como "pendiente" y recargamos; en el siguiente run se aplicará antes de los widgets
                    st.session_state["pending_plan_lp"] = plan
                    st.rerun()

    # --- Reset TAB 2 ---
    st.markdown("---")
    if st.button("🔄 Restablecer", key="reset_tab2"):
        keys_lp = [
            "Valor actual de tu cartera invertida (€)",
            "Ahorros extra iniciales a considerar (cuentas, colchón, etc.) (€)",
            "Objetivo de patrimonio futuro que quieres conseguir (€)",
            "Años hasta el objetivo",
            "Rentabilidad anual estimada (%)",
            "Tener en cuenta impuestos sobre plusvalías al vender todo al final",
            "Modo de aportación",
            "¿Con cuánto te gustaría empezar aportando cada mes? (€)",
            "¿Qué porcentaje de tu sueldo quieres que represente la aportación mensual? (%) (opcional)",
        ]
        for key in keys_lp:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# ============================
# TAB 3: PLAN DE VIVIENDA
# ============================
with tab3:
    st.header("Plan de vivienda (ahorro para la entrada)")

    # Si hay un plan de vivienda pendiente de cargar, volcamos sus valores ANTES de instanciar los widgets
    pending_plan_viv = st.session_state.pop("pending_plan_viv", None)
    if pending_plan_viv:
        st.session_state["Precio estimado de la vivienda (€)"] = pending_plan_viv["house_price"]
        st.session_state["% de entrada que exige el banco (%)"] = pending_plan_viv["entrada_pct"]
        st.session_state["Años hasta la compra"] = pending_plan_viv["years_house"]
        st.session_state["Ahorro ya destinado a la entrada (€)"] = pending_plan_viv["ahorro_actual_entrada"]
        st.session_state["Rentabilidad anual estimada del ahorro para la entrada (%)"] = pending_plan_viv["anual_return_house_input"]

    st.markdown(
        """
Este modo está pensado para planificar **la entrada de una vivienda**.

1. Indicas el precio objetivo de la vivienda y el % de entrada (por ejemplo, 20%).  
2. Indicas los **gastos asociados** (ITP, notaría, gestoría, etc.) como porcentaje sobre el precio.  
3. Dices cuántos años faltan hasta la compra y cuánto tienes ya ahorrado para la entrada.  
4. Asumes una rentabilidad anual para el dinero que destines a este objetivo.  
5. Eliges **aportación constante** o **aportación creciente**.  
6. Opcionalmente, puedes activar el cálculo de **impuestos sobre plusvalías** al vender la hucha al final.  
7. Además, puedes simular una **hipoteca** (tipo y plazo) para ver la cuota aproximada.

La app te indica cuánto ahorrar, cuánto aportar al mes, en cuánto tiempo y te da una recomendación de cartera para este objetivo.
"""
    )

    col1, col2 = st.columns(2)

    with col1:
        house_price = st.number_input(
            "Precio estimado de la vivienda (€)",
            min_value=0.0,
            step=5000.0,
            value=200000.0,
            key="Precio estimado de la vivienda (€)",
        )
        entrada_pct = st.number_input(
            "% de entrada que exige el banco (%)",
            min_value=0.0,
            max_value=60.0,
            step=1.0,
            value=20.0,
            key="% de entrada que exige el banco (%)",
        )
        gastos_pct = st.number_input(
            "Gastos asociados (ITP, notaría, gestoría, etc.) sobre el precio (%)",
            min_value=0.0,
            max_value=20.0,
            step=0.5,
            value=10.0,
            key="Gastos asociados vivienda (%)",
        )
        years_house = st.number_input(
            "Años hasta la compra",
            min_value=1,
            max_value=40,
            step=1,
            value=7,
            key="Años hasta la compra",
        )

    with col2:
        ahorro_actual_entrada = st.number_input(
            "Ahorro ya destinado a la entrada (€)",
            min_value=0.0,
            step=1000.0,
            value=0.0,
            key="Ahorro ya destinado a la entrada (€)",
        )
        anual_return_house_input = st.number_input(
            "Rentabilidad anual estimada del ahorro para la entrada (%)",
            min_value=0.0,
            max_value=15.0,
            step=0.5,
            value=4.0,
            key="Rentabilidad anual estimada del ahorro para la entrada (%)",
        )
        annual_return_house = anual_return_house_input / 100.0

        apply_tax_house = st.checkbox(
            "Tener en cuenta impuestos sobre plusvalías al vender la hucha al final",
            value=False,
            help=(
                "Si lo marcas, la cuota mensual se calculará para que el efectivo objetivo sea NETO, "
                "después de pagar impuestos sobre las plusvalías con tramos progresivos."
            ),
            key="Tener en cuenta impuestos vivienda",
        )

        tipo_hipoteca_input = st.number_input(
            "Tipo de interés anual aproximado de la hipoteca (%)",
            min_value=0.0,
            max_value=10.0,
            step=0.1,
            value=3.0,
            key="Tipo interés hipoteca",
        )
        plazo_hipoteca_years = st.number_input(
            "Plazo de la hipoteca (años)",
            min_value=1,
            max_value=40,
            step=1,
            value=30,
            key="Plazo hipoteca años",
        )

    # --- Elegir modo de aportación (constante/creciente) para la entrada
    modo_house = st.radio(
        "Modo de aportación para la entrada",
        options=["Constante", "Creciente"],
        index=0,
        help="Constante = mismo importe todos los meses. Creciente = empiezas con una cantidad y vas subiendo cada año.",
        key="Modo de aportación vivienda",
    )

    initial_monthly_house = 0
    if modo_house == "Creciente":
        initial_monthly_house = st.number_input(
            "¿Con cuánto te gustaría empezar aportando cada mes para la entrada? (€)",
            min_value=0,
            step=50,
            value=300,
            key="Aportación inicial vivienda",
        )

    entrada_objetivo = house_price * entrada_pct / 100.0
    gastos_totales = house_price * gastos_pct / 100.0
    objetivo_total_efectivo = entrada_objetivo + gastos_totales
    restante_necesario = max(0.0, objetivo_total_efectivo - ahorro_actual_entrada)

    st.markdown(
        f"Entrada objetivo: **{entrada_objetivo:,.0f} €** (~{entrada_pct:.0f}% de {house_price:,.0f} €)."
    )
    st.markdown(
        f"Gastos asociados estimados: **{gastos_totales:,.0f} €** (~{gastos_pct:.1f}% sobre el precio)."
    )
    st.markdown(
        f"Total de efectivo objetivo (entrada + gastos): **{objetivo_total_efectivo:,.0f} €**."
    )
    st.markdown(
        f"De ese total, te faltan por ahorrar aproximadamente **{restante_necesario:,.0f} €**."
    )

    # Simulación rápida de hipoteca
    hipoteca_principal = max(0.0, house_price - entrada_objetivo)
    if hipoteca_principal > 0 and tipo_hipoteca_input >= 0 and plazo_hipoteca_years > 0:
        r_mensual = (tipo_hipoteca_input / 100.0) / 12.0
        n_meses_hipoteca = int(plazo_hipoteca_years * 12)
        if r_mensual > 0:
            cuota_mensual_hipoteca = hipoteca_principal * r_mensual * (1 + r_mensual) ** n_meses_hipoteca / (
                (1 + r_mensual) ** n_meses_hipoteca - 1
            )
        else:
            cuota_mensual_hipoteca = hipoteca_principal / n_meses_hipoteca
        st.markdown(
            f"💳 Hipoteca simulada: principal aproximado **{hipoteca_principal:,.0f} €**, "
            f"cuota mensual estimada **{cuota_mensual_hipoteca:,.0f} €** a {plazo_hipoteca_years:.0f} años "
            f"con un tipo del {tipo_hipoteca_input:.1f}%."
        )

    if st.button("🏠 Calcular plan de ahorro para la entrada"):
        if house_price <= 0 or entrada_pct <= 0:
            st.error("Introduce un precio de vivienda y un porcentaje de entrada mayores que 0.")
        elif years_house <= 0:
            st.error("Los años hasta la compra deben ser mayores que 0.")
        elif restante_necesario <= 0:
            st.success(
                "Con lo que ya tienes ahorrado para la entrada, en principio llegarías al objetivo sin necesidad de aportar más."
            )
        elif modo_house == "Creciente" and initial_monthly_house <= 0:
            st.error("La aportación inicial para la entrada debe ser mayor que 0 si eliges modo creciente.")
        else:
            months_house = int(years_house) * 12

            if modo_house == "Constante":
                # === MODO CONSTANTE ===
                if apply_tax_house:
                    # Buscamos la aportación mensual para que el objetivo total (entrada + gastos) sea NETO tras impuestos
                    def net_final_house_with_monthly(C: float):
                        C_int = int(round(C))
                        if C_int < 0:
                            C_int = 0
                        final_val_sim, _ = simulate_constant_plan(
                            current_total=ahorro_actual_entrada,
                            monthly_contribution=C_int,
                            years=int(years_house),
                            annual_return=annual_return_house,
                            extra_savings=0.0,
                        )
                        principal_total_sim = ahorro_actual_entrada + C_int * months_house
                        gain_sim = max(0.0, final_val_sim - principal_total_sim)
                        tax_sim = compute_progressive_tax(gain_sim)
                        net_final_sim = final_val_sim - tax_sim
                        return net_final_sim, final_val_sim, gain_sim, tax_sim

                    net0, _, _, _ = net_final_house_with_monthly(0.0)
                    if net0 >= objetivo_total_efectivo:
                        mensual_entrada = 0
                        final_entrada = ahorro_actual_entrada
                        principal_total_entrada = ahorro_actual_entrada
                        gain_entrada = 0.0
                        tax_entrada = 0.0
                        net_final_entrada = final_entrada
                        series_entrada = [final_entrada] * months_house
                    else:
                        low = 0.0
                        high = max(objetivo_total_efectivo / max(months_house, 1) * 3, 5000.0)
                        final_entrada = 0.0
                        gain_entrada = 0.0
                        tax_entrada = 0.0
                        net_final_entrada = 0.0
                        for _ in range(40):
                            mid = (low + high) / 2
                            net_mid, final_mid, gain_mid, tax_mid = net_final_house_with_monthly(mid)
                            if net_mid < objetivo_total_efectivo:
                                low = mid
                            else:
                                high = mid
                                final_entrada = final_mid
                                gain_entrada = gain_mid
                                tax_entrada = tax_mid
                                net_final_entrada = net_mid
                        mensual_entrada = int(round(high))
                        final_entrada, series_entrada = simulate_constant_plan(
                            current_total=ahorro_actual_entrada,
                            monthly_contribution=mensual_entrada,
                            years=int(years_house),
                            annual_return=annual_return_house,
                            extra_savings=0.0,
                        )
                        principal_total_entrada = ahorro_actual_entrada + mensual_entrada * months_house
                        gain_entrada = max(0.0, final_entrada - principal_total_entrada)
                        tax_entrada = compute_progressive_tax(gain_entrada)
                        net_final_entrada = final_entrada - tax_entrada
                else:
                    # Sin impuestos: objetivo total bruto (entrada + gastos)
                    mensual_entrada = required_constant_monthly_for_goal(
                        current_total=ahorro_actual_entrada,
                        objetivo_final=objetivo_total_efectivo,
                        years=int(years_house),
                        annual_return=annual_return_house,
                        extra_savings=0.0,
                        tax_rate=0.0,
                    )
                    final_entrada, series_entrada = simulate_constant_plan(
                        current_total=ahorro_actual_entrada,
                        monthly_contribution=mensual_entrada,
                        years=int(years_house),
                        annual_return=annual_return_house,
                        extra_savings=0.0,
                    )
                    principal_total_entrada = ahorro_actual_entrada + mensual_entrada * months_house
                    gain_entrada = max(0.0, final_entrada - principal_total_entrada)
                    tax_entrada = 0.0
                    net_final_entrada = final_entrada

                st.subheader("📌 Plan de ahorro para la entrada (aportación constante)")
                objetivo_texto = "NETOS" if apply_tax_house else "brutos"
                st.write(
                    f"Para llegar a un efectivo total de **{objetivo_total_efectivo:,.0f} € {objetivo_texto}** "
                    f"(entrada + gastos) en **{int(years_house)} años**, "
                    f"con una rentabilidad anual estimada del **{anual_return_house_input:.1f}%**, "
                    f"deberías ahorrar/invertir aproximadamente **{mensual_entrada} € al mes** dedicados a este objetivo."
                )

                st.write(
                    f"Patrimonio bruto estimado en la hucha al final: **{final_entrada:,.0f} €**"
                )
                st.write(
                    f"Aportaciones totales realizadas: **{principal_total_entrada:,.0f} €**"
                )
                st.write(
                    f"Plusvalía estimada (beneficio antes de impuestos): **{gain_entrada:,.0f} €**"
                )
                if apply_tax_house:
                    st.write(
                        f"Impuestos estimados sobre plusvalías (según tramos progresivos): **{tax_entrada:,.0f} €**"
                    )
                    st.write(
                        f"Efectivo neto estimado tras impuestos: **{net_final_entrada:,.0f} €**"
                    )

                st.markdown("#### Evolución estimada del ahorro para la entrada")
                df_entrada = pd.DataFrame(
                    {
                        "Año": [m / 12 for m in range(1, len(series_entrada) + 1)],
                        "Ahorro_estimado_€": series_entrada,
                    }
                )
                st.line_chart(df_entrada, x="Año", y="Ahorro_estimado_€")

                st.caption(
                    "Es una simulación sencilla del crecimiento de la 'hucha' para la entrada, "
                    "suponiendo aportaciones constantes y una rentabilidad media estable."
                )

            else:
                # === MODO CRECIENTE ===
                if apply_tax_house:
                    # Buscamos la aportación final mensual para que el efectivo total sea NETO tras impuestos
                    def net_final_house_with_final_monthly(F: float):
                        F_float = float(F)
                        final_val_sim, _ = simulate_dca_ramp(
                            initial_monthly=initial_monthly_house,
                            final_monthly=F_float,
                            years=int(years_house),
                            annual_return=annual_return_house,
                            initial_value=ahorro_actual_entrada,
                        )
                        contrib_total_sim = months_house * (initial_monthly_house + F_float) / 2.0
                        principal_total_sim = ahorro_actual_entrada + contrib_total_sim
                        gain_sim = max(0.0, final_val_sim - principal_total_sim)
                        tax_sim = compute_progressive_tax(gain_sim)
                        net_final_sim = final_val_sim - tax_sim
                        return net_final_sim, final_val_sim, gain_sim, tax_sim

                    net0, _, _, _ = net_final_house_with_final_monthly(initial_monthly_house)
                    if net0 >= objetivo_total_efectivo:
                        final_monthly_house = initial_monthly_house
                        final_entrada_grow, series_entrada_grow = simulate_dca_ramp(
                            initial_monthly=initial_monthly_house,
                            final_monthly=final_monthly_house,
                            years=int(years_house),
                            annual_return=annual_return_house,
                            initial_value=ahorro_actual_entrada,
                        )
                        contrib_total = months_house * (initial_monthly_house + final_monthly_house) / 2.0
                        principal_total_entrada = ahorro_actual_entrada + contrib_total
                        gain_entrada = max(0.0, final_entrada_grow - principal_total_entrada)
                        tax_entrada = compute_progressive_tax(gain_entrada)
                        net_final_entrada = final_entrada_grow - tax_entrada
                    else:
                        low = initial_monthly_house
                        high = max(initial_monthly_house * 3, 5000.0)
                        final_entrada_grow = 0.0
                        gain_entrada = 0.0
                        tax_entrada = 0.0
                        net_final_entrada = 0.0
                        series_entrada_grow = []
                        for _ in range(40):
                            mid = (low + high) / 2
                            net_mid, final_mid, gain_mid, tax_mid = net_final_house_with_final_monthly(mid)
                            if net_mid < objetivo_total_efectivo:
                                low = mid
                            else:
                                high = mid
                                final_entrada_grow = final_mid
                                gain_entrada = gain_mid
                                tax_entrada = tax_mid
                                net_final_entrada = net_mid
                        final_monthly_house = int(round(high))
                        final_entrada_grow, series_entrada_grow = simulate_dca_ramp(
                            initial_monthly=initial_monthly_house,
                            final_monthly=final_monthly_house,
                            years=int(years_house),
                            annual_return=annual_return_house,
                            initial_value=ahorro_actual_entrada,
                        )
                        contrib_total = months_house * (initial_monthly_house + final_monthly_house) / 2.0
                        principal_total_entrada = ahorro_actual_entrada + contrib_total
                        gain_entrada = max(0.0, final_entrada_grow - principal_total_entrada)
                        tax_entrada = compute_progressive_tax(gain_entrada)
                        net_final_entrada = final_entrada_grow - tax_entrada
                else:
                    # Sin impuestos: objetivo total bruto
                    final_monthly_house, _ = required_growing_monthlies_for_goal(
                        current_total=ahorro_actual_entrada,
                        objetivo_final=objetivo_total_efectivo,
                        years=int(years_house),
                        annual_return=annual_return_house,
                        initial_monthly=initial_monthly_house,
                        extra_savings=0.0,
                        tax_rate=0.0,
                    )
                    final_entrada_grow, series_entrada_grow = simulate_dca_ramp(
                        initial_monthly=initial_monthly_house,
                        final_monthly=final_monthly_house,
                        years=int(years_house),
                        annual_return=annual_return_house,
                        initial_value=ahorro_actual_entrada,
                    )
                    contrib_total = months_house * (initial_monthly_house + final_monthly_house) / 2.0
                    principal_total_entrada = ahorro_actual_entrada + contrib_total
                    gain_entrada = max(0.0, final_entrada_grow - principal_total_entrada)
                    tax_entrada = 0.0
                    net_final_entrada = final_entrada_grow

                # Construimos resumen anual de aportaciones
                resumen_anual_house = []
                for año in range(1, int(years_house) + 1):
                    start_idx = (año - 1) * 12
                    end_idx = año * 12 - 1
                    if months_house > 1:
                        start_month = int(
                            round(
                                initial_monthly_house
                                + (final_monthly_house - initial_monthly_house) * (start_idx / (months_house - 1))
                            )
                        )
                        end_month = int(
                            round(
                                initial_monthly_house
                                + (final_monthly_house - initial_monthly_house) * (end_idx / (months_house - 1))
                            )
                        )
                    else:
                        start_month = final_monthly_house
                        end_month = final_monthly_house
                    avg_month = int(round((start_month + end_month) / 2))
                    resumen_anual_house.append(
                        {
                            "Año": año,
                            "Inicio_€/mes": start_month,
                            "Fin_€/mes": end_month,
                            "Media_€/mes": avg_month,
                        }
                    )

                st.subheader("📌 Plan de ahorro para la entrada (aportación creciente)")
                objetivo_texto = "NETOS" if apply_tax_house else "brutos"
                st.write(
                    f"Para llegar a un efectivo total de **{objetivo_total_efectivo:,.0f} € {objetivo_texto}** "
                    f"(entrada + gastos) en **{int(years_house)} años**, "
                    f"con una rentabilidad anual estimada del **{anual_return_house_input:.1f}%**, "
                    f"deberías empezar aportando **{initial_monthly_house} € al mes** y terminar aportando "
                    f"aproximadamente **{final_monthly_house} € al mes** a este objetivo."
                )

                df_resumen_house = pd.DataFrame(resumen_anual_house)
                st.markdown("#### Aportaciones aproximadas por año (plan entrada vivienda)")
                st.dataframe(df_resumen_house)

                st.markdown(
                    "Cada fila representa un año del plan: \n"
                    "- **Inicio_€/mes**: cuánto aportarías al comienzo de ese año.\n"
                    "- **Fin_€/mes**: cuánto aportarías al final de ese año.\n"
                    "- **Media_€/mes**: aportación mensual media aproximada durante ese año."
                )

                st.write(
                    f"Patrimonio bruto estimado en la hucha al final: **{final_entrada_grow:,.0f} €**"
                )
                st.write(
                    f"con una rentabilidad anual estimada del {anual_return_house_input:.1f}%."
)


# ============================
# TAB 4: ANÁLISIS DE CARTERA
# ============================
with tab4:
    st.header("Análisis de cartera a partir del universo de activos")

    st.markdown(
        """
En esta pestaña puedes construir una **cartera de análisis** eligiendo activos
del universo completo (CSV) y asignándoles un valor en euros.

1. Busca un activo por **nombre**, **ISIN** o cualquier palabra clave.  
2. Añádelo a tu cartera de análisis con un valor actual (€).  
3. Cuando tengas varios activos añadidos, pulsa en **Calcular estadísticas** para ver:
   - Distribución por **región**
   - Distribución por **tipo de activo**
   - Distribución por **divisa**
   - Distribución por **subtipo de ETF** (Equity Global, EM Equity, Bond, etc.)
   - Top 10 posiciones por peso
   - Tabla resumen de la cartera con todos los metadatos relevantes
"""
    )

    # Cargamos universo completo desde el CSV grande
    universe_df = load_universe_csv()
    if universe_df.empty:
        st.error(
            "No se ha podido cargar el universo de activos desde 'TradeRepublic_Activos_Completo.csv'. "
            "Asegúrate de que el fichero existe en la misma carpeta que esta app."
        )
    else:
        # Inicializamos la cartera de análisis en sesión
        if "analysis_portfolio" not in st.session_state:
            st.session_state["analysis_portfolio"] = pd.DataFrame(
                columns=[
                    "ISIN",
                    "Name",
                    "Type",
                    "Region",
                    "Country",
                    "ETF_Provider",
                    "ETF_Subtype",
                    "Currency_Name",
                    "Value_€",
                ]
            )

        st.subheader("🔎 Buscar y añadir activos a la cartera de análisis")

        # Desplegable con buscador interno de Streamlit (sin tabla aparte)
        label_df = universe_df.copy()
        label_df["Label"] = label_df.apply(
            lambda r: f"{r.get('Name','')} ({r.get('ISIN','')}) - {r.get('Type','')} {r.get('Region','')}",
            axis=1,
        )

        options = ["(elige un activo)"] + label_df["Label"].tolist()
        selected_label = st.selectbox(
            "Escribe para buscar por nombre/ISIN y selecciona el activo",
            options=options,
            index=0,
            help="Empieza a escribir y usa el buscador interno del desplegable para filtrar.",
        )

        selected_row = None
        if selected_label != "(elige un activo)":
            selected_row = label_df.loc[label_df["Label"] == selected_label].iloc[0]

        col_add1, col_add2 = st.columns(2)
        with col_add1:
            valor_para_anadir = st.number_input(
                "Valor actual (€) a asignar al activo seleccionado",
                min_value=0.0,
                step=100.0,
                value=0.0,
            )
        with col_add2:
            if st.button("➕ Añadir activo a mi cartera de análisis"):
                if selected_row is None:
                    st.error("Primero selecciona un activo de la lista de resultados.")
                elif valor_para_anadir <= 0:
                    st.error("El valor asignado debe ser mayor que 0 €.")
                else:
                    # Construimos una fila con los metadatos relevantes
                    new_row = {
                        "ISIN": selected_row.get("ISIN", ""),
                        "Name": selected_row.get("Name", ""),
                        "Type": selected_row.get("Type", ""),
                        "Region": selected_row.get("Region", ""),
                        "Country": selected_row.get("Country", ""),
                        "ETF_Provider": selected_row.get("ETF_Provider", ""),
                        "ETF_Subtype": selected_row.get("ETF_Subtype", ""),
                        "Currency_Name": selected_row.get("Currency_Name", ""),
                        "Value_€": float(valor_para_anadir),
                    }

                    portfolio_df = st.session_state["analysis_portfolio"].copy()

                    # Si ya existe ese ISIN en la cartera, sumamos al valor existente
                    if not portfolio_df.empty and new_row["ISIN"] in portfolio_df["ISIN"].values:
                        portfolio_df.loc[
                            portfolio_df["ISIN"] == new_row["ISIN"], "Value_€"
                        ] += new_row["Value_€"]
                    else:
                        portfolio_df = pd.concat(
                            [portfolio_df, pd.DataFrame([new_row])],
                            ignore_index=True,
                        )

                    st.session_state["analysis_portfolio"] = portfolio_df
                    st.success(
                        f"Activo '{new_row['Name']}' añadido/actualizado en la cartera de análisis "
                        f"con {new_row['Value_€']:.2f} €."
                    )

        st.markdown("---")
        st.subheader("📂 Cartera de análisis actual")

        portfolio_df = st.session_state["analysis_portfolio"].copy()

        if portfolio_df.empty:
            st.info("Todavía no hay activos en la cartera de análisis.")
        else:
            # Permitimos editar solo la columna de valor para retocar manualmente
            editable_df = portfolio_df.copy()
            editable_df = st.data_editor(
                editable_df,
                column_config={
                    "Value_€": st.column_config.NumberColumn(
                        "Valor actual (€)",
                        min_value=0.0,
                        step=100.0,
                    )
                },
                disabled=[
                    "ISIN",
                    "Name",
                    "Type",
                    "Region",
                    "Country",
                    "ETF_Provider",
                    "ETF_Subtype",
                    "Currency_Name",
                ],
                use_container_width=True,
                key="analysis_portfolio_editor",
            )

            # Actualizamos sesión con posibles cambios en valores
            st.session_state["analysis_portfolio"] = editable_df
            portfolio_df = editable_df

            total_value = float(portfolio_df["Value_€"].sum())
            st.markdown(f"**Valor total de la cartera de análisis:** {total_value:,.2f} €")

            if st.button("📊 Calcular estadísticas de la cartera de análisis"):
                if total_value <= 0:
                    st.error("El valor total de la cartera debe ser mayor que 0 €.")
                else:
                    # Calculamos pesos
                    portfolio_df = portfolio_df.copy()
                    portfolio_df["Weight_%"] = portfolio_df["Value_€"] / total_value * 100.0

                    st.markdown("### 1️⃣ Top 10 posiciones por peso")
                    top10 = portfolio_df.sort_values("Weight_%", ascending=False).head(10)
                    st.dataframe(
                        top10[
                            ["Name", "ISIN", "Type", "Region", "Currency_Name", "Value_€", "Weight_%"]
                        ],
                        use_container_width=True,
                    )

                    # ==========================
                    # DETECCIÓN TEMA STREAMLIT
                    # ==========================
                    theme_base = st.get_option("theme.base")
                    text_color = st.get_option("theme.textColor")
                    if theme_base == "dark" or not text_color:
                        text_color = "#FFFFFF"
                    else:
                        text_color = text_color or "#000000"

                    # ==========================
                    # 2️⃣ Distribución por región
                    # ==========================
                    st.markdown("### 2️⃣ Distribución por región")
                    region_expo = (
                        portfolio_df.groupby("Region", dropna=False)["Value_€"]
                        .sum()
                        .reset_index()
                    )
                    region_expo["Weight_%"] = region_expo["Value_€"] / total_value * 100.0

                    fig_reg, ax_reg = plt.subplots()
                    fig_reg.patch.set_facecolor("none")
                    ax_reg.set_facecolor("none")

                    wedges, texts, autotexts = ax_reg.pie(
                        region_expo["Weight_%"],
                        labels=region_expo["Region"].fillna("Desconocida"),
                        autopct="%1.1f%%",
                        startangle=90,
                    )
                    ax_reg.axis("equal")

                    # Textos del pie adaptados al tema
                    for t in texts + autotexts:
                        t.set_color(text_color)

                    st.pyplot(fig_reg)

                    # ==========================
                    # 3️⃣ Distribución por tipo
                    # ==========================
                    st.markdown("### 3️⃣ Distribución por tipo de activo")
                    type_expo = (
                        portfolio_df.groupby("Type", dropna=False)["Value_€"]
                        .sum()
                        .reset_index()
                    )
                    type_expo["Weight_%"] = type_expo["Value_€"] / total_value * 100.0

                    fig_type, ax_type = plt.subplots()
                    fig_type.patch.set_facecolor("none")
                    ax_type.set_facecolor("none")

                    ax_type.bar(
                        type_expo["Type"].fillna("Desconocido"),
                        type_expo["Weight_%"],
                    )
                    ax_type.set_ylabel("% de la cartera")
                    ax_type.set_xlabel("Tipo de activo")
                    plt.xticks(rotation=30, ha="right")

                    # Colores de texto y ejes según tema
                    ax_type.tick_params(colors=text_color)
                    ax_type.yaxis.label.set_color(text_color)
                    ax_type.xaxis.label.set_color(text_color)
                    for spine in ax_type.spines.values():
                        spine.set_color(text_color)

                    st.pyplot(fig_type)

                    # ==========================
                    # 4️⃣ Distribución por divisa
                    # ==========================
                    st.markdown("### 4️⃣ Distribución por divisa")
                    currency_expo = (
                        portfolio_df.groupby("Currency_Name", dropna=False)["Value_€"]
                        .sum()
                        .reset_index()
                    )
                    currency_expo["Weight_%"] = currency_expo["Value_€"] / total_value * 100.0

                    fig_cur, ax_cur = plt.subplots()
                    fig_cur.patch.set_facecolor("none")
                    ax_cur.set_facecolor("none")

                    ax_cur.bar(
                        currency_expo["Currency_Name"].fillna("Desconocida"),
                        currency_expo["Weight_%"],
                    )
                    ax_cur.set_ylabel("% de la cartera")
                    ax_cur.set_xlabel("Divisa")
                    plt.xticks(rotation=30, ha="right")

                    ax_cur.tick_params(colors=text_color)
                    ax_cur.yaxis.label.set_color(text_color)
                    ax_cur.xaxis.label.set_color(text_color)
                    for spine in ax_cur.spines.values():
                        spine.set_color(text_color)

                    st.pyplot(fig_cur)

                    # ==========================
                    # 5️⃣ Distribución subtipo ETF
                    # ==========================
                    st.markdown("### 5️⃣ Distribución por subtipo de ETF (solo ETFs)")
                    etf_only = portfolio_df[portfolio_df["Type"] == "ETF"].copy()
                    if etf_only.empty:
                        st.info("No hay ETFs en esta cartera de análisis, así que no puede mostrarse esta distribución.")
                    else:
                        etf_sub_expo = (
                            etf_only.groupby("ETF_Subtype", dropna=False)["Value_€"]
                            .sum()
                            .reset_index()
                        )
                        etf_sub_expo["Weight_%"] = etf_sub_expo["Value_€"] / total_value * 100.0

                        fig_sub, ax_sub = plt.subplots()
                        fig_sub.patch.set_facecolor("none")
                        ax_sub.set_facecolor("none")

                        ax_sub.bar(
                            etf_sub_expo["ETF_Subtype"].fillna("Sin clasificar"),
                            etf_sub_expo["Weight_%"],
                        )
                        ax_sub.set_ylabel("% de la cartera")
                        ax_sub.set_xlabel("Subtipo de ETF")
                        plt.xticks(rotation=30, ha="right")

                        ax_sub.tick_params(colors=text_color)
                        ax_sub.yaxis.label.set_color(text_color)
                        ax_sub.xaxis.label.set_color(text_color)
                        for spine in ax_sub.spines.values():
                            spine.set_color(text_color)

                        st.pyplot(fig_sub)

                    # ==========================
                    # 6️⃣ Tabla resumen completa
                    # ==========================
                    st.markdown("### 6️⃣ Tabla resumen completa de la cartera")
                    st.dataframe(
                        portfolio_df[
                            [
                                "Name",
                                "ISIN",
                                "Type",
                                "Region",
                                "Country",
                                "ETF_Provider",
                                "ETF_Subtype",
                                "Currency_Name",
                                "Value_€",
                                "Weight_%"
                            ]
                        ],
                        use_container_width=True,
                    )

    # --- Reset TAB 4 ---
    st.markdown("---")
    if st.button("🔄 Restablecer análisis", key="reset_tab4"):
        if "analysis_portfolio" in st.session_state:
            del st.session_state["analysis_portfolio"]
        st.rerun()
