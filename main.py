import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io

st.set_page_config(page_title="Simulación de Interés Compuesto", layout="wide")

def map_sqx_csv_to_standard(df):
    rename_dict = {
        "Open time": "Open Time",
        "Close time": "Close Time",
        "Open price": "Open Price",
        "Close price": "Close Price",
        "Symbol": "Item",
        "Profit/Loss": "Profit",
        "profit/loss": "Profit",
        "Profit": "Profit",
    }
    columns_renamed = {k: v for k, v in rename_dict.items() if k in df.columns}
    df = df.rename(columns=columns_renamed)
    return df

def format_miles(n):
    try:
        return "{:,.0f}".format(n).replace(',', 'X').replace('.', ',').replace('X', '.')
    except:
        return n if n is not None else ""

riesgo_opciones = [
    "Balance actual (cada trade)",
    "Marca de agua (max histórico) (cada trade)",
    "Balance (día 1 de cada mes)",
    "Marca de agua (día 1 de cada mes)",
    "Balance (días 1 y 15 de cada mes)",
    "Marca de agua (días 1 y 15 de cada mes)"
]

st.title("Simulación de Interés Compuesto sobre Trades Históricos")

# =================== NUEVO BLOQUE DE CARGA ===================
df = None
origen = None

# Carga múltiple de CSV en un porfolio
uploaded_csvs = st.file_uploader(
    "Carga uno o varios CSV (portfolio) list of trades del SQX",
    type=["csv"],
    key="uploaded_csvs",
    accept_multiple_files=True
)
uploaded_xlsx = st.file_uploader(
    "O carga tu EXCEL con trades del programa ANALIZA FACIL (solo uno)",
    type=["xlsx"],
    key="uploaded_xlsx"
)

if uploaded_xlsx is not None:
    df = pd.read_excel(uploaded_xlsx)
    origen = "Excel"
elif uploaded_csvs:  # Uno o más csv
    dfs = []
    for csvfile in uploaded_csvs:
        tmpdf = pd.read_csv(csvfile, sep=None, engine='python')
        dfs.append(tmpdf)
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        origen = f"{len(dfs)} CSV" if len(dfs) > 1 else "CSV"

if origen:
    st.info(f"Archivo cargado: {origen}")

# =============== FIN CARGA =========================

# Variable oculta para "valor_inicial"
if 'valor_inicial' not in st.session_state:
    st.session_state['valor_inicial'] = 200.0
valor_inicial = st.session_state['valor_inicial']

colA, colB, colD = st.columns([2, 2, 4])
with colA:
    balance_ini = st.number_input("Balance Inicial", min_value=1000, max_value=1_000_000, value=10_000, step=100)
with colB:
    riesgo_pc = st.number_input("Riesgo por trade (%) y POR ROBOT!!", min_value=0.1, max_value=30.0, value=2.0, step=0.1)
with colD:
    base_riesgo = st.selectbox(
        "Tipo de base para riesgo variable",
        options=riesgo_opciones,
        index=0
    )

date_col = None
if df is not None:
    df.columns = df.columns.str.strip()
    df = map_sqx_csv_to_standard(df)

    if "Profit" not in df.columns:
        st.error(
            f"No se encuentra ninguna columna de 'Profit' o 'Profit/Loss' (columnas importadas: {list(df.columns)})")
        st.stop()

    df["Profit"] = pd.to_numeric(df["Profit"], errors="coerce")
    df = df.dropna(subset=["Profit"])
    if df.empty:
        st.error("El archivo no tiene registros válidos en la columna 'Profit'.")
        st.stop()

    if "Close Time" in df.columns:
        df["Close Time"] = pd.to_datetime(df["Close Time"], errors="coerce")
        df = df.dropna(subset=["Close Time"])
        df = df.sort_values("Close Time")
        date_col = "Close Time"
    elif "Open Time" in df.columns:
        df["Open Time"] = pd.to_datetime(df["Open Time"], errors="coerce")
        df = df.dropna(subset=["Open Time"])
        df = df.sort_values("Open Time")
        date_col = "Open Time"

    if date_col:
        min_fecha, max_fecha = df[date_col].min(), df[date_col].max()
        colf1, colf2 = st.columns(2)
        with colf1:
            fecha_ini = st.date_input("Filtro inicio (fecha)", min_fecha, min_value=min_fecha, max_value=max_fecha)
        with colf2:
            fecha_fin = st.date_input("Filtro fin (fecha)", max_fecha, min_value=min_fecha, max_value=max_fecha)
        mask = (df[date_col] >= pd.Timestamp(fecha_ini)) & (df[date_col] <= pd.Timestamp(fecha_fin))
        df = df[mask]
        if df.empty:
            st.warning("No hay trades en ese rango de fechas.")
    else:
        fecha_ini = None
        fecha_fin = None

def calcular_simulacion(df, balance_ini, riesgo_pc, valor_inicial, periodo, base_riesgo, date_col=None):
    balances = [balance_ini]
    riesgos = [valor_inicial]
    profits = []
    max_balances = [balance_ini]
    drawdowns = [0]
    actualizar_riesgo_indices = set()
    if periodo == 0 or not date_col:
        actualizar_riesgo_indices = set(df.index)
    else:
        fechas = df[date_col].dt.normalize()
        if periodo == 1:
            actualizar_riesgo_indices = set(fechas[fechas.dt.day == 1].index)
        elif periodo == 2:
            actualizar_riesgo_indices = set(fechas[(fechas.dt.day == 1) | (fechas.dt.day == 15)].index)
    last_risk = valor_inicial
    for idx, row in df.iterrows():
        balance = balances[-1]
        prev_max_balance = max_balances[-1]
        if valor_inicial == 0:
            R_multiple = 0
        else:
            R_multiple = row["Profit"] / valor_inicial
        recompute_risk = (idx in actualizar_riesgo_indices or idx == 0)
        if recompute_risk:
            if base_riesgo == 0:
                riesgo_base = balance
            else:
                riesgo_base = prev_max_balance
            last_risk = max(riesgo_base * riesgo_pc / 100, 20)
        riesgo = last_risk
        profit_this = riesgo * R_multiple
        new_balance = balance + profit_this
        balances.append(new_balance)
        max_balance = max(prev_max_balance, new_balance)
        max_balances.append(max_balance)
        drawdown = max_balance - new_balance
        drawdowns.append(drawdown)
        riesgos.append(riesgo)
        profits.append(profit_this)
    return balances[1:], riesgos[1:], profits, max_balances[1:], drawdowns[1:]

def max_drawdown_percent(balances:list):
    array = np.array(balances)
    if len(array) == 0:
        return np.nan
    vals, idxs = [], []
    max_so_far = array[0]
    min_pct = 0
    for i in range(1, len(array)):
        max_so_far = max(max_so_far, array[i])
        pct = (array[i] - max_so_far) / max_so_far if max_so_far != 0 else 0
        min_pct = min(min_pct, pct)
    return abs(min_pct)*100

######################
######  L1 ###########

if df is not None and not df.empty:
    st.markdown("<h2 style='margin-top:32px; margin-bottom:0.2em;'>Laboratorio 1: Simulación principal</h2>", unsafe_allow_html=True)
    col_lab1_btn = st.columns([2,2,2,2,2])
    lab1_boton = col_lab1_btn[0].button("Laboratorio 1: Simulación principal", type="primary")
    lab1_res = st.session_state.get("lab1_resultados", False)

    if lab1_boton:
        st.session_state["lab1_resultados"] = True
        lab1_res = True

    if lab1_res:
        idx_riesgo_sel = riesgo_opciones.index(base_riesgo)
        if idx_riesgo_sel in [0, 1]:
            periodo_real, base_real = 0, idx_riesgo_sel
        elif idx_riesgo_sel in [2, 3]:
            periodo_real, base_real = 1, idx_riesgo_sel - 2
        elif idx_riesgo_sel in [4, 5]:
            periodo_real, base_real = 2, idx_riesgo_sel - 4

        df_cal1 = df.copy()
        df_cal1["R_multiple"] = df_cal1["Profit"] / valor_inicial
        balances, riesgos, profits, max_balances, drawdowns = calcular_simulacion(
            df_cal1, balance_ini, riesgo_pc, valor_inicial, periodo_real, base_real, date_col
        )
        df_cal1["Balance"] = balances
        df_cal1["Riesgo_usado"] = riesgos
        df_cal1["Profit_recalculado"] = profits
        df_cal1["Max_balance"] = max_balances
        df_cal1["Drawdown"] = drawdowns

        if df_cal1["Balance"].min() < (balance_ini * 0.1):
            st.markdown(
                """
                <div style='background-color: #ff1744; color: white; font-size: 2.2em; font-weight: bold; 
                    text-align: center; padding: 16px; border-radius: 8px; margin-bottom: 20px;'>
                    ⚠️ CUENTA QUEMADA ⚠️<br>
                    El balance cayó por debajo del 10% del capital inicial.
                </div>
                """, unsafe_allow_html=True
            )

        with st.expander("Resultados del Laboratorio 1", expanded=False):
            profit_total = df_cal1["Profit_recalculado"].sum()
            dd_max = df_cal1["Drawdown"].max()
            ret_dd = profit_total / dd_max if dd_max > 0 else np.nan

            col1, col2, col3 = st.columns([3, 3, 3])
            with col1:
                st.markdown("#### Profit total")
                st.markdown(
                    f"<div style='font-size:2.6em; font-weight:bold; text-align:center;'>${profit_total:,.2f}</div>",
                    unsafe_allow_html=True)
            with col2:
                st.markdown("#### Drawdown Máximo")
                st.markdown(f"<div style='font-size:2.6em; font-weight:bold; text-align:center;'>${dd_max:,.2f}</div>",
                            unsafe_allow_html=True)
            with col3:
                st.markdown("#### Retorno/DD (RetDD)")
                st.markdown(f"<div style='font-size:2.6em; font-weight:bold; text-align:center;'>{ret_dd:.2f}</div>",
                            unsafe_allow_html=True)

            with st.expander("📈 Evolución del Balance (Interés Compuesto)", expanded=False):
                if date_col:
                    fig = px.line(df_cal1, x=date_col, y="Balance", labels={date_col: "Fecha", "Balance": "Balance"})
                else:
                    fig = px.line(df_cal1, y="Balance", labels={"index": "Trade", "Balance": "Balance"})
                st.plotly_chart(fig, use_container_width=True)

            with st.expander("📉 Evolución del Drawdown", expanded=False):
                if date_col:
                    fig_dd = px.line(df_cal1, x=date_col, y="Drawdown", labels={date_col: "Fecha", "Drawdown": "Drawdown"})
                else:
                    fig_dd = px.line(df_cal1, y="Drawdown", labels={"index": "Trade", "Drawdown": "Drawdown"})
                st.plotly_chart(fig_dd, use_container_width=True)

            with st.expander("Ver detalles tabla", expanded=False):
                base_cols = []
                if "Ticket" in df_cal1.columns: base_cols.append("Ticket")
                if "Symbol" in df_cal1.columns: base_cols.append("Symbol")
                if date_col: base_cols.append(date_col)
                base_cols += ["Profit", "Balance", "Drawdown", "Riesgo_usado", "Max_balance"]
                base_cols = [c for c in base_cols if c in df_cal1.columns]
                st.dataframe(df_cal1[base_cols], use_container_width=True)

            excel_buffer = io.BytesIO()
            df_cal1[base_cols].to_excel(excel_buffer, index=False, engine='openpyxl')
            excel_buffer.seek(0)
            st.download_button(
                label="Descargar resultados (Excel)",
                data=excel_buffer,
                file_name="interes_compuesto_trades.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    # Separación visual
    st.markdown(
        "<div style='height: 36px;'></div>"
        "<hr style='height:3px;border:none;background:linear-gradient(90deg,#b2ebf2,#e3f2fd,#b2ebf2);'>"
        "<div style='height: 22px;'></div>",
        unsafe_allow_html=True
    )
###################################
#########  L2 #####################

    st.markdown("<h2 style='margin-bottom: 0.5em;'>Laboratorio 2: Tablas comparativas por año</h2>",
                unsafe_allow_html=True)
    riesgos_col1, riesgos_col2, riesgos_col3, riesgos_col4 = st.columns(4)
    with riesgos_col1:
        riesgo_1 = st.number_input("Riesgo 1 (%)", min_value=0.1, max_value=50.0, value=20.0, key="l2_riesgo_1")
    with riesgos_col2:
        riesgo_2 = st.number_input("Riesgo 2 (%)", min_value=0.1, max_value=50.0, value=15.0, key="l2_riesgo_2")
    with riesgos_col3:
        riesgo_3 = st.number_input("Riesgo 3 (%)", min_value=0.1, max_value=50.0, value=10.0, key="l2_riesgo_3")
    with riesgos_col4:
        riesgo_4 = st.number_input("Riesgo 4 (%)", min_value=0.1, max_value=50.0, value=6.0, key="l2_riesgo_4")

    robots = st.number_input("nº de robots", min_value=1, value=1, step=1,
                             help="Esta variable solo afecta al label de los encabezados, no a los cálculos.")

    st.markdown(
        "<div style='font-size:1.2em;font-weight:bold;color:#1976D2;margin:14px 0 10px 0'>"
        "Riesgos comparativos POR ROBOT <span style='color:#333;font-size:0.92em;'>(si tiene x robots el riesgo se multiplicará por x)</span>"
        "</div>",
        unsafe_allow_html=True
    )

    if date_col is not None:
        df["__year"] = pd.to_datetime(df[date_col]).dt.year
    else:
        # fallback: si no hay fecha, simulate un año "0"
        df["__year"] = 0
    boton_lab2 = st.button("Calcula Profit y DD", type="primary")

    def build_cmp_table(tabla, columnas, years, quemadas_tabla, porcentaje=False):
        html = '<table style="border-collapse: collapse; width: 100%; font-size: 1.02em;"><thead><tr>'
        for col in columnas:
            html += f'<th style="border:1px solid #bbb; padding:6px">{col}</th>'
        html += "</tr></thead><tbody>"
        for i, y in enumerate(years):
            html += "<tr>"
            html += f'<td style="border:1px solid #bbb; padding:6px; text-align:center">{y}</td>'
            html += f'<td style="border:1px solid #bbb; padding:6px; text-align:center;">{format_miles(tabla[0][i]) if pd.notnull(tabla[0][i]) else ""}</td>'
            for j in range(len(riesgos_tabla)):
                for k in range(2):
                    idx = 1 + 2 * j + k
                    val = tabla[idx][i]
                    quemada = quemadas_tabla[idx - 1][i]
                    if quemada:
                        html += '<td style="border:1px solid #bbb; text-align:center; color:red; font-weight:bold; background:#fff5f5">QUEMADA</td>'
                    else:
                        if porcentaje and pd.notnull(val):
                            html += f'<td style="border:1px solid #bbb; text-align:center;">{round(val,2)}%</td>'
                        else:
                            html += f'<td style="border:1px solid #bbb; text-align:center;">{format_miles(val) if pd.notnull(val) else ""}</td>'
            html += "</tr>"
        # Sumatorio: para profit suma total, para DD muestra máximo (peor DD)
        html += "<tr style='background:#EFF6FA; font-weight:bold'>"
        html += "<td style='border:1px solid #bbb; text-align:center;'>Total</td>"
        if porcentaje:
            max_total = max([x for x in tabla[0] if (x is not None and not pd.isnull(x))])
            html += f"<td style='border:1px solid #bbb; text-align:center;font-weight:bold'>{round(max_total,2)}%</td>"
        else:
            sum_profit = sum(x for x in tabla[0] if (x is not None and not pd.isnull(x)))
            html += f"<td style='border:1px solid #bbb; text-align:center;'>{format_miles(sum_profit)}</td>"
        for col_idx in range(1, len(tabla)):
            if any(q for q in quemadas_tabla[col_idx - 1]):
                html += "<td style='border:1px solid #bbb; text-align:center; color:#bbb;'></td>"
            else:
                vals = [tabla[col_idx][i] for i in range(len(years)) if pd.notnull(tabla[col_idx][i])]
                if porcentaje and vals:
                    maxx = max(vals)
                    html += f"<td style='border:1px solid #bbb; text-align:center;font-weight:bold'>{round(maxx,2)}%</td>"
                else:
                    suma = sum(x for x in vals if (x is not None and not pd.isnull(x)))
                    html += f"<td style='border:1px solid #bbb; text-align:center; font-weight:bold'>{format_miles(suma)}</td>"
        html += "</tr>"
        html += "</tbody></table>"
        return html

    # Estas funciones pueden mantenerse igual
    def profit_simple_con_balanceini(df_filtrado, balance_ini):
        balance = balance_ini
        for p in df_filtrado["Profit"]:
            balance += p
        return balance - balance_ini if not df_filtrado.empty else None

    def profit_periodo_base(riesgo, periodo, base):
        profits_by_year = []
        quemadas = []
        for y in years:
            df_y = df[df["__year"] == y]
            if df_y.empty:
                profits_by_year.append(np.nan)
                quemadas.append(False)
            else:
                balances_y, _, profits_y, _, _ = calcular_simulacion(
                    df_y, balance_ini, riesgo, valor_inicial, periodo, base, date_col)
                profits_by_year.append(np.sum(profits_y))
                quemadas.append(np.min(balances_y) <= 0)
        return profits_by_year, quemadas

    def dd_periodo_base(riesgo, periodo, base):
        dds_by_year = []
        quemadas = []
        for y in years:
            df_y = df[df["__year"] == y]
            if df_y.empty:
                dds_by_year.append(np.nan)
                quemadas.append(False)
            else:
                balances_y, _, _, _, _ = calcular_simulacion(
                    df_y, balance_ini, riesgo, valor_inicial, periodo, base, date_col)
                max_ddpct = max_drawdown_percent(balances_y)
                dds_by_year.append(max_ddpct)
                quemadas.append(np.min(balances_y) <= 0)
        return dds_by_year, quemadas


    if boton_lab2:
        # Definir columna año, años y riesgos
        if date_col is not None:
            df["__year"] = pd.to_datetime(df[date_col]).dt.year
        else:
            df["__year"] = 0
        years = sorted(df["__year"].dropna().unique())
        riesgos_tabla = [riesgo_1, riesgo_2, riesgo_3, riesgo_4]

        columnas_tabla1 = ["Año", "Profit simple"]
        for r in riesgos_tabla:
            porcentaje = int(r * robots)
            columnas_tabla1 += [f"{porcentaje}% Balance", f"{porcentaje}% Watermark"]
        columnas_tabla2 = columnas_tabla1[:]
        columnas_tabla3 = columnas_tabla1[:]

        #--- PROFIT
        profit_simple_per_year = []
        idxs = df.groupby("__year").indices
        for y in years:
            if y in idxs:
                dfx = df.iloc[idxs[y]]
                profit_simple_y = profit_simple_con_balanceini(dfx, balance_ini)
                profit_simple_per_year.append(profit_simple_y)
            else:
                profit_simple_per_year.append(None)

        # ---T1
        tabla1 = [profit_simple_per_year]
        quemadas_tabla1 = []
        for riesgo in riesgos_tabla:
            vals, q = profit_periodo_base(riesgo, periodo=0, base=0)
            tabla1.append(vals)
            quemadas_tabla1.append(q)
            vals, q = profit_periodo_base(riesgo, periodo=0, base=1)
            tabla1.append(vals)
            quemadas_tabla1.append(q)
        st.markdown("#### 1ª Tabla: Cada trade (Profit)")
        st.markdown(build_cmp_table(tabla1, columnas_tabla1, years, quemadas_tabla1), unsafe_allow_html=True)

        # ---T2
        tabla2 = [profit_simple_per_year]
        quemadas_tabla2 = []
        for riesgo in riesgos_tabla:
            vals, q = profit_periodo_base(riesgo, periodo=1, base=0)
            tabla2.append(vals)
            quemadas_tabla2.append(q)
            vals, q = profit_periodo_base(riesgo, periodo=1, base=1)
            tabla2.append(vals)
            quemadas_tabla2.append(q)
        st.markdown("#### 2ª Tabla: Día 1 de cada mes (Profit)")
        st.markdown(build_cmp_table(tabla2, columnas_tabla2, years, quemadas_tabla2), unsafe_allow_html=True)

        # ---T3
        tabla3 = [profit_simple_per_year]
        quemadas_tabla3 = []
        for riesgo in riesgos_tabla:
            vals, q = profit_periodo_base(riesgo, periodo=2, base=0)
            tabla3.append(vals)
            quemadas_tabla3.append(q)
            vals, q = profit_periodo_base(riesgo, periodo=2, base=1)
            tabla3.append(vals)
            quemadas_tabla3.append(q)
        st.markdown("#### 3ª Tabla: Día 1 y 15 cada mes (Profit)")
        st.markdown(build_cmp_table(tabla3, columnas_tabla3, years, quemadas_tabla3), unsafe_allow_html=True)

        st.markdown("---")

        #--- DD
        # ---T1 DD
        tabla1dd = [[max_drawdown_percent([balance_ini+profit_simple_per_year[i] if profit_simple_per_year[i] is not None else balance_ini  for i in range(len(years)) ])] * len(years)]
        quemadas_tabla1dd = []
        for riesgo in riesgos_tabla:
            vals, q = dd_periodo_base(riesgo, periodo=0, base=0)
            tabla1dd.append(vals)
            quemadas_tabla1dd.append(q)
            vals, q = dd_periodo_base(riesgo, periodo=0, base=1)
            tabla1dd.append(vals)
            quemadas_tabla1dd.append(q)
        st.markdown("#### 1ª Tabla: Cada trade (DD%)")
        st.markdown(build_cmp_table(tabla1dd, columnas_tabla1, years, quemadas_tabla1dd, porcentaje=True), unsafe_allow_html=True)

        # ---T2 DD
        tabla2dd = [[max_drawdown_percent([balance_ini+profit_simple_per_year[i] if profit_simple_per_year[i] is not None else balance_ini  for i in range(len(years)) ])] * len(years)]
        quemadas_tabla2dd = []
        for riesgo in riesgos_tabla:
            vals, q = dd_periodo_base(riesgo, periodo=1, base=0)
            tabla2dd.append(vals)
            quemadas_tabla2dd.append(q)
            vals, q = dd_periodo_base(riesgo, periodo=1, base=1)
            tabla2dd.append(vals)
            quemadas_tabla2dd.append(q)
        st.markdown("#### 2ª Tabla: Día 1 de cada mes (DD%)")
        st.markdown(build_cmp_table(tabla2dd, columnas_tabla2, years, quemadas_tabla2dd, porcentaje=True), unsafe_allow_html=True)

        # ---T3 DD
        tabla3dd = [[max_drawdown_percent([balance_ini+profit_simple_per_year[i] if profit_simple_per_year[i] is not None else balance_ini  for i in range(len(years)) ])] * len(years)]
        quemadas_tabla3dd = []
        for riesgo in riesgos_tabla:
            vals, q = dd_periodo_base(riesgo, periodo=2, base=0)
            tabla3dd.append(vals)
            quemadas_tabla3dd.append(q)
            vals, q = dd_periodo_base(riesgo, periodo=2, base=1)
            tabla3dd.append(vals)
            quemadas_tabla3dd.append(q)
        st.markdown("#### 3ª Tabla: Día 1 y 15 cada mes (DD%)")
        st.markdown(build_cmp_table(tabla3dd, columnas_tabla3, years, quemadas_tabla3dd, porcentaje=True), unsafe_allow_html=True)

        # ---- Excel ----
        import xlsxwriter
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            workbook = writer.book
            worksheet = workbook.add_worksheet('Laboratorio 2')
            writer.sheets['Laboratorio 2'] = worksheet
            bold_center = workbook.add_format({'bold': True, 'align': 'center', 'valign': 'vcenter'})
            center = workbook.add_format({'align': 'center', 'valign': 'vcenter'})
            title_format = workbook.add_format({'bold': True, 'align': 'left', 'valign': 'vcenter', 'font_size': 13, 'bg_color': '#F3F7FD'})
            total_format = workbook.add_format({'bg_color': '#EFF6FA', 'bold': True, 'align': 'center', 'valign': 'vcenter', 'num_format': '#,##0'})
            miles_format = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'num_format': '#,##0'})
            percent_format = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'num_format': '0.00%'})
            max_cols = max(len(columnas_tabla1), len(columnas_tabla2), len(columnas_tabla3))

            def value_for_xlsx(val, quemada, porcentaje=False):
                if quemada:
                    return "QUEMADA"
                if pd.isnull(val):
                    return ""
                try:
                    if porcentaje:
                        return float(val)
                    else:
                        intval = int(round(float(val)))
                        return intval
                except:
                    return val

            def write_table(ws, df, startrow, title, porcentaje=False):
                ws.merge_range(startrow, 0, startrow, max_cols - 1, title, title_format)
                startrow += 1
                for j, col in enumerate(df.columns):
                    ws.write(startrow, j, col, bold_center)
                for i, row in enumerate(df.values):
                    for j, val in enumerate(row):
                        if porcentaje and (isinstance(val, float) or isinstance(val, int)) and not pd.isnull(val):
                            ws.write(startrow+1+i, j, val/100, percent_format)
                        elif (isinstance(val, (int, float)) and not isinstance(val, bool) and not pd.isnull(val)):
                            try:
                                intval = int(round(float(val)))
                                ws.write(startrow + 1 + i, j, intval, miles_format)
                            except:
                                ws.write(startrow + 1 + i, j, val, center)
                        else:
                            ws.write(startrow + 1 + i, j, val, center)
                # Total row
                total_row = ["Total"]
                if porcentaje:
                    try:
                        maxy = max([x for x in df.iloc[:,1] if x not in ("", "QUEMADA") and not pd.isnull(x)])
                        total_row.append(maxy)
                    except:
                        total_row.append("")
                else:
                    try:
                        total_row.append(int(round(df.iloc[:,1].replace(["QUEMADA", ""], np.nan).astype(float).sum())))
                    except:
                        total_row.append("")
                for c in range(2, len(df.columns)):
                    col_vals = df.iloc[:, c]
                    if "QUEMADA" in col_vals.values:
                        total_row.append("")
                    else:
                        if porcentaje:
                            try:
                                total_row.append(max([x for x in col_vals if x not in ("", "QUEMADA") and not pd.isnull(x)]) )
                            except:
                                total_row.append("")
                        else:
                            try:
                                total_row.append(int(round(col_vals.replace(["QUEMADA", ""], np.nan).astype(float).sum())))
                            except:
                                total_row.append("")
                for j, val in enumerate(total_row):
                    if porcentaje and (isinstance(val,float) or isinstance(val,int)):
                        ws.write(startrow + 1 + len(df), j, val/100, percent_format)
                    elif isinstance(val, (int, float)) and not isinstance(val, bool):
                        ws.write(startrow + 1 + len(df), j, val, total_format)
                    else:
                        ws.write(startrow + 1 + len(df), j, val, total_format)
                return startrow + 1 + len(df) + 2

            # DATAFRAMES PROFIT
            df1 = pd.DataFrame(
                [[y] + [int(round(tabla1[0][i])) if pd.notnull(tabla1[0][i]) else ""] +
                 [value_for_xlsx(tabla1[1 + j][i], quemadas_tabla1[j][i]) for j in range(len(tabla1) - 1)]
                 for i, y in enumerate(years)],
                columns=columnas_tabla1
            )
            df2 = pd.DataFrame(
                [[y] + [int(round(tabla2[0][i])) if pd.notnull(tabla2[0][i]) else ""] +
                 [value_for_xlsx(tabla2[1 + j][i], quemadas_tabla2[j][i]) for j in range(len(tabla2) - 1)]
                 for i, y in enumerate(years)],
                columns=columnas_tabla2
            )
            df3 = pd.DataFrame(
                [[y] + [int(round(tabla3[0][i])) if pd.notnull(tabla3[0][i]) else ""] +
                 [value_for_xlsx(tabla3[1 + j][i], quemadas_tabla3[j][i]) for j in range(len(tabla3) - 1)]
                 for i, y in enumerate(years)],
                columns=columnas_tabla3
            )
            # DATAFRAMES DD
            df1dd = pd.DataFrame(
                [[y] + [tabla1dd[0][i]] +
                 [value_for_xlsx(tabla1dd[1 + j][i], quemadas_tabla1dd[j][i], porcentaje=True) for j in range(len(tabla1dd) - 1)]
                 for i, y in enumerate(years)],
                columns=columnas_tabla1
            )
            df2dd = pd.DataFrame(
                [[y] + [tabla2dd[0][i]] +
                 [value_for_xlsx(tabla2dd[1 + j][i], quemadas_tabla2dd[j][i], porcentaje=True) for j in range(len(tabla2dd) - 1)]
                 for i, y in enumerate(years)],
                columns=columnas_tabla2
            )
            df3dd = pd.DataFrame(
                [[y] + [tabla3dd[0][i]] +
                 [value_for_xlsx(tabla3dd[1 + j][i], quemadas_tabla3dd[j][i], porcentaje=True) for j in range(len(tabla3dd) - 1)]
                 for i, y in enumerate(years)],
                columns=columnas_tabla3
            )

            nextrow = 0
            nextrow = write_table(worksheet, df1, nextrow, "1ª Tabla: Cada trade (Profit)", porcentaje=False)
            nextrow = write_table(worksheet, df2, nextrow, "2ª Tabla: Día 1 de cada mes (Profit)", porcentaje=False)
            nextrow = write_table(worksheet, df3, nextrow, "3ª Tabla: Día 1 y 15 cada mes (Profit)", porcentaje=False)
            nextrow = write_table(worksheet, df1dd, nextrow, "1ª Tabla: Cada trade (DD%)", porcentaje=True)
            nextrow = write_table(worksheet, df2dd, nextrow, "2ª Tabla: Día 1 de cada mes (DD%)", porcentaje=True)
            nextrow = write_table(worksheet, df3dd, nextrow, "3ª Tabla: Día 1 y 15 cada mes (DD%)", porcentaje=True)

            for col in range(max_cols):
                worksheet.set_column(col, col, 18)

        output.seek(0)
        st.download_button(
            label="Descargar resultados Laboratorio 2 (Excel)",
            data=output,
            file_name="laboratorio2_resultados.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

else:
    st.info("Sube un archivo CSV o Excel para comenzar.")