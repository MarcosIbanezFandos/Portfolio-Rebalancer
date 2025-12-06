import pdfplumber
import re
import pandas as pd
import json
import os

# --- CONFIGURACIÓN DE ARCHIVOS ---
INPUT_PDF = "Instrument_Universe_DE_en.pdf"
OUTPUT_EXCEL = "TradeRepublic_Activos_Completo.xlsx"
OUTPUT_CSV = "TradeRepublic_Activos_Completo.csv"
OUTPUT_JSON = "TradeRepublic_Activos_Completo.json"

# --- 1. BASES DE DATOS Y MAPAS ---

# Lista de Proveedores de ETF comunes para detección
ETF_PROVIDERS = [
    "iShares", "Vanguard", "Amundi", "Xtrackers", "Lyxor", "SPDR", 
    "Invesco", "WisdomTree", "UBS", "L&G", "HSBC", "JPM", "Franklin",
    "VanEck", "Global X", "Ossiam", "BNP Paribas", "Deka", "PIMCO", 
    "Fidelity", "First Trust", "State Street"
]

# Mapa ISIN -> País (Extendido)
COUNTRY_MAP = {
    "AN": "Netherlands Antilles", "AT": "Austria", "AU": "Australia",
    "BE": "Belgium", "BG": "Bulgaria", "BM": "Bermuda", "BR": "Brazil",
    "BS": "Bahamas", "CA": "Canada", "CH": "Switzerland", "CN": "China",
    "CY": "Cyprus", "CZ": "Czech Republic", "DE": "Germany", "DK": "Denmark",
    "EE": "Estonia", "ES": "Spain", "FI": "Finland", "FO": "Faroe Islands",
    "FR": "France", "GB": "United Kingdom", "GG": "Guernsey", "GI": "Gibraltar",
    "GR": "Greece", "HK": "Hong Kong", "HU": "Hungary", "ID": "Indonesia",
    "IE": "Ireland", "IL": "Israel", "IM": "Isle of Man", "IT": "Italy",
    "JE": "Jersey", "JP": "Japan", "KY": "Cayman Islands", "LI": "Liechtenstein",
    "LR": "Liberia", "LT": "Lithuania", "LU": "Luxembourg", "LV": "Latvia",
    "MA": "Morocco", "MC": "Monaco", "MH": "Marshall Islands", "MT": "Malta",
    "MU": "Mauritius", "MX": "Mexico", "NL": "Netherlands", "NO": "Norway",
    "NZ": "New Zealand", "PA": "Panama", "PE": "Peru", "PG": "Papua New Guinea",
    "PL": "Poland", "PR": "Puerto Rico", "PT": "Portugal", "SE": "Sweden",
    "SG": "Singapore", "SI": "Slovenia", "SK": "Slovakia", "TH": "Thailand",
    "TR": "Turkey", "TW": "Taiwan", "US": "United States", "VG": "Virgin Islands",
    "VN": "Vietnam", "ZA": "South Africa"
}

# Mapa de Regiones
REGION_GROUPS = {
    "Europe": {"AT", "BE", "BG", "CH", "CY", "CZ", "DE", "DK", "EE", "ES", "FI", "FO", "FR", "GB", "GI", "GR", "HU", "IE", "IS", "IT", "LI", "LT", "LU", "LV", "MC", "MT", "NL", "NO", "PL", "PT", "RO", "SE", "SI", "SK", "TR"},
    "North America": {"US", "CA", "MX", "PR"},
    "Asia-Pacific": {"AU", "CN", "HK", "ID", "JP", "KR", "MY", "NZ", "PH", "SG", "TH", "TW", "VN", "PG"},
    "Latin America": {"AR", "BR", "CL", "CO", "PE", "PA"},
    "Africa/Middle East": {"ZA", "IL", "AE", "QA", "SA", "MA", "EG"},
    "Offshore/Islands": {"BM", "BS", "KY", "VG", "JE", "GG", "IM", "AN", "MU", "MH", "LR"},
}

# Mapa País (código ISIN) -> divisa principal (para inferir cuando no se menciona en el nombre)
CURRENCY_BY_COUNTRY = {
    # Eurozona
    "AT": "EUR", "BE": "EUR", "BG": "EUR", "CY": "EUR", "DE": "EUR", "EE": "EUR",
    "ES": "EUR", "FI": "EUR", "FR": "EUR", "GR": "EUR", "IE": "EUR", "IT": "EUR",
    "LT": "EUR", "LU": "EUR", "LV": "EUR", "MT": "EUR", "NL": "EUR", "PT": "EUR",
    "SI": "EUR", "SK": "EUR",
    # Europa no euro
    "GB": "GBP", "GG": "GBP", "JE": "GBP", "IM": "GBP", "DK": "DKK", "SE": "SEK",
    "NO": "NOK", "CH": "CHF", "CZ": "CZK", "HU": "HUF", "PL": "PLN", "RO": "RON",
    # Norteamérica
    "US": "USD", "CA": "CAD", "MX": "MXN",
    # Asia-Pacífico
    "JP": "JPY", "CN": "CNY", "HK": "HKD", "SG": "SGD", "AU": "AUD", "NZ": "NZD",
    "KR": "KRW", "TW": "TWD",
    # Latinoamérica
    "BR": "BRL", "CL": "CLP", "CO": "COP", "PE": "PEN", "AR": "ARS",
    # África / Oriente Medio
    "ZA": "ZAR", "IL": "ILS", "AE": "AED", "SA": "SAR", "MA": "MAD", "EG": "EGP",
    # Offshore / Islas típicas
    "BM": "USD", "KY": "USD", "VG": "USD", "BS": "USD", "AN": "ANG", "MU": "MUR",
}

def infer_currency_from_country(code: str):
    """
    Intenta inferir la divisa a partir del código de país ISIN.
    Solo se usa como relleno cuando no se ha encontrado una divisa en el nombre.
    """
    return CURRENCY_BY_COUNTRY.get(code)

# Regex para detectar línea base: ISIN (12 caracteres) + espacio + Nombre
LINE_RE = re.compile(r"^([A-Z]{2}[A-Z0-9]{10})\s+(.+)$")

# Regex para detectar divisas en el nombre (ampliado y case-insensitive)
# Incluye códigos ISO de divisas habituales: USD, EUR, GBP, CHF, JPY, CAD, AUD, NZD, SEK, NOK, DKK, HKD, SGD,
# CNY, TWD, KRW, INR, BRL, MXN, ZAR, PLN, HUF, CZK, RUB, TRY, etc.
CURRENCY_RE = re.compile(
    r"\b(USD|EUR|GBP|CHF|JPY|CAD|AUD|NZD|SEK|NOK|DKK|HKD|SGD|CNY|RMB|TWD|KRW|INR|BRL|MXN|ZAR|PLN|HUF|CZK|RUB|TRY)\b",
    re.IGNORECASE,
)

# --- 2. FUNCIONES DE ANÁLISIS ---

def infer_region_info(isin: str, asset_type: str | None = None):
    """
    Obtiene País y Macro-Región desde el ISIN.

    - Para acciones (asset_type == "Stock" o None): se intenta mapear el código
      de país del ISIN a una macro-región usando REGION_GROUPS (como antes).
    - Para ETFs (asset_type == "ETF"): NO inferimos la región a partir del ISIN,
      porque muchos ETFs europeos (ej. ISIN irlandés) invierten en otras zonas
      del mundo. En ese caso devolvemos la región vacía ("").
    """
    if len(isin) < 2:
        return "XX", "Unknown", ""

    code = isin[:2]
    country_name = COUNTRY_MAP.get(code, f"Other ({code})")

    # Para ETFs no inferimos región desde el ISIN
    if asset_type == "ETF":
        macro = ""
    else:
        macro = ""
        for region_name, codes in REGION_GROUPS.items():
            if code in codes:
                macro = region_name
                break

    return code, country_name, macro

def clean_name(name: str) -> str:
    """Limpia ruido del PDF en el nombre del activo."""
    # Eliminar palabras clave de cabecera si se pegaron
    name = name.replace("TRADING UNIVERSE", "").replace("Stocks", "").replace("ETF", "")
    # Eliminar comillas y espacios extra
    name = name.replace('"', '').strip()
    name = re.sub(r"\s{2,}", " ", name)
    return name

def analyze_etf_provider(name: str) -> str:
    """Intenta identificar el emisor del ETF (busca el proveedor en cualquier parte del nombre)."""
    name_upper = name.upper()
    best_match = None
    for provider in ETF_PROVIDERS:
        prov_up = provider.upper()
        if prov_up in name_upper:
            # Preferimos el primer match; en caso de colisión, el primero de la lista manda
            best_match = provider
            break
    return best_match if best_match is not None else "Other/Unknown"

def analyze_distribution_policy(name: str) -> str:
    """Detecta si es Acumulación o Distribución (evitando falsos positivos como 'Acciona')."""
    name_up = name.upper()

    # Acumulación: buscamos '(ACC)' o ' ACC ' o ' ACC' al final, no dentro de palabras
    if "(ACC)" in name_up or " ACC " in name_up or name_up.endswith(" ACC"):
        return "Accumulating"

    # Distribución: patrones típicos '(DIST)', ' DIST ' o ' DIST' al final, o '(DIS)'
    if "(DIST)" in name_up or " DIST " in name_up or name_up.endswith(" DIST") or "(DIS)" in name_up:
        return "Distributing"

    return "Unknown"

# Clasificación de subtipo de ETF usando regex/keywords sobre el Name
def classify_etf_subtype(name: str) -> str:
    """
    Clasifica de forma aproximada el subtipo de ETF a partir del nombre.
    Ejemplos: 'Equity Global', 'EM Equity', 'Bond', 'Sector Tech', etc.
    """
    n = name.upper()

    # Bonos / renta fija
    if "BOND" in n or "TREASURY" in n or "FIXED INCOME" in n or "GOVERNMENT" in n:
        return "Bond"

    # Emergentes renta variable
    if "EMERGING" in n or "EMERGING MARKETS" in n or "EM EQUITY" in n:
        return "EM Equity"

    # Global / Mundo
    if "WORLD" in n or "ACWI" in n or "ALL COUNTRY" in n or "GLOBAL" in n:
        return "Equity Global"

    # País/región equity (simplificado)
    if "USA" in n or "US " in n or "U.S." in n or "S&P 500" in n:
        return "Equity USA"
    if "EUROPE" in n or "EUROSTOXX" in n or "EURO STOXX" in n:
        return "Equity Europe"
    if "JAPAN" in n or "NIKKEI" in n:
        return "Equity Japan"
    if "ASIA" in n or "PACIFIC" in n or "APAC" in n:
        return "Equity Asia-Pacific"

    # Sectores
    if "TECHNOLOGY" in n or "INFORMATION TECHNOLOGY" in n or "TECH " in n or n.endswith("TECH"):
        return "Sector Tech"
    if "HEALTH CARE" in n or "HEALTHCARE" in n:
        return "Sector Health"
    if "ENERGY" in n or "OIL & GAS" in n:
        return "Sector Energy"
    if "FINANCIAL" in n or "BANKS" in n:
        return "Sector Financial"

    # Materias primas / commodities
    if "GOLD" in n or "SILVER" in n or "COMMODITY" in n or "COMMODITIES" in n:
        return "Commodities"

    # REITs / inmobilario
    if "REIT" in n or "REAL ESTATE" in n:
        return "Real Estate"

    return "Other/Unclassified"

def extract_currency(name: str):
    """
    Extrae la divisa mencionada en el nombre del fondo.

    - Usa CURRENCY_RE (case-insensitive).
    - Si el nombre parece ser de un producto 'physical' de materias primas
      (ej. 'Physical Gold', 'Physical Silver', etc.), devolvemos None
      para tratarlo como commodity sin divisa asociada en esta clasificación.
    """
    upper_name = name.upper()

    # Regla especial: productos 'physical' sobre metales/commodities → sin divisa
    if "PHYSICAL" in upper_name and any(
        kw in upper_name for kw in ["GOLD", "SILVER", "PLATINUM", "PALLADIUM", "METAL", "COMMODITY", "COMMODITIES"]
    ):
        return None

    m = CURRENCY_RE.search(name)
    if not m:
        return None
    # Normalizamos siempre a mayúsculas el código devuelto
    return m.group(1).upper()

def is_adr_stock(name: str, asset_type: str) -> bool:
    """Detecta si es un ADR (American Depositary Receipt)."""
    if asset_type == "ETF": 
        return False
    upper_name = name.upper()
    if "ADR" in upper_name or "DEPOSITARY" in upper_name or " ADS" in upper_name:
        return True
    return False

# --- 3. PROCESO PRINCIPAL ---

def main():
    if not os.path.exists(INPUT_PDF):
        print(f"Error: No se encuentra el archivo '{INPUT_PDF}'")
        return

    data_rows = []
    current_section = "Stock" # Por defecto empezamos en Acciones

    print(f"🚀 Iniciando extracción inteligente de {INPUT_PDF}...")
    
    with pdfplumber.open(INPUT_PDF) as pdf:
        total_pages = len(pdf.pages)
        
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if not text:
                continue
            
            # --- REGLA MAESTRA: A partir de la página 274 son ETFs ---
            if i >= 274:
                current_section = "ETF"

            # Logs de progreso
            if i % 20 == 0 or i == total_pages:
                print(f"   -> Procesando página {i}/{total_pages} (Modo: {current_section})")

            # Procesar líneas
            lines = text.splitlines()
            for raw_line in lines:
                line = raw_line.strip()
                
                # Ignorar líneas basura
                if "TRADING UNIVERSE" in line or line in ["ETF", "Stocks"]:
                    continue
                if line.isdigit(): # Número de página suelto
                    continue
                if line.startswith("[source"): # Artefactos de tu ejemplo de texto
                    continue

                # Intentar machear ISIN + Nombre
                match = LINE_RE.match(line)
                if match:
                    isin, raw_name = match.groups()
                    name = clean_name(raw_name)
                    
                    # --- DETERMINAR TIPO ---
                    asset_type = current_section
                    name_up = name.upper()

                    # Política de distribución (solo para columna, NO para decidir ETF/Stock)
                    dist_policy = analyze_distribution_policy(name)

                    # Decisión de tipo:
                    # - Si estamos en la sección de ETFs (páginas >= 274) → ETF seguro
                    # - O si el nombre contiene 'UCITS' o ' ETF' de forma explícita
                    if current_section == "ETF" or "UCITS" in name_up or " ETF" in name_up or name_up.endswith("ETF"):
                        asset_type = "ETF"

                    # --- ANÁLISIS DE DATOS ---
                    code, country, region = infer_region_info(isin, asset_type)
                    
                    provider = None
                    is_adr = False

                    # Divisa:
                    # - Para ETFs: primero intentamos detectar por nombre y, si no, inferimos por país.
                    # - Para acciones (stocks): asumimos directamente la divisa principal del país del ISIN.
                    if asset_type == "ETF":
                        currency = extract_currency(name)
                        if currency is None:
                            currency = infer_currency_from_country(code)
                        provider = analyze_etf_provider(name)
                    else:
                        currency = infer_currency_from_country(code)
                        is_adr = is_adr_stock(name, asset_type)

                    # Subtipo de ETF (solo si es ETF)
                    etf_subtype = classify_etf_subtype(name) if asset_type == "ETF" else None

                    # Clave de búsqueda simplificada (minúsculas, sin espacios dobles)
                    search_key = re.sub(r"\s+", " ", name).strip().lower()
                    # Guardar registro
                    data_rows.append({
                        "ISIN": isin,
                        "Name": name,
                        "Type": asset_type,
                        "Region": region,
                        "Country": country,
                        "Country_Code": code,
                        "ETF_Provider": provider,
                        "ETF_Subtype": etf_subtype,
                        "Distribution": dist_policy,
                        "Currency_Name": currency,
                        "Is_ADR": is_adr,
                        "Page": i,
                        "Search_Key": search_key,
                    })

    # --- 4. EXPORTACIÓN Y FORMATO ---
    
    print("📊 Generando tablas y eliminando duplicados...")
    df = pd.DataFrame(data_rows)
    
    # Deduplicación: Si un ISIN sale 2 veces, nos quedamos el primero
    df = df.drop_duplicates(subset=["ISIN"], keep="first")
    
    # Estadísticas rápidas
    total = len(df)
    etfs = len(df[df['Type']=='ETF'])
    stocks = len(df[df['Type']=='Stock'])
    
    print("\n" + "="*40)
    print(f"RESUMEN DE EXTRACCIÓN")
    print("="*40)
    print(f"Total Activos: {total}")
    print(f"Acciones     : {stocks}")
    print(f"ETFs         : {etfs}")
    print("-" * 40)
    
    # Guardar Excel
    print(f"💾 Guardando Excel: {OUTPUT_EXCEL}")
    try:
        df.to_excel(OUTPUT_EXCEL, index=False)
        print("   -> Éxito.")
    except Exception as e:
        print(f"   -> Error guardando Excel (quizás falta openpyxl): {e}")

    # Guardar CSV (Backup)
    print(f"💾 Guardando CSV: {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    
    # Guardar JSON
    print(f"💾 Guardando JSON: {OUTPUT_JSON}")
    df.to_json(OUTPUT_JSON, orient="records", indent=2, force_ascii=False)

    print("\n¡Proceso completado! Revisa el archivo Excel generado.")

if __name__ == "__main__":
    main()