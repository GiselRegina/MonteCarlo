import streamlit as st
import numpy as np
import pandas as pd
import math
import random

def f_opcion_a(x):
    # f(x) = 1 / (e^x + e^{-x}) = 1/(2 cosh x)
    return 1.0 / (2.0 * np.cosh(x))

def f_opcion_b(x):
    # f(x) = 2 / (e^x + e^{-x}) = 1/cosh(x)
    return 1.0 / np.cosh(x)

class MonteCarloIntegrator:
    def __init__(self, f, a, b, n, seed=None):
        self.f = f
        self.a = a
        self.b = b
        self.n = n
        if seed is not None:
            random.seed(seed)

    def run(self):
        xs = [self.a + (self.b - self.a) * random.random() for _ in range(self.n)]
        fs = [self.f(x) for x in xs]
        factor = (self.b - self.a) / self.n
        areas = [factor * fi for fi in fs]
        estimacion = sum(areas)
        return xs, fs, areas, estimacion

st.set_page_config(page_title="Simulador Montecarlo", layout="wide")

st.title("Simulador del Método de Montecarlo")
st.caption("Gisel Regina Benítez Calvillo — A00228137")

st.sidebar.header("Parámetros")

# Función a integrar
opcion = st.sidebar.selectbox(
    "Seleccione la función a integrar",
    ("Opción 1", "Opción 2")
)

# Mostrar función elegida en formato matemático real
if opcion == "Opción 1":
    st.latex(r"f(x)=\frac{1}{e^{x}+e^{-x}}")
    funcion = f_opcion_a
else:
    st.latex(r"f(x)=\frac{2}{e^{x}+e^{-x}}")
    funcion = f_opcion_b

funcion = f_opcion_a if opcion.startswith("1") else f_opcion_b

# Intervalos
a_input = st.sidebar.text_input("Límite inferior (a)", value="-6")
b_input = st.sidebar.text_input("Límite superior (b)", value="6")

# Manejo de infinito
def parse_intervalo(valor):
    valor = valor.strip().lower()
    if valor in ["inf", "+inf"]:
        return 6
    if valor == "-inf":
        return -6
    return float(valor)

a = parse_intervalo(a_input)
b = parse_intervalo(b_input)

# Tamaño de muestra
n = st.sidebar.number_input("Tamaño de muestra (n)", min_value=1, value=1000)

# Semilla opcional
seed_input = st.sidebar.text_input("Semilla (opcional)")
seed = int(seed_input) if seed_input.strip().isdigit() else None

# Botón
if st.sidebar.button("▶ Ejecutar simulación"):
    mc = MonteCarloIntegrator(funcion, a, b, n, seed)
    xs, fs, areas, estimacion = mc.run()

    st.success("Simulación ejecutada correctamente")

    # Tabs para organizar contenido
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Tabla de resultados", "Gráfica f(xᵢ)", "Gráfica de áreas", "Métricas"]
    )

    # TABLA
    with tab1:
        st.subheader("Resultados punto por punto")
        df = pd.DataFrame({
            "xᵢ (valores aleatorios)": xs,
            "f(xᵢ)": fs,
            "(b-a)/n * f(xᵢ)": areas
        })
        st.dataframe(df, use_container_width=True)

    # GRÁFICA DE ALTURAS
    with tab2:
        st.subheader("Alturas f(xᵢ)")
        st.line_chart(fs)

    # GRÁFICA DE ÁREAS
    with tab3:
        st.subheader("Áreas calculadas")
        st.line_chart(areas)

    # MÉTRICAS
    with tab4:
        st.subheader("Métricas de la simulación")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Estimación de la integral", f"{estimacion:.6f}")
            st.metric("Media de f(xᵢ)", f"{np.mean(fs):.6f}")

        with col2:
            st.metric("Varianza", f"{np.var(fs):.6f}")
            st.metric("Desviación estándar", f"{np.std(fs):.6f}")

else:
    st.info("Configura los parámetros en el menú lateral y presiona **Ejecutar simulación**.")
