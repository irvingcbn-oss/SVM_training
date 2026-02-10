import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.stats import mode
from IPython.display import display

# -----------------------------------
def filtrar_senal(datos, fs, lowcut=50, highcut=499.9, orden=4):
    nyq = 0.5 * fs
    low = lowcut / nyq                                         #frecuencia nyq = (frecuencia de muestreo)/2
    high = highcut / nyq                                       #se normalizan las frecuencias con nyq
    b, a = butter(orden, [low, high], btype='band')            #devuelve coeficientes del filtro
    return filtfilt(b, a, datos)                               #devuelve señal filtrada

# ---------------------------
def dividir_ventanas(senal, fs, duracion_ventana, paso_ventana):

    longitud_ventana = int(fs * duracion_ventana)
    paso = int(fs * paso_ventana)

    ventanas = [
        senal[i:i + longitud_ventana]
        for i in range(0, len(senal) - longitud_ventana + 1, paso)
    ]
    return ventanas

# ---------------------------
def rectificar(senal):
    return np.abs(senal)

# -----------------------------------------
def calcular_caracteristicas(ventanas_senal, ventanas_etiqueta, umbral=0.05):
    caracteristicas = []
    for i, (v, etiqueta) in enumerate(zip(ventanas_senal, ventanas_etiqueta)):
        rms = np.sqrt(np.mean(v**2))
        wl = np.sum(np.abs(np.diff(v)))
        willison = np.sum(np.abs(np.diff(v)) > umbral)
        etiqueta_ventana = mode(etiqueta, keepdims=True)[0][0]  # etiqueta más frecuente
        caracteristicas.append({
            'Ventana': i,
            'RMS': rms,
            'WaveformLength': wl,
            'WillisonAmplitude': willison,
            'Etiqueta': etiqueta_ventana
        })
    return pd.DataFrame(caracteristicas)

# -------------------------------
def analizar_senal(X, etiquetas, fs, duracion_ventana, paso_ventana):
    senal_filtrada = filtrar_senal(X, fs)
    senal_rectificada = rectificar(senal_filtrada)

    ventanas_senal = dividir_ventanas(senal_rectificada, fs, duracion_ventana, paso_ventana)
    ventanas_etiqueta = dividir_ventanas(etiquetas, fs, duracion_ventana, paso_ventana)

    df_caracteristicas = calcular_caracteristicas(ventanas_senal, ventanas_etiqueta)
    return df_caracteristicas

# -------------------------------
if _name_ == "_main_":
    # Cargar CSV
    df = pd.read_csv("emg_base_datos_M.csv")

    # Ajusta las columnas según tu archivo
    X = df.iloc[:, 1].values  # Columna de señal (index 1)
    y = df.iloc[:, 2].values  # Columna de etiquetas (index 2)

    fs = 1000  # Hz
    duracion_ventana = 1.0 # segundos
    paso_ventana = 0.5 # 1 ms de paso para solapamiento máximo (un vector de características por muestra)

    resultados = analizar_senal(X, y, fs, duracion_ventana, paso_ventana)

    # Guardar CSV con etiquetas
    resultados.to_csv("base_datos_joel.csv", index=False)
    print("Características con etiquetas guardadas en 'base_datos_joel.csv'")
    try:
        display(resultados)
    except ImportError:
        print(resultados.head())