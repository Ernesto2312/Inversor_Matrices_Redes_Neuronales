"""
Inversor de Matrices 4x4 mediante Red Neuronal + Refinamiento Algebraico (Newton-Schulz)
==========================================================================================

IDEA GENERAL DEL PROYECTO:
En vez de calcular la inversa con Gauss-Jordan, usamos una red neuronal (MLP) como
aproximador inicial rapido, y luego refinamos ese resultado con un metodo algebraico
iterativo (Newton-Schulz) hasta alcanzar precision de maquina.

Hay DOS caminos segun que tan dificil es la matriz:

  - Matrices BIEN CONDICIONADAS (cond < 10): la red, entrenada especificamente en este
    dominio, da un punto de partida confiable. Newton-Schulz lo refina muy rapido.

  - Matrices MAS DIFICILES/CASI SINGULARES (10 <= cond < 50): la red ya no es confiable
    aqui (no fue entrenada para esto), asi que usamos una inicializacion ALGEBRAICA
    CLASICA que matematicamente GARANTIZA que Newton-Schulz converja. Es decir: en estos
    casos mas complicados, el algebra resuelve el problema directamente, sin depender de
    la red, pero llegando IGUAL a precision de maquina.

En ambos casos, el resultado final es casi exacto (error tipico ~1e-7).
"""

import torch
import torch.nn as nn
import copy

torch.manual_seed(42)  # reproducibilidad


# ============================================================
# 1. GENERACION DE DATOS DE ENTRENAMIENTO
# ============================================================
def generar_dataset(n_muestras, n=4, max_cond=10):
    """
    Genera matrices aleatorias invertibles, filtradas por numero de condicion.
    No se calcula la inversa real aqui porque entrenamos con perdida ALGEBRAICA
    (no necesitamos saber la inversa verdadera para entrenar - ver entrenar_red()).
    """
    matrices = []
    while len(matrices) < n_muestras:
        A = torch.randn(n, n)
        if torch.linalg.cond(A) < max_cond:
            matrices.append(A)
    return torch.stack(matrices)  # forma: (n_muestras, n, n)


# ============================================================
# 2. ARQUITECTURA DE LA RED
# ============================================================
class RedInversora(nn.Module):
    """MLP simple: recibe una matriz nxn aplanada, devuelve su aproximacion de inversa aplanada."""
    def __init__(self, n=4):
        super().__init__()
        self.n = n
        entrada_salida = n * n
        self.red = nn.Sequential(
            nn.Linear(entrada_salida, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, entrada_salida)
        )

    def forward(self, x):
        return self.red(x)

    def predecir_matriz(self, A):
        """Conveniencia: recibe una matriz (n,n) y devuelve la prediccion como matriz (n,n)."""
        x = A.flatten().unsqueeze(0)
        return self.forward(x).reshape(self.n, self.n)


# ============================================================
# 3. ENTRENAMIENTO CON PERDIDA ALGEBRAICA (RESIDUAL)
# ============================================================
def entrenar_red(modelo, A_train, A_test, epocas=1500, lr=0.001, paciencia=150):
    """
    En vez de comparar contra la inversa real (MSE directo), entrenamos minimizando
    el residuo algebraico ||A @ X - I||^2. Esto evita el problema de que los valores
    de la inversa real pueden ser enormes/inestables cerca de matrices mal condicionadas,
    y ademas es EXACTAMENTE la cantidad que Newton-Schulz necesita que sea pequena
    para poder converger.

    Usa train/test split + early stopping: nos quedamos con los pesos del modelo en el
    punto donde mejor generalizo a datos NUNCA vistos (evita overfitting).
    """
    n = modelo.n
    I = torch.eye(n).unsqueeze(0)  # se expande por broadcasting a (batch, n, n)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=lr)

    def perdida_residual(A_batch):
        pred = modelo(A_batch.reshape(-1, n * n)).reshape(-1, n, n)
        residuo = A_batch @ pred - I
        return (residuo ** 2).mean()

    mejor_test, mejores_pesos, sin_mejora = float('inf'), None, 0

    for epoca in range(epocas):
        modelo.train()
        optimizador.zero_grad()
        perdida_train = perdida_residual(A_train)
        perdida_train.backward()
        optimizador.step()

        modelo.eval()
        with torch.no_grad():
            perdida_test = perdida_residual(A_test).item()

        if perdida_test < mejor_test:
            mejor_test = perdida_test
            mejores_pesos = copy.deepcopy(modelo.state_dict())
            sin_mejora = 0
        else:
            sin_mejora += 1

        if epoca % 150 == 0:
            print(f"  Época {epoca:4d} | train: {perdida_train.item():.6f} | test: {perdida_test:.6f}")

        if sin_mejora >= paciencia:
            print(f"  Early stopping en época {epoca} (mejor test loss: {mejor_test:.6f})")
            break

    modelo.load_state_dict(mejores_pesos)
    return mejor_test


# ============================================================
# 4. INICIALIZACION ALGEBRAICA SEGURA (sin red neuronal)
# ============================================================
def inicializacion_segura(A):
    """
    Formula clasica de algebra lineal numerica que GARANTIZA que Newton-Schulz converja,
    sin importar que tan mal condicionada este A (mientras sea invertible):

        X0 = A^T / (||A||_1 * ||A||_inf)

    Esta es la inicializacion que usamos para matrices DIFICILES (cond entre 10 y 50),
    donde la red no fue entrenada y por lo tanto no es confiable.
    """
    norma_1 = torch.linalg.matrix_norm(A, ord=1)
    norma_inf = torch.linalg.matrix_norm(A, ord=float('inf'))
    return A.T / (norma_1 * norma_inf)


# ============================================================
# 5. REFINAMIENTO ITERATIVO: NEWTON-SCHULZ
# ============================================================
def newton_schulz(A, X0, max_iter=25, tol=1e-10):
    """
    Itera X_{k+1} = X_k(2I - A@X_k). Si el punto de partida X0 cumple ||I-A@X0|| < 1,
    esto converge CUADRATICAMENTE (el numero de cifras correctas se duplica en cada paso).
    Solo usa multiplicaciones de matrices, ningun calculo de determinante ni divisiones
    elemento a elemento.
    """
    n = A.shape[0]
    I = torch.eye(n)
    X = X0.clone()
    for k in range(max_iter):
        residuo = I - A @ X
        if torch.norm(residuo).item() < tol:
            break
        X = X @ (2 * I - A @ X)
        if torch.isnan(X).any() or torch.norm(X).item() > 1e8:
            return X, k + 1, False  # diverge
    return X, k + 1, True


# ============================================================
# 6. FUNCION PRINCIPAL: INVERTIR UNA MATRIZ
# ============================================================
def invertir(modelo, A, umbral_confianza=1.0):
    """
    Decide que camino tomar:
      - Si la prediccion de la red ya da un residuo pequeno (< umbral_confianza),
        la usamos como punto de partida -> camino "red neuronal"
      - Si no, usamos la inicializacion algebraica clasica, que SIEMPRE funciona
        -> camino "algebra clasica"
    En ambos casos, se refina con Newton-Schulz hasta precision de maquina.
    """
    n = A.shape[0]
    I = torch.eye(n)

    with torch.no_grad():
        X0_red = modelo.predecir_matriz(A)
    residuo_red = torch.linalg.matrix_norm(I - A @ X0_red, ord=2).item()

    if residuo_red < umbral_confianza:
        X0 = X0_red
        origen = "red neuronal"
    else:
        X0 = inicializacion_segura(A)
        origen = "algebra clasica"

    X_final, iteraciones, convergio = newton_schulz(A, X0)
    return X_final, origen, iteraciones, convergio, residuo_red


# ============================================================
# 7. EVALUACION: PROBAR EN AMBOS REGIMENES (facil y dificil)
# ============================================================
def evaluar_completo(modelo, n_pruebas=10, n=4):
    print("\n" + "=" * 70)
    print("EVALUACION: matrices BIEN CONDICIONADAS (cond < 10) -> espera usar la RED")
    print("=" * 70)
    _probar_lote(modelo, n_pruebas, n, cond_min=0, cond_max=10)

    print("\n" + "=" * 70)
    print("EVALUACION: matrices DIFICILES (10 <= cond < 50) -> espera usar ALGEBRA")
    print("=" * 70)
    _probar_lote(modelo, n_pruebas, n, cond_min=10, cond_max=50)


def _probar_lote(modelo, n_pruebas, n, cond_min, cond_max):
    usos_red, usos_algebra, exitos = 0, 0, 0
    for i in range(n_pruebas):
        A = torch.randn(n, n)
        c = torch.linalg.cond(A).item()
        while not (cond_min <= c < cond_max):
            A = torch.randn(n, n)
            c = torch.linalg.cond(A).item()

        A_inv_real = torch.inverse(A)
        X_final, origen, iters, convergio, residuo_red = invertir(modelo, A)
        error = torch.norm(A_inv_real - X_final).item()

        if origen == "red neuronal":
            usos_red += 1
        else:
            usos_algebra += 1
        if convergio and error < 1e-3:
            exitos += 1

        print(f"  Prueba {i:2d}: cond={c:6.2f} | origen={origen:14s} | "
              f"iteraciones={iters:2d} | error final={error:.2e}")

    print(f"\n  Resumen: {exitos}/{n_pruebas} exitosos | "
          f"usos red={usos_red} | usos algebra={usos_algebra}")


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================
if __name__ == "__main__":
    n = 4

    print("Generando dataset de entrenamiento (matrices bien condicionadas, cond<10)...")
    n_total = 6000
    A_data = generar_dataset(n_total, n=n, max_cond=10)
    n_train = int(n_total * 0.8)
    A_train, A_test = A_data[:n_train], A_data[n_train:]
    print(f"Train: {A_train.shape[0]} matrices | Test: {A_test.shape[0]} matrices\n")

    print("Entrenando la red con perdida algebraica (residual ||A@X - I||^2)...")
    modelo = RedInversora(n=n)
    entrenar_red(modelo, A_train, A_test)

    evaluar_completo(modelo, n_pruebas=10, n=n)