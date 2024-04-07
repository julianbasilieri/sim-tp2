import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon
import scipy.stats as stats
from tabulate import tabulate

# Definicion de las variables globales
n = None
minimo = None
maximo = None
rango = None
cant_intervalos = None
amplitud = None
media = None
varianza = None


def obtener_opcion(validas):
    opcion = input("Ingrese su opción: ")
    while opcion not in validas:
        print("Opción no válida. Inténtelo de nuevo.")
        opcion = input("Ingrese su opción: ")
    return int(opcion)


def generar_uniforme(a, b, tam_muestra):
    # Generar variables aleatorias con distribucion uniforme utilzando la formula 𝑋 = 𝐴 + 𝑅𝑁𝐷(𝐵 − 𝐴)
    list_rand = []
    for i in range(tam_muestra):
        ran = random.uniform(0, 1)
        list_rand.append(round(a + ran * (b - a), 4))
    return list_rand
    # return [round(random.uniform(a, b), 4) for _ in range(tam_muestra)]


def generar_exponencial(lam, tam_muestra):
    # Generar variables aleatorias con distribucion exponencial utilzando la formula X = -1/lam * ln(1 - RND)
    list_rand = []
    for i in range(tam_muestra):
        ran = random.uniform(0, 1)
        list_rand.append(round(-(1 / lam) * (math.log(1 - ran)), 4))
    return list_rand
    # return [round(random.expovariate(1 / lam), 4) for _ in range(tam_muestra)]


def generar_normal(med, desviacion, tam_muestra):
    # Generar variables aleatorias con distribucion normal utilzando el metodo de Box-Muller
    list_rand = []
    for i in range(tam_muestra):
        ran1 = random.uniform(0, 1)
        ran2 = random.uniform(0, 1)
        list_rand.append(round(((math.sqrt(-2 * math.log(ran1))) * math.cos(2 * math.pi * ran2)) * desviacion + med, 4))
    return list_rand
    # return [round(random.normalvariate(media, desviacion), 4) for _ in range(tam_muestra)]


def obtener_a_b():
    a = float(input("Ingrese el valor de a: "))
    b = float(input("Ingrese el valor de b: "))

    # Validar que a sea menor que b
    while a >= b:
        print("a debe ser menor a b")
        a = float(input("Ingrese el valor de a: "))
        b = float(input("Ingrese el valor de b: "))
    return a, b


def obtener_valor_entre(nombre, mini, maxi):
    valor = int(input("Ingrese el valor de " + nombre + "(entre " + str(mini) + " y " + str(maxi) + "): "))

    # Validar que el valor este entre un minimio y un maximo
    while valor < mini or valor > maxi:
        print("Tamaño de muestra inválido. Debe ser un número entre", str(mini), " y ", str(maxi))
        valor = int(input("Ingrese el valor de " + nombre + "(entre " + str(mini) + " y " + str(maxi) + "): "))
    return valor


def obtener_valor_mayor(nombre, mini):
    valor = float(input("Ingrese el valor de " + nombre + "(mayor a " + str(mini) + "): "))

    # Validar que el valor sea mayor a un minimo
    while valor <= mini:
        print("Tamaño de muestra inválido. Debe ser un número mayor a", str(mini))
        valor = float(input("Ingrese el valor de " + nombre + "(mayor a " + str(mini) + "): "))
    return valor


def calcular_limites():
    # Calcular los limites inferiores (minimo + amplitud) de todos los intervalos
    limites_inferiores = [minimo + i * amplitud for i in range(cant_intervalos)]

    # Calcular los limites superiores (limite_inferior + amplitud) de todos los intervalos
    limites_superiores = [limite_inf + amplitud for limite_inf in limites_inferiores]

    # Modifico el ultimo limite superior
    print("Maimo", maximo)
    limites_superiores[-1] = maximo
    return limites_inferiores, limites_superiores


def calcular_frec_obs(datos):
    frecuencias = [0] * cant_intervalos

    for dato in datos:
        intervalo = int((dato - minimo) / amplitud)
        if intervalo == cant_intervalos:
            intervalo -= 1  # Asegurarse de que los valores que caen en el límite superior se incluyan en el último
            # intervalo
        frecuencias[intervalo] += 1

    return frecuencias


def calcular_frec_esp_uni():
    # La frecuencia esperada es la misma para todos los intervalos
    return [n / cant_intervalos] * cant_intervalos


def calcular_frec_esp_exp(lim_inf, lim_sup, lam):
    # La frecuencia esperada es mayor en los primeros intervalos y disminuye en los siguientes intervalos
    frecuencia_esp = []

    for i in range(cant_intervalos):
        cal_sup = expon.cdf(lim_sup[i], scale=1 / lam)  # cal_sup = 1 - math.exp(-lam * lim_sup[i])
        cal_inf = expon.cdf(lim_inf[i], scale=1 / lam)  # cal_inf = 1 - math.exp(-lam * lim_inf[i])

        frecuencia_esp.append((cal_sup - cal_inf) * n)

    return frecuencia_esp


def calcular_frec_esp_nor(lim_inf, lim_sup, med, desviacion):
    # La frecuencia esperada es mayor en la media y disminuye para intervalos anteriores y posteriores a la media
    frecuencia_esp = []

    for i in range(cant_intervalos):
        cal_sup = stats.norm.cdf(lim_sup[i], loc=med, scale=desviacion)
        cal_inf = stats.norm.cdf(lim_inf[i], loc=med, scale=desviacion)

        frecuencia_esp.append((cal_sup - cal_inf) * n)
    return frecuencia_esp


def calcular_chi_cuadrado(observado, esperado):
    # Chi cuadrado va a ser mayor mientras haya mas diferencia entre la frecuencia observada y la frecuencia esperada
    chi = []
    for i in range(cant_intervalos):
        chi.append(round(((observado[i] - esperado[i]) ** 2) / esperado[i], 4))
    return chi


def acomodar_frec(matriz):
    # Cantidad de filas de la matriz
    num_filas = cant_intervalos

    i = 0

    # Iteracion sobre las filas de la matriz
    while i < num_filas:

        # Verificar si la frecuencia esperada es menor a 5
        if matriz['Frecuencia Esperada'][i] < 5:

            # Verificar si es la ultima fila de la tabla
            if i == num_filas - 1:

                # Si el tamaño de la muestra es muy pequeño puede que la frecuencia esperada nunca sea mayor a 5
                if i == 0:
                    break

                # Si es un array, obtener el mínimo y máximo, sino tomar el valor directamente
                if isinstance(matriz['Intervalo'][i - 1], list):
                    int_min = min(matriz['Intervalo'][i - 1])
                else:
                    int_min = matriz['Intervalo'][i - 1]

                if isinstance(matriz['Intervalo'][i], list):
                    int_max = max(matriz['Intervalo'][i])
                else:
                    int_max = matriz['Intervalo'][i]

                # Sumar todos los datos de la tabla con el intervalo anterior en caso de que este en el ultimo intervalo
                # y la frecuencia esperada sea menor a 5

                matriz['Intervalo'][i - 1] = [int_min, int_max]
                matriz['Limite Superior'][i - 1] = matriz['Limite Superior'][i]
                matriz['Frecuencia Observada'][i - 1] += matriz['Frecuencia Observada'][i]
                matriz['Frecuencia Esperada'][i - 1] += matriz['Frecuencia Esperada'][i]
                matriz['Chi Cuadrado'][i - 1] += matriz['Chi Cuadrado'][i]

                # Eliminar la fila actual(ultima fila) ya que la sume con la anterior
                del matriz['Intervalo'][i]
                del matriz['Limite Inferior'][i]
                del matriz['Limite Superior'][i]
                del matriz['Frecuencia Observada'][i]
                del matriz['Frecuencia Esperada'][i]
                del matriz['Chi Cuadrado'][i]

                # Salir del while debido a que no tengo mas intervalos
                break

            # La frecuencia esperada es menor a 5 pero no estoy en el ultimo intervalo
            else:
                # Inicializo variable j que es la que me va a determinar cuantos intervalos debo unir
                j = i + 1

                # Busco algun rango de intervalos donde la suma de las frecuencias esperadas sea mayor a 5
                while sum(matriz['Frecuencia Esperada'][i:j]) < 5:
                    j += 1

                    # En caso de llegar al final de la tabla y la suma de las frecuencias esperadas es menor que 5
                    # finaliza el while
                    if j == (cant_intervalos - 1):
                        break

                # Si es un array, obtener el mínimo y máximo, sino tomar el valor directamente
                if isinstance(matriz['Intervalo'][i], list):
                    int_min = min(matriz['Intervalo'][i])
                else:
                    int_min = matriz['Intervalo'][i]

                if isinstance(matriz['Intervalo'][i:j][-1], list):
                    int_max = max(matriz['Intervalo'][i:j][-1])
                else:
                    int_max = matriz['Intervalo'][i:j][-1]

                # Sumar todos los datos de la tabla que esten entre i y j
                # El rango del intervalo va a estar definido por el intervalo menor y el intervalo mayor
                matriz['Intervalo'][i] = [int_min, int_max]
                matriz['Limite Inferior'][i] = min(matriz['Limite Inferior'][i:j])
                matriz['Limite Superior'][i] = max(matriz['Limite Superior'][i:j])
                matriz['Frecuencia Observada'][i] = sum(matriz['Frecuencia Observada'][i:j])
                matriz['Frecuencia Esperada'][i] = sum(matriz['Frecuencia Esperada'][i:j])
                matriz['Chi Cuadrado'][i] = sum(matriz['Chi Cuadrado'][i:j])

                # Eliminar las filas que esten entre i y j
                del matriz['Intervalo'][i + 1:j]
                del matriz['Limite Inferior'][i + 1:j]
                del matriz['Limite Superior'][i + 1:j]
                del matriz['Frecuencia Observada'][i + 1:j]
                del matriz['Frecuencia Esperada'][i + 1:j]
                del matriz['Chi Cuadrado'][i + 1:j]

                # Actualizar el número de filas
                num_filas = len(matriz['Frecuencia Esperada'])

                # Volver a verificar si la fila actual tiene una frecuencia esperada mayor a 5
                i -= 1

        i += 1

    return matriz


def histograma(datos, matriz):
    plt.hist(datos, bins=cant_intervalos, edgecolor='black')
    plt.title('Histograma de Frecuencias')
    plt.xlabel('Valor')
    plt.ylabel('Frecuencia')

    for i in range(len(matriz['Limite Inferior'])):
        lim_inf = matriz['Limite Inferior'][i]
        lim_sup = matriz['Limite Superior'][i]

        # Agregar líneas verticales para mostrar los límites de los intervalos
        plt.axvline(x=lim_inf, color='green', linestyle='--', linewidth=1)
        plt.axvline(x=lim_sup, color='green', linestyle='--', linewidth=1)

        # Agregar el nombre del intervalo
        plt.text((lim_inf + lim_sup) / 2, 0, f'Intervalo {matriz["Intervalo"][i]}', ha='center', va='top', rotation=90,
                 color='green')

    plt.show()


def main():
    global n, minimo, maximo, rango, cant_intervalos, amplitud, media, varianza

    # Usuario ingresa la distribucion que desea
    print("TP2")
    print("Seleccione la distribución:")
    print("1. Uniforme")
    print("2. Exponencial")
    print("3. Normal")
    opcion_distribucion = obtener_opcion(['1', '2', '3'])

    # Usuario ingresa datos de la distribucion seleccionada
    if opcion_distribucion == 1:
        a, b = obtener_a_b()
        n = obtener_valor_entre("tamaño de muestra", 1, 1000000)

        # Generar datos con una distribucion uniforme
        datos = generar_uniforme(a, b, n)
    elif opcion_distribucion == 2:
        lam = obtener_valor_mayor("lambda", 0)
        n = obtener_valor_entre("tamaño de muestra", 1, 1000000)

        # Generar datos con una distribucion exponencial
        datos = generar_exponencial(lam, n)
    elif opcion_distribucion == 3:
        media = float(input("Ingrese el valor de la media: "))
        desviacion = obtener_valor_mayor("desviacion estandar", 0)
        n = obtener_valor_entre("tamaño de muestra", 1, 1000000)

        # Generar datos con una distribucion normal
        datos = generar_normal(media, desviacion, n)

    # Mostrar datos generados
    print("\nSerie de números generada:")
    print(datos)

    # Usuario ingresa numero de intervalos, pueden ser 10, 15, 20, 25
    print("\nSeleccione el número de intervalos para el histograma:")
    print("10 intervalos")
    print("15 intervalos")
    print("20 intervalos")
    print("25 intervalos")
    cant_intervalos = obtener_opcion(['10', '15', '20', '25'])

    # Calcular datos de los numeros aleatorios generados
    minimo = min(datos)
    maximo = max(datos)
    rango = round(maximo - minimo, 4)
    amplitud = round(rango / cant_intervalos, 4)
    media = round(np.mean(datos), 4)
    varianza = round(np.var(datos), 4)

    # Mostrar datos de los numeros aleatorios generados
    print("\nTamano de muestra:", n)
    print("Minimo:", minimo)
    print("Maximo:", maximo)
    print("Rango:", rango)
    print("Intervalos: ", cant_intervalos)
    print("Amplitud:", amplitud)
    print("Media:", media)
    print("Varianza:", varianza)

    # Calcular los limites inferiores y limites superiores de los intervalos
    lim_inf, lim_sup = calcular_limites()

    # Calcular la frecuencia observada
    frec_obs = calcular_frec_obs(datos)

    # Calcular la frecuencia esperada dependiendo de la distribucion ingresada por el usuario
    if opcion_distribucion == 1:
        frec_esp = calcular_frec_esp_uni()
    elif opcion_distribucion == 2:
        frec_esp = calcular_frec_esp_exp(lim_inf, lim_sup, lam)
    elif opcion_distribucion == 3:
        frec_esp = calcular_frec_esp_nor(lim_inf, lim_sup, media, desviacion)

    chi_cuadrado = calcular_chi_cuadrado(frec_obs, frec_esp)

    # Generar un matriz con todos los datos
    matriz_chi_cuadrado = {'Intervalo': list(range(1, cant_intervalos + 1)),
                           'Limite Inferior': lim_inf,
                           'Limite Superior': lim_sup,
                           'Frecuencia Observada': frec_obs,
                           'Frecuencia Esperada': frec_esp,
                           'Chi Cuadrado': chi_cuadrado}

    # Debido a que las frecuencias esperadas no pueden ser menores a 5 debemos acomodar la matriz
    matriz_chi_cuadrado = acomodar_frec(matriz_chi_cuadrado)

    # Mostrar la tabla con tabulate
    print("\nTabla de Chi Cuadrado:")
    print(tabulate(matriz_chi_cuadrado, headers="keys", tablefmt="double_grid", numalign="center"))

    # Mostrar las sumatorias de la frecuencia observada, frecuencia esperada y chi cuadrado
    print("\nSumatoria Frecuencia observada:", sum(frec_obs))
    print("\nSumatoria Frecuencia esperada:", round(sum(frec_esp), 4))
    print("\nChi Cuadrado calculado:", round(sum(chi_cuadrado), 4))

    # Mostrar el histograma
    histograma(datos, matriz_chi_cuadrado)


if __name__ == "__main__":
    main()
