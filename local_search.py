import pandas as pd
import random
import plotly.express as px

# Número de filas y columnas del terreno y su delimitación
columnas = 32
filas = 22
terreno = [[' ' for _ in range(filas)] for _ in range(columnas)]

# Lista de los niveles topográficos (del 1 al 6)
niveles_topograficos = [str(i) for i in range(1, 7)]

# Representaciones de los elementos
marcador_agua = '0'
marcador_obstaculo = '#'
marcador_rover = 'R'

# Diccionario con los colores de cada elemento
colors_topografia = {
    '1': 'rgb(231, 156, 163)',
    '2': 'rgb(206, 106, 113)',
    '3': 'rgb(156, 46, 53, 1)',
    '4': 'rgb(106, 26, 33)',
    '5': 'rgb(56, 6, 13)',
    '6': 'rgb(40, 4, 9)',
    '#': 'rgb(141, 145, 141)',
    '0': 'rgb(0, 183, 228)',
    'R': 'rgb(58, 41, 30)',
}

# Diccionario con los símbolos de cada elemento
symbols_topografia = {
    '1': 'hexagon',
    '2': 'hexagon',
    '3': 'hexagon',
    '4': 'hexagon',
    '5': 'hexagon',
    '6': 'hexagon',
    '#': 'x',
    '0': 'circle',
    'R': 'star',
}

# Diccionario con las etiquetas de cada elemento
labels = {
    '1': '1',
    '2': '2',
    '3': '3',
    '4': '4',
    '5': '5',
    '6': '6',
    '#': 'Obstacle',
    '0': 'Water',
    'R': 'Rover',
}

# Creación de los puntos de los niveles topográficos
for col in range(columnas):
    for fila in range(filas):
        terrain_type = random.choice(niveles_topograficos)
        terreno[col][fila] = terrain_type

# Modificación de los niveles topográficos para los puntos colindantes
for col in range(columnas):
    for fila in range(filas):
        current_topography = int(terreno[col][fila])
        for dcol in [-1, 0, 1]:
            for dfila in [-1, 0, 1]:
                neighbor_col = (col + dcol) % columnas
                neighbor_row = (fila + dfila) % filas
                probability = random.random()
                # 40% de probabilidad de solo un nivel más alto
                if probability < 0.45:
                    new_topography = max(1, min(5, current_topography + 1))
                    terreno[neighbor_col][neighbor_row] = str(new_topography)
                # 40% de probabilidad de solo un nivel más bajo
                elif probability < 0.9:
                    new_topography = max(1, min(5, current_topography - 1))
                    terreno[neighbor_col][neighbor_row] = str(new_topography)
                # 10% de probabilidad del mismo nivel
                elif probability < 0.95:
                    terreno[neighbor_col][neighbor_row] = str(current_topography)
                # 10% de probabilidad de cualquier otro nivel
                else:
                    new_topography = random.choice([t for t in niveles_topograficos if t != str(current_topography)])
                    terreno[neighbor_col][neighbor_row] = new_topography

# Adición de los obstáculos
num_obstaculos = int(columnas * filas * 0.1)
for _ in range(num_obstaculos):
    col_obstaculo = random.randint(0, columnas - 1)
    fila_obstaculo = random.randint(0, filas - 1)
    terreno[col_obstaculo][fila_obstaculo] = marcador_obstaculo

# Adición del agua
num_agua = int(columnas * filas * 0.02)
num_water_clusters = 5
cluster_size = num_agua // num_water_clusters
# Creación de conjunto de agua (cuerpo de agua)
for cluster in range(num_water_clusters):
    col_agua = random.randint(0, columnas - 1)
    fila_agua = random.randint(0, filas - 1)
    for col in range(cluster_size):
        for fila in range(cluster_size):
            col = (col_agua + col) % columnas
            row = (fila_agua + fila) % filas
            terreno[col][row] = marcador_agua
            
            # Modificación de los puntos colindantes para una probabilidad del 60% que sea nivel topográfico 1 y 40% cualquier otro
            for dcol in [-1, 0, 1]:
                for dfila in [-1, 0, 1]:
                    neighbor_col = (col + dcol) % columnas
                    neighbor_row = (row + dfila) % filas
                    if terreno[neighbor_col][neighbor_row] != marcador_agua:
                        if random.random() < 0.6:
                            terreno[neighbor_col][neighbor_row] = '1'
                        else:
                            random_topography = random.choice(['2', '3', '4', '5', '6'])
                            terreno[neighbor_col][neighbor_row] = random_topography

# Adicion del rover en la primera fila
col_rover = random.randint(1, columnas - 2)
fila_rover = random.randint(1, filas - 2)
terreno[col_rover][fila_rover] = marcador_rover
dcol, dfila = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
neighbor_col = (col_rover + dcol) % columnas
neighbor_row = (fila_rover + dfila) % filas
terreno[neighbor_col][neighbor_row] = str(random.randint(1, 6))
# Adición del límite del ambiente
for col in range(columnas):
    for fila in range(filas):
        if col == 0 or col == columnas - 1 or fila == 0 or fila == filas - 1:
            terreno[col][fila] = '%'

# Tranformación del terreno en un dataframe
terrain_df = []
for col in range(columnas):
    for fila in range(filas):
        terrain_type = terreno[col][fila]
        label = labels.get(terrain_type, terrain_type)  # Use specified label if available, otherwise use terrain type
        terrain_df.append({'x': col, 'y': fila, 'terrain': label, 'color': terrain_type, 'symbol': terrain_type})

# Creación del mapa interactivo
fig = px.scatter(
    terrain_df, 
    x='x', 
    y='y', 
    color='color',
    symbol='symbol',
    hover_data={'terrain': True, 'color': False, 'symbol': False},
    symbol_map=symbols_topografia,
    color_discrete_map=colors_topografia
)
fig.update_traces(marker=dict(size=20))
fig.update_layout(
    #plot_bgcolor='rgba(204, 102, 77, .9)',
    showlegend=False
)
fig.show()

import simpleai.search as ss
from simpleai.search.local import _exp_schedule
class RoverSearchProblem(ss.SearchProblem):
    '''
    Clase para representar el problema de búsqueda del rover.
    '''

    def __init__(self, environment, start_position):
        # Inicialización del problema de búsqueda con el estado inicial y las posiciones de destino posibles.
        super(RoverSearchProblem, self).__init__(initial_state=start_position)
        self.environment = environment
        self._actions = [('↑', (0, 1)),
                         ('↓', (0, -1)),
                         ('↗', (1, 1)),
                         ('↖', (-1, 1)), 
                         ('↘', (1, -1)),
                         ('↙', (-1, -1))]
        self.rover_terrain_value = {}

    def actions(self, state):
        '''
        Genera las acciones posibles desde un estado.
        '''
        return [action for action in self._actions if self._is_valid(state, self.result(state, action))]

    def _is_valid(self, current_state, new_state):
        '''
        Verifica si un movimiento es válido.
        '''
        current_col, current_row = current_state
        new_col, new_row = new_state
        if self.environment[new_col][new_row] in ['#', '%']:
            return False
        
        if (current_col, current_row) in self.rover_terrain_value:
            current_terrain = self.rover_terrain_value[(current_col, current_row)]
        else:
            current_terrain = random.randint(1, 7)
            self.rover_terrain_value[(current_col, current_row)] = current_terrain

        if (new_col, new_row) in self.rover_terrain_value:
            next_terrain = self.rover_terrain_value[(new_col, new_row)]
        else:
            next_terrain = (
                int(self.environment[new_col][new_row]) if self.environment[new_col][new_row] != 'R' else random.randint(1, 7)
            )

        # Check if the next terrain is within the allowable range
        if next_terrain in (current_terrain, current_terrain + 1, current_terrain - 1):
            return new_state
    
    def result(self, state, action):
        '''
        Resultado de aplicar una acción a un estado.
        '''
        col, row = state
        dc, dr = action[1]
        new_col, new_row = col + dc, row + dr
        new_state = (new_col, new_row)
        return new_state

    def value(self, state):
        col, row = state
        terrain_value = self.environment[col][row]
        if terrain_value == '0':
            return 0
        elif terrain_value == '1':
            return -1
        elif terrain_value == '2':
            return -2
        elif terrain_value == '3':
            return -3
        elif terrain_value == '4':
            return -4
        elif terrain_value == '5':
            return -5
        elif terrain_value == '6':
            return -6
        else:
            return random.randint(1,7)*-1
        
    def generate_random_state(self):
        """
        Generates a random state for genetic search. It's mainly used for the
        seed states in the initilization of genetic search.
        """
        lista = [0, 0]
        lista[0] = random.randint(2, columnas - 2)
        lista[1] = random.randint(2, filas - 2)
        return lista
    
temp = 18
def temperature(time):
    # Calcular la temperatura
    res = temp - time
    print("Temperatura: ",time,res)
    return res

# Definición de la posición inicial
start_position = (col_rover, fila_rover)

# Ejecución de los algoritmos con el problema propuesto:
rover_problem = RoverSearchProblem(terreno, start_position)
#print('\nSolución utilizando Hill-Climbing')
#result = ss.hill_climbing(rover_problem)
#print('\nSolución utilizando Random-Restart Hill Climbing')
#result = ss.hill_climbing_random_restarts(rover_problem,500)
print('\nSolución utilizando Simulated Annealing')
result = ss.simulated_annealing(rover_problem,temperature)
#result = ss.simulated_annealing(rover_problem,_exp_schedule)

print('\nCamino a la solución:')
for item,state in result.path():
    print(state)
    print(f"Valor = {abs(rover_problem.value(state))}\n")


