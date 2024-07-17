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


from simpleai.search import SearchProblem, astar, breadth_first, depth_first
class RoverSearchProblem(SearchProblem):
    '''
    Clase para representar el problema de búsqueda del rover.
    '''

    def __init__(self, environment, start_position, goal_positions):
        # Inicialización del problema de búsqueda con el estado inicial y las posiciones de destino posibles.
        super(RoverSearchProblem, self).__init__(initial_state=start_position)
        self.environment = environment
        self.goals = goal_positions
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
        
        current_terrain = self.environment[current_col][current_row]
        if current_terrain in ['R', '#', '0']:
            if self.rover_terrain_value:
                current_terrain = self.rover_terrain_value[(current_col, current_row)]
            else:
                current_terrain = random.randint(1, 6)
                self.rover_terrain_value[(current_col, current_row)] = current_terrain
        else:
            current_terrain = int(current_terrain)
        next_terrain = (
            self.rover_terrain_value[(new_col, new_row)] if self.environment[new_col][new_row] == 'R' else int(self.environment[new_col][new_row])
        )
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

    def is_goal(self, state):
        '''
        Verifica si el estado es uno de los objetivos.
        '''
        return state in self.goals

    def heuristic(self, state):
        '''
        Función heurística para la búsqueda A*.
        '''
        # Heurística 1: Distancia Manhattan
        distances1 = [abs(state[0] - goal[0]) + abs(state[1] - goal[1]) for goal in self.goals]
        # Heurística 2: Distancia Euclidiana ponderada por altitud
        distances2 = [(((goal[0] - state[0])**2 + (goal[1] - state[1])**2)**0.5) for goal in self.goals]
        # Heurísticas combinadas
        h = min(distances1) * 0.5 + min(distances2) * 0.5
        return h
    
    def cost(self, state, action, next_state):
        '''
        Costo de una acción.
        '''
        col, row = state
        dc, dr = action[1]
        new_col, new_row = col + dc, row + dr
        if self.environment[new_col][new_row] == 'R':
            cost = self.rover_terrain_value[(new_col, new_row)]
        else:
            cost = int(self.environment[new_col][new_row])
        return cost

# Definición de la posición inicial y las posiciones de destino
start_position = (col_rover, fila_rover)
water_positions = []
for col in range(columnas):
    for fila in range(filas):
        if terreno[col][fila] == '0':
            water_positions.append((col, fila))
if water_positions:
    goal_position = water_positions


# Ejecución de los algoritmos con el problema propuesto:
    
rover_problem = RoverSearchProblem(terreno, start_position, goal_position)
#print('\nSolucion utilizando BFS')
#result = breadth_first(rover_problem)
#print('\nSolución utilizando DFS')
#result = depth_first(rover_problem,True)
print('\nSolución utilizando A*')
result = astar(rover_problem)

print(f'Estado inicial: {start_position}')
print(f'Nivel de terreno del estado inicial: {rover_problem.rover_terrain_value[start_position]}')
print(f'Estado final: {result.state}')
print(f'Numero de pasos en la solución: {len(result.path())-1}')
print(f'Costo de la solución: {result.cost}')
print('\nCamino a la solución:')
path = result.path()
for i in range(len(path) - 1):
    current_step = path[i]
    next_step = path[i + 1]
    print(f'Rover se mueve en dirección {next_step[0]} para llegar de {current_step[1]} a {next_step[1]}')
print()


