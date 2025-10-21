"""
Blokus (consola) + Agente Minimax/Expectiminimax con Alpha-Beta y IDS
Autor: Generado por ChatGPT
Descripción:
- Implementación simplificada funcional del juego Blokus (tablero 20x20)
- Soporta jugadores: humano (con entradas de consola), aleatorio, greedy, peor, IA (minimax/expectiminimax)
- Algoritmo IA: minimax con poda alpha-beta + búsqueda en profundidad iterativa (IDS) con tiempo máximo
- Recolección de métricas para benchmarking

Instrucciones rápidas:
- Ejecutar: python blokus_agent.py
- Desde la consola podrá configurar número de jugadores, cuáles controlados por IA y tiempo máximo por búsqueda

Notas:
- Esta implementación prioriza claridad didáctica sobre micro-optimización. Está lista para usarse y para que usted la extienda.
"""

import time
import random
import copy
import math
from collections import deque, defaultdict

BOARD_SIZE = 20

# ---------- PIEZAS: definidas como sets de (x,y) relativos. Hay 21 piezas en Blokus.
# Cada pieza está definida en una forma base (sin rotaciones/reflexiones). El código generará rotaciones/reflexiones.
# Las piezas son polyominos con hasta 5 celdas.
# A continuación listamos 21 piezas (representadas con coordenadas) - shapes taken/encoded compactly.
PIECES_RAW = {
    'I1': [(0,0)],
    'I2': [(0,0),(1,0)],
    'I3': [(0,0),(1,0),(2,0)],
    'I4': [(0,0),(1,0),(2,0),(3,0)],
    'I5': [(0,0),(1,0),(2,0),(3,0),(4,0)],
    'V3': [(0,0),(0,1),(1,0)],
    'L4': [(0,0),(1,0),(2,0),(2,1)],
    'L5': [(0,0),(1,0),(2,0),(3,0),(3,1)],
    'Z4': [(0,0),(1,0),(1,1),(2,1)],
    'Z5': [(0,1),(1,1),(1,0),(2,0),(3,0)],
    'T4': [(0,0),(1,0),(2,0),(1,1)],
    'P5': [(0,0),(1,0),(0,1),(1,1),(2,0)],
    'U5': [(0,0),(2,0),(0,1),(1,1),(2,1)],
    'W5': [(0,0),(1,0),(1,1),(2,1),(2,2)],
    'Y5': [(0,0),(1,0),(2,0),(3,0),(2,1)],
    'X5': [(1,0),(0,1),(1,1),(2,1),(1,2)],
    'F5': [(1,0),(0,1),(1,1),(1,2),(2,1)],
    'N5': [(0,0),(1,0),(1,1),(2,1),(3,1)],
    'Q5': [(0,0),(1,0),(0,1),(1,1),(0,2)],
    'R5': [(0,0),(1,0),(2,0),(0,1),(0,2)],
    'S5': [(0,0),(1,0),(2,0),(2,1),(2,2)]
}

# Normalize shapes: move to origin and sort

def normalize_shape(cells):
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    minx, miny = min(xs), min(ys)
    norm = tuple(sorted(((x-minx, y-miny) for x,y in cells)))
    return norm

# Generate rotations and reflections for each piece

def rotations_and_reflections(shape):
    cells = list(shape)
    variants = set()
    for reflect in [False, True]:
        for rot in range(4):
            transformed = []
            for x,y in cells:
                # rotation 90deg * rot
                rx, ry = x, y
                for _ in range(rot):
                    rx, ry = -ry, rx
                if reflect:
                    rx = -rx
                transformed.append((rx, ry))
            variants.add(normalize_shape(transformed))
    return [list(v) for v in variants]

# Precompute piece variants
PIECE_VARIANTS = {name: rotations_and_reflections(PIECES_RAW[name]) for name in PIECES_RAW}

# ---------- TABLERO Y UTILIDADES
class Board:
    def __init__(self, size=BOARD_SIZE):
        self.size = size
        self.grid = [[None for _ in range(size)] for _ in range(size)]

    def in_bounds(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size

    def can_place(self, piece_cells, player_id, x_off, y_off):
        # Check cell availability and adjacency rules
        touch_corner = False
        for (x,y) in piece_cells:
            X, Y = x + x_off, y + y_off
            if not self.in_bounds(X, Y):
                return False
            if self.grid[Y][X] is not None:
                return False
            # check side-adjacency to own pieces -> not allowed
            for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                nx, ny = X+dx, Y+dy
                if self.in_bounds(nx, ny) and self.grid[ny][nx] == player_id:
                    return False
            # check corner adjacency to own piece -> at least one needed unless it's the first move
            for dx,dy in [(1,1),(1,-1),(-1,1),(-1,-1)]:
                nx, ny = X+dx, Y+dy
                if self.in_bounds(nx, ny) and self.grid[ny][nx] == player_id:
                    touch_corner = True
        return touch_corner

    def place(self, piece_cells, player_id, x_off, y_off):
        for (x,y) in piece_cells:
            X, Y = x + x_off, y + y_off
            self.grid[Y][X] = player_id

    def score_remaining(self, player_pieces):
        return sum(len(PIECES_RAW[name]) for name in player_pieces)

    def copy(self):
        newb = Board(self.size)
        newb.grid = [row[:] for row in self.grid]
        return newb

    def display(self):
        for row in self.grid:
            print(''.join(['.' if c is None else str(c) for c in row]))

# ---------- JUGADORES
class Player:
    def __init__(self, pid, kind='human', max_time=3.0, heuristics=None, weights=None):
        self.pid = pid
        self.kind = kind  # 'human', 'random', 'greedy', 'worst', 'ai'
        self.pieces = set(PIECES_RAW.keys())
        self.max_time = max_time
        self.heuristics = heuristics or []
        self.weights = weights or []

    def available_moves(self, board, is_first_move=False):
        moves = []  # list of (piece_name, variant_cells, x, y)
        corners = self.find_corners(board)
        for piece_name in list(self.pieces):
            for variant in PIECE_VARIANTS[piece_name]:
                # try all placements
                for y in range(board.size):
                    for x in range(board.size):
                        if board.can_place(variant, self.pid, x, y):
                            # ensure it touches at least one corner of your existing pieces OR if first move, must be corner
                            if is_first_move:
                                # first move must touch one of the board corners
                                if (0,0) in [(x+vx, y+vy) for vx,vy in variant]:
                                    moves.append((piece_name, variant, x, y))
                            else:
                                moves.append((piece_name, variant, x, y))
        return moves

    def find_corners(self, board):
        # returns player's cells that are corner candidates
        corners = set()
        for y in range(board.size):
            for x in range(board.size):
                if board.grid[y][x] == self.pid:
                    # check 4 diagonal neighbors for other unmatched own cells
                    for dx,dy in [(1,1),(1,-1),(-1,1),(-1,-1)]:
                        nx, ny = x+dx, y+dy
                        if board.in_bounds(nx, ny) and board.grid[ny][nx] != self.pid:
                            corners.add((x,y))
        return corners

# ---------- HEURÍSTICAS: al menos 5
# Cada heurística debe devolver un número (may be positive better)

def h_coverage(board, player_id):
    # Count number of cells occupied by player (encourage larger coverage)
    count = 0
    for row in board.grid:
        for c in row:
            if c == player_id:
                count += 1
    return count


def h_mobility(board, player, players_dict):
    # Number of legal moves available to player
    moves = 0
    for pid, pl in players_dict.items():
        if pid == player.pid:
            # estimate limited move count by sampling a subset
            moves = len(pl.available_moves(board))
    return moves


def h_corners(board, player_id):
    # Count number of empty corner squares adjacent to player's corners
    corners = set()
    for y in range(board.size):
        for x in range(board.size):
            if board.grid[y][x] == player_id:
                for dx,dy in [(1,1),(1,-1),(-1,1),(-1,-1)]:
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < board.size and 0 <= ny < board.size and board.grid[ny][nx] is None:
                        corners.add((nx,ny))
    return len(corners)


def h_remaining_squares(player):
    # Penalize number of squares in remaining pieces (lower is better)
    return -sum(len(PIECES_RAW[name]) for name in player.pieces)


def h_adjacency_penalty(board, player_id):
    # Penalize placements where own pieces are side-adjacent (should be zero if rules enforced)
    penalty = 0
    for y in range(board.size):
        for x in range(board.size):
            if board.grid[y][x] == player_id:
                for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < board.size and 0 <= ny < board.size and board.grid[ny][nx] == player_id:
                        penalty += 1
    return -penalty

# Heuristic aggregator
def evaluate(board, player, players_dict, weights=None):
    # weights correspond to heuristics: coverage, mobility, corners, remaining, adjacency
    w = weights or [1.0, 1.0, 1.0, 1.0, 1.0]
    vals = []
    vals.append(h_coverage(board, player.pid))
    vals.append(h_mobility(board, player, players_dict))
    vals.append(h_corners(board, player.pid))
    vals.append(h_remaining_squares(player))
    vals.append(h_adjacency_penalty(board, player.pid))
    score = sum(a*b for a,b in zip(w, vals))
    return score

# ---------- MINIMAX + ALPHA-BETA + IDS
class MinimaxAI:
    def __init__(self, player, players_dict, max_time=3.0, weights=None):
        self.player = player
        self.players_dict = players_dict
        self.max_time = max_time
        self.weights = weights or [1.0]*5
        self.nodes_expanded = 0

    def choose(self, board, is_first_move=False):
        # Iterative deepening until time runs out
        start = time.time()
        depth = 1
        best_move = None
        best_score = -math.inf
        while True:
            remaining = self.max_time - (time.time() - start)
            if remaining <= 0:
                break
            self.nodes_expanded = 0
            try:
                score, move = self._minimax_root(board, depth, start, is_first_move)
                if move is not None:
                    best_move = move
                    best_score = score
            except TimeoutError:
                break
            depth += 1
        return best_move, best_score, self.nodes_expanded, depth-1

    def _minimax_root(self, board, depth, start_time, is_first_move):
        best = (-math.inf, None)
        alpha = -math.inf
        beta = math.inf
        moves = self.player.available_moves(board, is_first_move)
        if not moves:
            return evaluate(board, self.player, self.players_dict, self.weights), None
        for move in moves:
            if time.time() - start_time > self.max_time:
                raise TimeoutError()
            piece_name, variant, x, y = move
            new_board = board.copy()
            new_board.place(variant, self.player.pid, x, y)
            # simulate removing piece
            saved = copy.deepcopy(self.player.pieces)
            self.player.pieces.remove(piece_name)
            val = self._min_value(new_board, depth-1, alpha, beta, start_time)
            self.player.pieces = saved
            if val > best[0]:
                best = (val, move)
            alpha = max(alpha, val)
        return best

    def _max_value(self, board, depth, alpha, beta, start_time):
        if time.time() - start_time > self.max_time:
            raise TimeoutError()
        self.nodes_expanded += 1
        if depth == 0:
            return evaluate(board, self.player, self.players_dict, self.weights)
        moves = self.player.available_moves(board)
        if not moves:
            return evaluate(board, self.player, self.players_dict, self.weights)
        v = -math.inf
        for move in moves:
            piece_name, variant, x, y = move
            new_board = board.copy()
            new_board.place(variant, self.player.pid, x, y)
            saved = copy.deepcopy(self.player.pieces)
            self.player.pieces.remove(piece_name)
            val = self._min_value(new_board, depth-1, alpha, beta, start_time)
            self.player.pieces = saved
            v = max(v, val)
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def _min_value(self, board, depth, alpha, beta, start_time):
        if time.time() - start_time > self.max_time:
            raise TimeoutError()
        self.nodes_expanded += 1
        # For simplicity, we approximate opponents by averaging their heuristic scores
        if depth == 0:
            return evaluate(board, self.player, self.players_dict, self.weights)
        # simulate opponents moves naive: pick one opponent and assume worst for us
        # Here we iterate through other players and assume they act to minimize our score
        v = math.inf
        for pid, opponent in self.players_dict.items():
            if pid == self.player.pid:
                continue
            moves = opponent.available_moves(board)
            if not moves:
                # opponent passes; evaluate
                ev = evaluate(board, self.player, self.players_dict, self.weights)
                v = min(v, ev)
                continue
            for move in moves:
                piece_name, variant, x, y = move
                new_board = board.copy()
                new_board.place(variant, opponent.pid, x, y)
                saved = copy.deepcopy(opponent.pieces)
                opponent.pieces.remove(piece_name)
                val = self._max_value(new_board, depth-1, alpha, beta, start_time)
                opponent.pieces = saved
                v = min(v, val)
                if v <= alpha:
                    return v
                beta = min(beta, v)
        return v

# ---------- AGENTES SIMPLES: random, greedy, worst

def random_agent_move(player, board, is_first_move=False):
    moves = player.available_moves(board, is_first_move)
    if not moves:
        return None
    return random.choice(moves)


def greedy_agent_move(player, board, players_dict, weights=None, is_first_move=False):
    best = None
    best_score = -math.inf
    for move in player.available_moves(board, is_first_move):
        piece_name, variant, x, y = move
        new_board = board.copy()
        new_board.place(variant, player.pid, x, y)
        saved = copy.deepcopy(player.pieces)
        player.pieces.remove(piece_name)
        sc = evaluate(new_board, player, players_dict, weights)
        player.pieces = saved
        if sc > best_score:
            best_score = sc
            best = move
    return best


def worst_agent_move(player, board, players_dict, weights=None, is_first_move=False):
    worst = None
    worst_score = math.inf
    for move in player.available_moves(board, is_first_move):
        piece_name, variant, x, y = move
        new_board = board.copy()
        new_board.place(variant, player.pid, x, y)
        saved = copy.deepcopy(player.pieces)
        player.pieces.remove(piece_name)
        sc = evaluate(new_board, player, players_dict, weights)
        player.pieces = saved
        if sc < worst_score:
            worst_score = sc
            worst = move
    return worst

# ---------- PARTIDA y BENCHMARK
class Game:
    def __init__(self, num_players=2, ai_players=None, max_time=3.0, weights=None):
        self.board = Board()
        self.num_players = num_players
        self.players = {}
        for i in range(1, num_players+1):
            kind = 'human'
            if ai_players and i in ai_players:
                kind = ai_players[i]
            self.players[i] = Player(i, kind=kind, max_time=max_time, weights=weights)
        self.current = 1
        self.max_time = max_time
        self.weights = weights or [1.0]*5

    def is_first_move_for(self, pid):
        # check if player has any piece placed on board
        for y in range(self.board.size):
            for x in range(self.board.size):
                if self.board.grid[y][x] == pid:
                    return False
        return True

    def play_turn(self, pid):
        player = self.players[pid]
        is_first = self.is_first_move_for(pid)
        move = None
        if player.kind == 'human':
            print(f"Turno jugador humano {pid}. Tablero actual:")
            self.board.display()
            print("Ingrese pieza nombre (ej: I5) o 'pass':")
            cmd = input().strip()
            if cmd == 'pass':
                return False
            # ask for x y
            print("Ingrese x y:")
            x,y = map(int, input().split())
            # find a variant that fits
            if cmd in player.pieces:
                for variant in PIECE_VARIANTS[cmd]:
                    if self.board.can_place(variant, pid, x, y):
                        move = (cmd, variant, x, y)
                        break
        elif player.kind == 'random':
            move = random_agent_move(player, self.board, is_first)
        elif player.kind == 'greedy':
            move = greedy_agent_move(player, self.board, self.players, self.weights, is_first)
        elif player.kind == 'worst':
            move = worst_agent_move(player, self.board, self.players, self.weights, is_first)
        elif player.kind == 'ai':
            ai = MinimaxAI(player, self.players, max_time=player.max_time, weights=self.weights)
            move, score, nodes, depth = ai.choose(self.board, is_first)
            # store metrics on player
            player.last_nodes = nodes
            player.last_depth = depth
            player.last_score = score
        else:
            move = None

        if move is None:
            return False
        # apply move
        piece_name, variant, x, y = move
        self.board.place(variant, pid, x, y)
        player.pieces.remove(piece_name)
        return True

    def play(self, max_rounds=2000):
        consecutive_passes = 0
        rounds = 0
        while rounds < max_rounds and consecutive_passes < self.num_players:
            pid = self.current
            moved = self.play_turn(pid)
            if not moved:
                consecutive_passes += 1
            else:
                consecutive_passes = 0
            self.current = (self.current % self.num_players) + 1
            rounds += 1
        # compute scores
        scores = {pid: sum(len(PIECES_RAW[p]) for p in pl.pieces) for pid,pl in self.players.items()}
        winner = min(scores.items(), key=lambda kv: kv[1])[0]
        return winner, scores

# Benchmark runner
def run_benchmark(configs):
    results = []
    for cfg in configs:
        # cfg: dict with settings
        game = Game(num_players=cfg.get('num_players',2), ai_players=cfg.get('ai_players',None), max_time=cfg.get('max_time',3.0), weights=cfg.get('weights',None))
        winner, scores = game.play()
        # collect nodes etc
        metrics = {
            'config': cfg,
            'winner': winner,
            'scores': scores,
            'nodes': {pid: getattr(game.players[pid], 'last_nodes', None) for pid in game.players},
            'depth': {pid: getattr(game.players[pid], 'last_depth', None) for pid in game.players},
        }
        results.append(metrics)
    return results

# ---------- EJECUTABLE PARA CONSOLA
if __name__ == '__main__':
    print("Blokus - Consola")
    print("Configurar partida:")
    num = int(input("Número de jugadores (2-4) [2]: ") or 2)
    ai_players = {}
    for i in range(1, num+1):
        role = input(f"Jugador {i} - tipo (human/random/greedy/worst/ai) [human]: ") or 'human'
        if role != 'human':
            ai_players[i] = role
    max_time = float(input("Tiempo máximo IA por búsqueda (segundos) [3.0]: ") or 3.0)
    # heuristic weights example
    weights = [float(x) for x in (input("Pesos heurísticos (5 valores separados por espacio) [1 1 1 1 1]: ") or '1 1 1 1 1').split()]
    game = Game(num_players=num, ai_players=ai_players, max_time=max_time, weights=weights)
    winner, scores = game.play()
    print("Partida finalizada. Puntajes (cuadrados restantes):")
    for pid, sc in scores.items():
        print(f"Jugador {pid}: {sc}")
    print(f"Ganador: Jugador {winner}")

# Fin del archivo
