# -*- coding: utf-8 -*-
"""
minesweeper_env.py — Ambiente Campo Minado para Gymnasium
=========================================================

Este arquivo implementa o jogo Campo Minado como um "ambiente" de Reinforcement Learning
usando a biblioteca Gymnasium (a versão moderna do OpenAI Gym).

POR QUE PRECISAMOS DESTE ARQUIVO?
----------------------------------
Algoritmos de RL (como PPO, A2C, DQN) não entendem "jogos" diretamente.
Eles entendem 3 coisas:
  1. ESTADO (observation): o que o agente "vê" — no nosso caso, o tabuleiro
  2. AÇÃO (action): o que o agente pode "fazer" — no nosso caso, clicar numa célula
  3. RECOMPENSA (reward): feedback numérico — positivo = bom, negativo = ruim

Este arquivo traduz o Campo Minado para essa linguagem.

INSPIRAÇÃO:
-----------
A lógica do jogo (colocação de minas, contagem de vizinhos, abertura em cascata)
foi inspirada no repositório github.com/aylint/gym-minesweeper, mas reescrita
do zero com a API moderna do Gymnasium e várias melhorias.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# =============================================================================
# CONSTANTES — Valores que representam o estado de cada célula
# =============================================================================

MINE = -1       # Célula contém uma mina (o agente nunca vê isso até pisar nela)
CLOSED = -2     # Célula fechada (não revelada ainda)
# Valores >= 0 indicam quantas minas vizinhas a célula tem (0 a 8)


# =============================================================================
# FUNÇÕES AUXILIARES DO JOGO
# =============================================================================
# Estas funções implementam a lógica pura do Campo Minado.
# São funções "puras" — recebem dados e retornam resultados, sem efeitos colaterais.

def place_mines(board_size, num_mines):
    """
    Cria um tabuleiro e distribui minas aleatoriamente.

    POR QUE SEPARAR NUMA FUNÇÃO?
    Porque o tabuleiro precisa ser regenerado a cada novo episódio (reset).
    Ter uma função separada mantém o código organizado e testável.

    Parâmetros:
        board_size (int): tamanho do tabuleiro (board_size x board_size)
        num_mines (int): quantidade de minas a serem colocadas

    Retorna:
        np.array: tabuleiro board_size x board_size, com MINE nas posições com mina e 0 no resto
    """
    board = np.zeros((board_size, board_size), dtype=np.int32)

    # Gera posições aleatórias únicas para as minas
    # np.random.choice sem reposição garante que não colocamos 2 minas na mesma célula
    total_cells = board_size * board_size
    mine_positions = np.random.choice(total_cells, size=num_mines, replace=False)

    for pos in mine_positions:
        row = pos // board_size  # Divisão inteira para obter a linha
        col = pos % board_size   # Resto da divisão para obter a coluna
        board[row, col] = MINE

    return board


def count_neighbour_mines(board, x, y):
    """
    Conta quantas minas existem ao redor de uma célula (x, y).

    No Campo Minado, cada célula pode ter até 8 vizinhas:
        NW  N  NE
        W  [C]  E
        SW  S  SE

    Parâmetros:
        board (np.array): tabuleiro com as minas
        x (int): linha da célula
        y (int): coluna da célula

    Retorna:
        int: número de minas vizinhas (0 a 8)
    """
    board_size = board.shape[0]
    count = 0

    # Percorre todas as células vizinhas (incluindo diagonais)
    for dx in range(-1, 2):       # -1, 0, 1
        for dy in range(-1, 2):   # -1, 0, 1
            nx, ny = x + dx, y + dy

            # Verifica se a posição vizinha está dentro do tabuleiro
            if 0 <= nx < board_size and 0 <= ny < board_size:
                if board[nx, ny] == MINE:
                    count += 1

    return count


def open_cells_cascade(board, visible_board, x, y):
    """
    Abre células em cascata quando uma célula com 0 minas vizinhas é revelada.

    POR QUE ISSO EXISTE?
    No Campo Minado real, quando você clica numa célula que não tem nenhuma mina vizinha,
    o jogo automaticamente abre todas as células vizinhas (e continua recursivamente
    se essas também tiverem 0 vizinhas). Isso acelera o jogo.

    Parâmetros:
        board (np.array): tabuleiro real (com posição das minas)
        visible_board (np.array): tabuleiro visível para o jogador
        x, y (int): posição da célula a partir da qual iniciar a cascata

    Retorna:
        int: número de novas células abertas pela cascata
    """
    board_size = board.shape[0]
    cells_opened = 0

    # Usamos uma pilha (stack) em vez de recursão para evitar estouro de pilha
    # em tabuleiros grandes
    stack = [(x, y)]

    while stack:
        cx, cy = stack.pop()

        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = cx + dx, cy + dy

                # Verifica se está dentro do tabuleiro E se a célula ainda está fechada
                if (0 <= nx < board_size and 0 <= ny < board_size
                        and visible_board[nx, ny] == CLOSED):

                    neighbour_count = count_neighbour_mines(board, nx, ny)
                    visible_board[nx, ny] = neighbour_count
                    cells_opened += 1

                    # Se esta célula também tem 0 vizinhas, adiciona à pilha
                    # para continuar a cascata
                    if neighbour_count == 0:
                        stack.append((nx, ny))

    return cells_opened


# =============================================================================
# AMBIENTE GYMNASIUM
# =============================================================================

class MinesweeperEnv(gym.Env):
    """
    Ambiente de Campo Minado compatível com Gymnasium.

    POR QUE HERDAR DE gym.Env?
    --------------------------
    A classe gym.Env é a interface padrão que todos os algoritmos de RL esperam.
    Ao implementar os métodos reset() e step(), qualquer algoritmo (PPO, A2C, DQN)
    pode treinar neste ambiente sem modificação.

    ESPAÇO DE OBSERVAÇÃO (o que o agente vê):
    ------------------------------------------
    Matriz board_size × board_size com valores normalizados entre 0 e 1:
      - 0.0 = célula fechada (CLOSED)
      - 0.1 = célula aberta com 0 vizinhas minas
      - 0.2 a 0.9 = célula aberta com 1-8 vizinhas minas
      - 1.0 = mina (game over)

    POR QUE NORMALIZAR? Redes neurais funcionam MUITO melhor com valores
    entre 0 e 1 do que com valores arbitrários como -2, -1, 0, 1, ...

    ESPAÇO DE AÇÃO (o que o agente pode fazer):
    -------------------------------------------
    Um número inteiro de 0 a (board_size² - 1) representando a célula a clicar.
    Exemplo num tabuleiro 5×5: ação 7 = linha 1, coluna 2 (7 // 5 = 1, 7 % 5 = 2)

    ACTION MASKING (máscara de ações):
    ----------------------------------
    O agente recebe uma máscara indicando quais ações são válidas (células fechadas).
    Isso IMPEDE que o agente "desperdice" jogadas clicando em células já abertas.
    Sem isso, o agente levaria MUITO mais tempo para aprender.
    """

    metadata = {"render_modes": ["ansi", "human"], "render_fps": 4}

    def __init__(self, board_size=5, num_mines=3, render_mode=None):
        """
        Inicializa o ambiente.

        Parâmetros:
            board_size (int): tamanho do tabuleiro (default: 5 para treino rápido)
            num_mines (int): número de minas (default: 3)
            render_mode (str): "ansi" para texto, "human" para visualização
        """
        super().__init__()

        self.board_size = board_size
        self.num_mines = num_mines
        self.render_mode = render_mode

        # ---------------------------------------------------------------
        # ESPAÇO DE OBSERVAÇÃO
        # ---------------------------------------------------------------
        # Box = espaço contínuo (valores de ponto flutuante)
        # shape = (1, board_size, board_size) — uma matriz 3D para suportar CnnPolicy (Canal de Imagem)
        # low=0.0, high=1.0 — todos os valores normalizados neste intervalo
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1, self.board_size, self.board_size),
            dtype=np.float32
        )

        # ---------------------------------------------------------------
        # ESPAÇO DE AÇÃO
        # ---------------------------------------------------------------
        # Discrete = espaço discreto (números inteiros)
        # O agente escolhe um número de 0 a (board_size² - 1)
        # que é convertido em coordenada (x, y)
        self.action_space = spaces.Discrete(self.board_size * self.board_size)

        # Variáveis internas (serão inicializadas no reset)
        self.board = None          # Tabuleiro real (com posição das minas)
        self.visible_board = None  # O que o agente vê
        self.action_mask = None    # Máscara de ações válidas
        self.steps_taken = 0       # Contador de passos no episódio atual

    def _place_mines(self):
        """
        Cria um tabuleiro com minas usando o gerador de números aleatórios
        do Gymnasium (self.np_random), garantindo reprodutibilidade.

        POR QUE NÃO USAR np.random?
        O Gymnasium tem seu próprio gerador (self.np_random) que é controlado
        pela seed passada no reset(). Se usarmos np.random global, o ambiente
        não será determinístico — mesma seed pode gerar tabuleiros diferentes.
        """
        board = np.zeros((self.board_size, self.board_size), dtype=np.int32)
        total_cells = self.board_size * self.board_size

        # Usa o gerador do Gymnasium para reprodutibilidade
        mine_positions = self.np_random.choice(
            total_cells, size=self.num_mines, replace=False
        )

        for pos in mine_positions:
            row = pos // self.board_size
            col = pos % self.board_size
            board[row, col] = MINE

        return board

    def _normalize_board(self, board):
        """
        Normaliza o tabuleiro para valores entre 0 e 1.

        POR QUE NORMALIZAR?
        Redes neurais aprendem melhor quando os inputs estão numa faixa padrão.
        Sem normalização, valores como -2 e +1000 confundem o treinamento.

        Mapeamento:
            CLOSED (-2) → 0.0
            0 vizinhas   → 0.1
            1 vizinha    → 0.2
            ...
            8 vizinhas   → 0.9
            MINE (-1)    → 1.0
        """
        normalized = np.zeros_like(board, dtype=np.float32)

        # Células fechadas → 0.0
        normalized[board == CLOSED] = 0.0

        # Células abertas (0-8 vizinhas) → 0.1 a 0.9
        for i in range(9):
            normalized[board == i] = (i + 1) / 10.0

        # Minas → 1.0
        normalized[board == MINE] = 1.0

        # Para CnnPolicy, adicionamos uma dimensão extra "canal" -> (1, board_size, board_size)
        return np.expand_dims(normalized, axis=0)

    def _get_action_mask(self):
        """
        Retorna a máscara de ações válidas.

        POR QUE ACTION MASKING?
        Sem máscara, o agente pode escolher clicar em células já abertas.
        Isso é um desperdício — a ação não faz nada útil.
        A máscara diz ao agente: "estas são as células que você PODE clicar".

        Retorna:
            np.array: vetor de tamanho board_size², com 1 para ações válidas
                      e 0 para inválidas
        """
        # Uma célula é válida se ainda está fechada (CLOSED)
        mask = (self.visible_board.flatten() == CLOSED).astype(np.int8)
        return mask

    def reset(self, seed=None, options=None):
        """
        Reinicia o ambiente para um novo episódio (nova partida).

        POR QUE reset() EXISTE?
        No RL, o agente joga MILHARES de partidas. Cada partida é um "episódio".
        Quando o episódio termina (vitória ou derrota), chamamos reset() para
        começar um novo jogo com um tabuleiro novo.

        Retorna:
            observation (np.array): tabuleiro normalizado (tudo fechado no início)
            info (dict): informações extras, inclui a máscara de ações
        """
        super().reset(seed=seed)

        # Cria novo tabuleiro com minas em posições aleatórias
        # Usa self.np_random (gerador do Gymnasium) para garantir reprodutibilidade
        self.board = self._place_mines()

        # Tabuleiro visível: tudo fechado no início
        self.visible_board = np.full(
            (self.board_size, self.board_size), CLOSED, dtype=np.int32
        )

        self.steps_taken = 0

        # Atualiza a máscara de ações
        self.action_mask = self._get_action_mask()

        observation = self._normalize_board(self.visible_board)
        info = {"action_mask": self.action_mask}

        return observation, info

    def step(self, action):
        """
        Executa uma ação (clicar numa célula) e retorna o resultado.

        ESTE É O MÉTODO MAIS IMPORTANTE DO AMBIENTE.
        É aqui que o ciclo do RL acontece:
          Agente escolhe ação → Ambiente processa → Retorna resultado

        Parâmetros:
            action (int): índice da célula a clicar (0 a board_size²-1)

        Retorna:
            observation (np.array): novo estado do tabuleiro
            reward (float): recompensa pela ação
            terminated (bool): True se o jogo acabou (vitória ou derrota)
            truncated (bool): True se cortamos o episódio (limite de passos)
            info (dict): informações extras (action_mask, etc.)
        """
        # Converte ação (número único) em coordenada (linha, coluna)
        x = action // self.board_size
        y = action % self.board_size

        self.steps_taken += 1

        # ---------------------------------------------------------------
        # CASO 1: Célula já aberta (ação inválida)
        # ---------------------------------------------------------------
        # Se o agente clicou numa célula que já está aberta,
        # damos uma punição leve e NÃO terminamos o jogo.
        if self.visible_board[x, y] != CLOSED:
            observation = self._normalize_board(self.visible_board)
            self.action_mask = self._get_action_mask()
            info = {"action_mask": self.action_mask}
            return observation, -2.0, False, False, info

        # ---------------------------------------------------------------
        # CASO 2: Pisou numa mina! (GAME OVER)
        # ---------------------------------------------------------------
        if self.board[x, y] == MINE:
            self.visible_board[x, y] = MINE
            observation = self._normalize_board(self.visible_board)
            self.action_mask = self._get_action_mask()
            info = {"action_mask": self.action_mask}

            # Recompensa: -10.0 (forte penalidade)
            # terminated = True (o jogo acabou porque o agente perdeu)
            return observation, -10.0, True, False, info

        # ---------------------------------------------------------------
        # CASO 3: Célula segura!
        # ---------------------------------------------------------------
        neighbours = count_neighbour_mines(self.board, x, y)
        self.visible_board[x, y] = neighbours
        cells_opened = 1  # A célula que acabamos de abrir

        # Se tem 0 vizinhas com mina, abre em cascata
        if neighbours == 0:
            cascade_opened = open_cells_cascade(
                self.board, self.visible_board, x, y
            )
            cells_opened += cascade_opened

        # ---------------------------------------------------------------
        # CÁLCULO DA RECOMPENSA
        # ---------------------------------------------------------------
        # POR QUE REWARD SHAPING?
        # Se déssemos recompensa APENAS na vitória (+1000), o agente
        # teria dificuldade de aprender, porque vitórias são raras no início.
        # Reward shaping dá "dicas" intermediárias para guiar o aprendizado.

        # Recompensa base: +2.0 por célula segura revelada (Reward Shaping melhorado)
        reward = 2.0

        # Bônus por cascata: +1.0 por cada célula extra aberta
        # Isso ensina o agente que clicar em áreas com 0 vizinhas é vantajoso
        if cells_opened > 1:
            reward += 1.0 * (cells_opened - 1)

        # ---------------------------------------------------------------
        # VERIFICAÇÃO DE VITÓRIA
        # ---------------------------------------------------------------
        # O agente vence quando TODAS as células seguras foram reveladas.
        # Ou seja, as únicas células fechadas restantes são as minas.
        closed_cells = np.count_nonzero(self.visible_board == CLOSED)

        if closed_cells == self.num_mines:
            # VITÓRIA! Bônus grande.
            reward += 20.0
            observation = self._normalize_board(self.visible_board)
            self.action_mask = self._get_action_mask()
            info = {"action_mask": self.action_mask}
            return observation, reward, True, False, info

        # Jogo continua...
        observation = self._normalize_board(self.visible_board)
        self.action_mask = self._get_action_mask()
        info = {"action_mask": self.action_mask}

        return observation, reward, False, False, info

    def action_masks(self):
        """
        Retorna a máscara de ações para uso com MaskablePPO do sb3-contrib.

        POR QUE UM MÉTODO SEPARADO?
        O MaskablePPO da biblioteca sb3-contrib espera um método chamado
        action_masks() no ambiente. Ele chama este método automaticamente
        antes de cada ação para saber quais ações são válidas.
        """
        return self.action_mask

    def render(self):
        """
        Renderiza o tabuleiro no terminal.

        POR QUE render()?
        Para podermos VISUALIZAR o que o agente está fazendo.
        Essencial para debug e para demonstrar o projeto.
        """
        if self.render_mode == "ansi" or self.render_mode == "human":
            return self._render_ansi()

    def _render_ansi(self):
        """
        Renderiza o tabuleiro como texto no terminal.

        Símbolos:
            ■ = célula fechada (não revelada)
            * = mina
            0-8 = número de minas vizinhas
        """
        symbols = {
            CLOSED: "#",
            MINE: "*",
        }

        lines = []
        # Cabeçalho com índices das colunas
        header = "    " + "  ".join(f"{j}" for j in range(self.board_size))
        lines.append(header)
        lines.append("   " + "---" * self.board_size)

        for i in range(self.board_size):
            row_str = f"{i} | "
            for j in range(self.board_size):
                val = self.visible_board[i, j]
                if val in symbols:
                    row_str += f"{symbols[val]}  "
                else:
                    row_str += f"{val}  "
            lines.append(row_str)

        output = "\n".join(lines)
        print(output)
        return output

    def get_board_string(self):
        """
        Retorna uma representação em string do tabuleiro para visualização.
        Útil para o notebook, sem imprimir automaticamente.
        """
        symbols = {CLOSED: "#", MINE: "X"}
        lines = []
        header = "    " + "  ".join(f"{j}" for j in range(self.board_size))
        lines.append(header)
        lines.append("   " + "---" * self.board_size)

        for i in range(self.board_size):
            row_str = f"{i} | "
            for j in range(self.board_size):
                val = self.visible_board[i, j]
                if val in symbols:
                    row_str += f"{symbols[val]}  "
                else:
                    row_str += f"{val}  "
            lines.append(row_str)

        return "\n".join(lines)


# =============================================================================
# TESTE RÁPIDO — Roda quando executamos este arquivo diretamente
# =============================================================================
# Isso permite testar o ambiente sem precisar do notebook:
#   python minesweeper_env.py

if __name__ == "__main__":
    print("=" * 50)
    print("TESTE DO AMBIENTE MINESWEEPER")
    print("=" * 50)

    # Cria o ambiente
    env = MinesweeperEnv(board_size=5, num_mines=3, render_mode="human")
    obs, info = env.reset()

    print("\nTabuleiro inicial (tudo fechado):")
    env.render()

    print(f"\nShape da observacao: {obs.shape}")
    print(f"Acoes validas: {info['action_mask'].sum()} de {env.action_space.n}")

    # Joga aleatoriamente até o jogo acabar
    print("\n--- Jogando aleatoriamente ---\n")
    done = False
    total_reward = 0
    step_num = 0

    while not done:
        # Escolhe uma ação aleatória VÁLIDA (usando a máscara)
        valid_actions = np.where(info["action_mask"] == 1)[0]
        if len(valid_actions) == 0:
            break
        action = np.random.choice(valid_actions)

        x, y = action // env.board_size, action % env.board_size
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step_num += 1

        print(f"Passo {step_num}: clicou em ({x}, {y}) -> recompensa: {reward:+.1f}")
        env.render()
        print()

    print(f"Jogo terminou! Recompensa total: {total_reward:.1f}")
    print(f"{'VITORIA!' if total_reward > 0 and terminated else 'DERROTA!'}")

    # Teste de compatibilidade com Gymnasium
    print("\n--- Verificando compatibilidade com Gymnasium ---")
    try:
        from gymnasium.utils.env_checker import check_env
        test_env = MinesweeperEnv(board_size=5, num_mines=3)
        check_env(test_env, skip_render_check=True)
        print("[OK] Ambiente compativel com Gymnasium!")
    except Exception as e:
        print(f"[AVISO] Problema encontrado: {e}")
