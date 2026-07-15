# -*- coding: utf-8 -*-
"""
visualizar_agente.py — Interface Grafica para ver o Agente Jogando
===================================================================

Este script cria uma janela Pygame que mostra o agente de RL
jogando Campo Minado passo a passo, com animacoes e cores.

Uso:
    python visualizar_agente.py

    Controles:
        ESPACO  = Avancar um passo
        ENTER   = Jogar automatico (modo continuo)
        R       = Nova partida
        ESC     = Sair
        1       = Velocidade lenta (1 passo/s)
        2       = Velocidade media (3 passos/s)
        3       = Velocidade rapida (10 passos/s)
        A       = Modo aleatorio (compara com agente treinado)
        T       = Modo treinado (agente PPO)
"""

import numpy as np
import pygame
import sys
import os
import time

from minesweeper_env import MinesweeperEnv, MINE, CLOSED

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample_obs = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_obs).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations):
        return self.linear(self.cnn(observations))
# =============================================================================
# CONFIGURACOES VISUAIS
# =============================================================================

# Tamanho do tabuleiro
BOARD_SIZE = 5
NUM_MINES = 3

# Dimensoes da janela
CELL_SIZE = 80          # Tamanho de cada celula em pixels
MARGIN = 4              # Espaco entre celulas
HEADER_HEIGHT = 120     # Altura do cabecalho com informacoes
FOOTER_HEIGHT = 80      # Altura do rodape

# Calculo automatico do tamanho da janela
BOARD_PIXEL_SIZE = BOARD_SIZE * (CELL_SIZE + MARGIN) + MARGIN
WINDOW_WIDTH = max(BOARD_PIXEL_SIZE + 40, 500)
WINDOW_HEIGHT = HEADER_HEIGHT + BOARD_PIXEL_SIZE + FOOTER_HEIGHT + 20

# Cores (RGB)
COLORS = {
    "bg":           (30, 30, 46),       # Fundo escuro
    "header_bg":    (45, 45, 65),       # Fundo do cabecalho
    "cell_closed":  (88, 91, 112),      # Celula fechada
    "cell_hover":   (108, 111, 132),    # Celula com destaque (ultima jogada)
    "cell_open":    (49, 50, 68),       # Celula aberta
    "cell_mine":    (243, 139, 168),    # Mina (vermelho)
    "cell_safe0":   (166, 227, 161),    # 0 vizinhas (verde claro)
    "cell_flag":    (249, 226, 175),    # Bandeira (amarelo)
    "text_white":   (205, 214, 244),    # Texto branco
    "text_dim":     (147, 153, 178),    # Texto cinza
    "text_green":   (166, 227, 161),    # Texto verde
    "text_red":     (243, 139, 168),    # Texto vermelho
    "text_yellow":  (249, 226, 175),    # Texto amarelo
    "text_blue":    (137, 180, 250),    # Texto azul
    "border":       (69, 71, 90),       # Borda
    "win_bg":       (30, 102, 49),      # Fundo vitoria
    "lose_bg":      (139, 30, 30),      # Fundo derrota
}

# Cores dos numeros (1-8) no Campo Minado classico
NUMBER_COLORS = {
    0: COLORS["cell_safe0"],
    1: (137, 180, 250),     # Azul
    2: (166, 227, 161),     # Verde
    3: (243, 139, 168),     # Vermelho
    4: (180, 142, 255),     # Roxo
    5: (250, 179, 135),     # Laranja
    6: (148, 226, 213),     # Ciano
    7: (245, 224, 220),     # Branco
    8: (147, 153, 178),     # Cinza
}


def load_trained_model():
    """
    Tenta carregar o modelo treinado.
    Se nao existir, retorna None (usara agente aleatorio).
    """
    model_paths = [
        "models/ppo_minesweeper_optimized.zip",
        "models/ppo_minesweeper.zip",
    ]

    for path in model_paths:
        if os.path.exists(path):
            try:
                from sb3_contrib import MaskablePPO
                model = MaskablePPO.load(path)
                print(f"Modelo carregado: {path}")
                return model, path
            except Exception as e:
                print(f"Erro ao carregar {path}: {e}")

    print("Nenhum modelo treinado encontrado em models/")
    print("Rode 'python campo_minado_rl.py' primeiro para treinar.")
    print("Por enquanto, usando apenas agente aleatorio.\n")
    return None, None


class MinesweeperVisualizer:
    """
    Interface grafica para visualizar o agente jogando Campo Minado.
    """

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Campo Minado - Agente de RL")

        # Fontes
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 20)
        self.font_small = pygame.font.SysFont("Consolas", 14)
        self.font_cell = pygame.font.SysFont("Consolas", 36, bold=True)
        self.font_cell_small = pygame.font.SysFont("Consolas", 24, bold=True)

        # Ambiente
        self.env = MinesweeperEnv(board_size=BOARD_SIZE, num_mines=NUM_MINES)
        self.obs = None
        self.info = None
        self.reset_game()

        # Modelo treinado
        self.model, self.model_path = load_trained_model()

        # Estado do jogo
        self.auto_play = False
        self.speed = 3            # Passos por segundo
        self.last_step_time = 0
        self.use_trained = True   # True = agente treinado, False = aleatorio
        self.last_action = None   # Ultima acao tomada (para highlight)
        self.game_over = False
        self.won = False

        # Estatisticas
        self.total_games = 0
        self.total_wins = 0
        self.stats_trained = {"games": 0, "wins": 0}
        self.stats_random = {"games": 0, "wins": 0}

        # Animacao
        self.flash_timer = 0
        self.flash_cell = None

    def reset_game(self):
        """Reinicia o jogo."""
        self.obs, self.info = self.env.reset()
        self.game_over = False
        self.won = False
        self.step_count = 0
        self.total_reward = 0
        self.last_action = None
        self.flash_cell = None

    def get_agent_action(self):
        """
        Obtem a proxima acao do agente (treinado ou aleatorio).
        """
        if self.use_trained and self.model is not None:
            action, _ = self.model.predict(
                self.obs, deterministic=True,
                action_masks=self.info["action_mask"]
            )
            return int(action)
        else:
            # Agente aleatorio
            valid_actions = np.where(self.info["action_mask"] == 1)[0]
            if len(valid_actions) == 0:
                return None
            return np.random.choice(valid_actions)

    def do_step(self):
        """Executa um passo do agente."""
        if self.game_over:
            return

        action = self.get_agent_action()
        if action is None:
            return

        self.last_action = action
        self.flash_cell = (action // BOARD_SIZE, action % BOARD_SIZE)
        self.flash_timer = pygame.time.get_ticks()

        self.obs, reward, terminated, truncated, self.info = self.env.step(action)
        self.step_count += 1
        self.total_reward += reward

        if terminated or truncated:
            self.game_over = True
            self.won = (reward > 5)  # Bonus de vitoria
            self.total_games += 1

            if self.use_trained:
                self.stats_trained["games"] += 1
                if self.won:
                    self.stats_trained["wins"] += 1
                    self.total_wins += 1
            else:
                self.stats_random["games"] += 1
                if self.won:
                    self.stats_random["wins"] += 1
                    self.total_wins += 1

    def draw_header(self):
        """Desenha o cabecalho com informacoes."""
        # Fundo do cabecalho
        header_rect = pygame.Rect(0, 0, WINDOW_WIDTH, HEADER_HEIGHT)
        pygame.draw.rect(self.screen, COLORS["header_bg"], header_rect)
        pygame.draw.line(self.screen, COLORS["border"],
                         (0, HEADER_HEIGHT), (WINDOW_WIDTH, HEADER_HEIGHT), 2)

        # Titulo
        title = self.font_large.render("CAMPO MINADO - RL", True, COLORS["text_white"])
        self.screen.blit(title, (WINDOW_WIDTH // 2 - title.get_width() // 2, 10))

        # Info do agente
        if self.use_trained and self.model is not None:
            agent_text = "Agente: PPO Treinado"
            agent_color = COLORS["text_green"]
        elif self.use_trained and self.model is None:
            agent_text = "Agente: SEM MODELO (treine primeiro)"
            agent_color = COLORS["text_red"]
        else:
            agent_text = "Agente: Aleatorio"
            agent_color = COLORS["text_yellow"]

        agent_surface = self.font_medium.render(agent_text, True, agent_color)
        self.screen.blit(agent_surface, (15, 50))

        # Status
        speed_text = f"Vel: {self.speed}/s"
        auto_text = "AUTO" if self.auto_play else "MANUAL"
        status = self.font_small.render(
            f"Passo: {self.step_count}  |  Reward: {self.total_reward:+.1f}  |  {auto_text}  |  {speed_text}",
            True, COLORS["text_dim"]
        )
        self.screen.blit(status, (15, 80))

        # Estatisticas
        if self.use_trained:
            stats = self.stats_trained
        else:
            stats = self.stats_random
        win_rate = (stats["wins"] / stats["games"] * 100) if stats["games"] > 0 else 0
        stats_text = self.font_small.render(
            f"Partidas: {stats['games']}  |  Vitorias: {stats['wins']}  |  Win Rate: {win_rate:.0f}%",
            True, COLORS["text_dim"]
        )
        self.screen.blit(stats_text, (15, 100))

    def draw_board(self):
        """Desenha o tabuleiro."""
        board = self.env.visible_board
        real_board = self.env.board

        # Offset para centralizar o tabuleiro
        offset_x = (WINDOW_WIDTH - BOARD_PIXEL_SIZE) // 2
        offset_y = HEADER_HEIGHT + 10

        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                x = offset_x + col * (CELL_SIZE + MARGIN) + MARGIN
                y = offset_y + row * (CELL_SIZE + MARGIN) + MARGIN
                cell_rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)

                val = board[row, col]

                # Determina a cor da celula
                if val == CLOSED:
                    color = COLORS["cell_closed"]
                elif val == MINE:
                    color = COLORS["cell_mine"]
                elif val == 0:
                    color = (60, 90, 60)  # Verde escuro para 0
                else:
                    # Celula aberta com numero
                    color = COLORS["cell_open"]

                # Highlight da ultima jogada
                current_time = pygame.time.get_ticks()
                if (self.flash_cell == (row, col) and
                        current_time - self.flash_timer < 500):
                    # Brilho pulsante
                    brightness = abs(((current_time - self.flash_timer) % 500) - 250) / 250
                    color = tuple(min(255, int(c + 60 * brightness)) for c in color)

                # Desenha a celula
                pygame.draw.rect(self.screen, color, cell_rect, border_radius=6)

                # Borda sutil
                pygame.draw.rect(self.screen, COLORS["border"], cell_rect,
                                 width=1, border_radius=6)

                # Texto dentro da celula
                if val == CLOSED:
                    # Celula fechada — mostra quadrado
                    inner_rect = pygame.Rect(x + 20, y + 20, CELL_SIZE - 40, CELL_SIZE - 40)
                    pygame.draw.rect(self.screen, COLORS["text_dim"], inner_rect,
                                     border_radius=3)
                elif val == MINE:
                    # Mina — mostra X
                    text = self.font_cell.render("X", True, (255, 255, 255))
                    text_rect = text.get_rect(center=cell_rect.center)
                    self.screen.blit(text, text_rect)
                elif val == 0:
                    # Sem vizinhas — mostra ponto
                    text = self.font_cell_small.render(".", True, COLORS["text_dim"])
                    text_rect = text.get_rect(center=cell_rect.center)
                    self.screen.blit(text, text_rect)
                else:
                    # Numero 1-8
                    num_color = NUMBER_COLORS.get(val, COLORS["text_white"])
                    text = self.font_cell.render(str(val), True, num_color)
                    text_rect = text.get_rect(center=cell_rect.center)
                    self.screen.blit(text, text_rect)

        # Se game over, mostra onde estavam TODAS as minas
        if self.game_over and not self.won:
            for row in range(BOARD_SIZE):
                for col in range(BOARD_SIZE):
                    if real_board[row, col] == MINE and board[row, col] != MINE:
                        x = offset_x + col * (CELL_SIZE + MARGIN) + MARGIN
                        y = offset_y + row * (CELL_SIZE + MARGIN) + MARGIN
                        cell_rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
                        # Semi-transparente vermelho
                        s = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
                        s.fill((243, 139, 168, 100))
                        self.screen.blit(s, (x, y))
                        # Desenha X menor
                        text = self.font_cell_small.render("x", True, (243, 139, 168))
                        text_rect = text.get_rect(center=cell_rect.center)
                        self.screen.blit(text, text_rect)

    def draw_footer(self):
        """Desenha o rodape com controles."""
        footer_y = WINDOW_HEIGHT - FOOTER_HEIGHT
        pygame.draw.line(self.screen, COLORS["border"],
                         (0, footer_y), (WINDOW_WIDTH, footer_y), 2)

        # Game over overlay
        if self.game_over:
            if self.won:
                msg = "VITORIA!"
                color = COLORS["text_green"]
            else:
                msg = "DERROTA!"
                color = COLORS["text_red"]

            result = self.font_large.render(msg, True, color)
            self.screen.blit(result, (WINDOW_WIDTH // 2 - result.get_width() // 2, footer_y + 5))

            hint = self.font_small.render("R = Nova partida  |  ESC = Sair", True, COLORS["text_dim"])
            self.screen.blit(hint, (WINDOW_WIDTH // 2 - hint.get_width() // 2, footer_y + 45))
        else:
            controls = self.font_small.render(
                "ESPACO=Passo  ENTER=Auto  R=Reset  T/A=Treinado/Aleatorio  1/2/3=Velocidade",
                True, COLORS["text_dim"]
            )
            self.screen.blit(controls, (WINDOW_WIDTH // 2 - controls.get_width() // 2, footer_y + 10))

            # Proxima acao do agente
            if not self.game_over:
                action = self.get_agent_action()
                if action is not None:
                    row, col = action // BOARD_SIZE, action % BOARD_SIZE
                    next_text = self.font_small.render(
                        f"Proxima acao: celula ({row}, {col})",
                        True, COLORS["text_blue"]
                    )
                    self.screen.blit(next_text,
                                     (WINDOW_WIDTH // 2 - next_text.get_width() // 2, footer_y + 35))

    def draw_legend(self):
        """Desenha legenda lateral (se couber)."""
        pass  # Simplificado para caber na tela

    def run(self):
        """Loop principal da interface grafica."""
        clock = pygame.time.Clock()
        running = True

        print("\n--- Interface Grafica Iniciada ---")
        print("Controles:")
        print("  ESPACO  = Avancar um passo")
        print("  ENTER   = Modo automatico")
        print("  R       = Nova partida")
        print("  T       = Agente treinado (PPO)")
        print("  A       = Agente aleatorio")
        print("  1/2/3   = Velocidade")
        print("  ESC     = Sair\n")

        while running:
            current_time = time.time()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

                    elif event.key == pygame.K_SPACE:
                        # Passo manual
                        if self.game_over:
                            self.reset_game()
                        else:
                            self.do_step()

                    elif event.key == pygame.K_RETURN:
                        # Toggle auto play
                        self.auto_play = not self.auto_play

                    elif event.key == pygame.K_r:
                        # Reset
                        self.reset_game()
                        self.auto_play = False

                    elif event.key == pygame.K_t:
                        # Agente treinado
                        self.use_trained = True
                        self.reset_game()

                    elif event.key == pygame.K_a:
                        # Agente aleatorio
                        self.use_trained = False
                        self.reset_game()

                    elif event.key == pygame.K_1:
                        self.speed = 1
                    elif event.key == pygame.K_2:
                        self.speed = 3
                    elif event.key == pygame.K_3:
                        self.speed = 10

            # Auto play
            if self.auto_play and not self.game_over:
                if current_time - self.last_step_time >= 1.0 / self.speed:
                    self.do_step()
                    self.last_step_time = current_time

            # Auto reset apos game over no modo auto
            if self.auto_play and self.game_over:
                if current_time - self.last_step_time >= 1.5:  # Pausa de 1.5s
                    self.reset_game()
                    self.last_step_time = current_time

            # Desenha
            self.screen.fill(COLORS["bg"])
            self.draw_header()
            self.draw_board()
            self.draw_footer()

            pygame.display.flip()
            clock.tick(60)  # 60 FPS

        pygame.quit()
        print("\nInterface encerrada.")
        print(f"Estatisticas finais:")
        if self.stats_trained["games"] > 0:
            wr = self.stats_trained["wins"] / self.stats_trained["games"] * 100
            print(f"  PPO:      {self.stats_trained['wins']}/{self.stats_trained['games']} vitorias ({wr:.0f}%)")
        if self.stats_random["games"] > 0:
            wr = self.stats_random["wins"] / self.stats_random["games"] * 100
            print(f"  Aleatorio: {self.stats_random['wins']}/{self.stats_random['games']} vitorias ({wr:.0f}%)")


if __name__ == "__main__":
    viz = MinesweeperVisualizer()
    viz.run()
