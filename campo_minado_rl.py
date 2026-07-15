# -*- coding: utf-8 -*-
"""campo_minado_rl.py

Projeto Final - Inteligencia Computacional
============================================
Aprendizado por Reforco aplicado ao Campo Minado

Este notebook treina agentes de RL para jogar Campo Minado,
compara algoritmos (PPO vs A2C) e faz tuning de hiperparametros com Optuna.

Para rodar no Google Colab:
    1. Faça upload deste arquivo e do minesweeper_env.py
    2. Execute as celulas em ordem

Para rodar localmente:
    pip install -r requirements.txt
    python campo_minado_rl.py
"""

# =============================================================================
# SECAO 1: SETUP E INSTALACAO DE DEPENDENCIAS
# =============================================================================
# Se estiver no Google Colab, descomente as linhas abaixo:
# !pip install gymnasium stable-baselines3 sb3-contrib optuna matplotlib

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend nao-interativo para salvar graficos
import matplotlib.pyplot as plt
import time
import os
import warnings
warnings.filterwarnings('ignore')  # Limpa warnings durante treinamento

# Importa nosso ambiente customizado
from minesweeper_env import MinesweeperEnv

print("=" * 60)
print("  PROJETO FINAL: CAMPO MINADO COM APRENDIZADO POR REFORCO")
print("=" * 60)


# =============================================================================
# SECAO 2: ENTENDENDO O AMBIENTE
# =============================================================================
# Antes de treinar qualquer agente, precisamos entender como o ambiente funciona.
# Vamos criar o ambiente e explorar os espacos de observacao e acao.

print("\n" + "=" * 60)
print("  SECAO 2: ENTENDENDO O AMBIENTE")
print("=" * 60)

# Cria o ambiente com tabuleiro 5x5 e 3 minas
env = MinesweeperEnv(board_size=5, num_mines=3, render_mode="human")
obs, info = env.reset(seed=42)

print("\n--- O que o agente 've' (observacao) ---")
print(f"Formato (shape): {obs.shape}")
print(f"Tipo: {obs.dtype}")
print(f"Valores: min={obs.min()}, max={obs.max()}")
print("\nNo inicio, todas as celulas estao fechadas (valor 0.0):")
print(obs)

print("\n--- Espacos do ambiente ---")
print(f"Espaco de observacao: {env.observation_space}")
print(f"  -> Matriz {env.board_size}x{env.board_size} com valores entre 0.0 e 1.0")
print(f"Espaco de acao: {env.action_space}")
print(f"  -> Um numero inteiro de 0 a {env.action_space.n - 1}")
print(f"  -> Cada numero mapeia para uma celula: acao 7 = linha {7 // 5}, coluna {7 % 5}")

print("\n--- Mascara de acoes (action mask) ---")
print(f"Total de acoes validas: {info['action_mask'].sum()} de {env.action_space.n}")
print("A mascara impede o agente de clicar em celulas ja abertas.")

# Demonstra uma jogada
print("\n--- Demonstrando uma jogada ---")
action = 12  # Centro do tabuleiro (linha 2, coluna 2)
print(f"Acao escolhida: {action} -> celula ({action // 5}, {action % 5})")
obs, reward, terminated, truncated, info = env.step(action)
print(f"Recompensa recebida: {reward:+.1f}")
print(f"Jogo terminou? {terminated}")
print(f"Observacao apos a jogada:")
print(obs)
env.render()

env.close()


# =============================================================================
# SECAO 3: AGENTE ALEATORIO (BASELINE)
# =============================================================================
# O agente aleatorio e nosso ponto de referencia MINIMO.
# Se o agente treinado nao for MELHOR que o aleatorio, nao aprendeu nada.
#
# POR QUE PRECISAMOS DE UM BASELINE?
# Imagine que seu agente treinado vence 20% das partidas. Isso e bom?
# Depende! Se o aleatorio vence 15%, seu agente e so um pouco melhor.
# Mas se o aleatorio vence 2%, seu agente e 10x melhor!
# O baseline da CONTEXTO para os resultados.

print("\n" + "=" * 60)
print("  SECAO 3: AGENTE ALEATORIO (BASELINE)")
print("=" * 60)


def evaluate_random_agent(env, num_episodes=1000):
    """
    Avalia o agente aleatorio jogando num_episodes partidas.

    O agente simplesmente escolhe uma acao ALEATORIA entre as acoes validas.
    Usamos a mascara de acoes para garantir que ele so clique em celulas fechadas.

    Retorna:
        dict: metricas de desempenho (win_rate, avg_reward, avg_cells_opened)
    """
    wins = 0
    total_rewards = []
    cells_opened_list = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        cells_opened = 0

        while not done:
            # Escolhe acao aleatoria entre as VALIDAS
            valid_actions = np.where(info["action_mask"] == 1)[0]
            if len(valid_actions) == 0:
                break
            action = np.random.choice(valid_actions)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            cells_opened += 1

        total_rewards.append(episode_reward)
        cells_opened_list.append(cells_opened)

        # Verifica se venceu (recompensa > 10 indica vitoria)
        if episode_reward > 10:
            wins += 1

    results = {
        "win_rate": wins / num_episodes * 100,
        "avg_reward": np.mean(total_rewards),
        "std_reward": np.std(total_rewards),
        "avg_cells_opened": np.mean(cells_opened_list),
        "total_episodes": num_episodes,
    }
    return results


print("\nAvaliando agente aleatorio em 1000 partidas...")
env_eval = MinesweeperEnv(board_size=5, num_mines=3)
random_results = evaluate_random_agent(env_eval, num_episodes=1000)

print(f"\n--- Resultados do Agente Aleatorio ---")
print(f"Taxa de vitoria:        {random_results['win_rate']:.1f}%")
print(f"Recompensa media:       {random_results['avg_reward']:.2f} (+/- {random_results['std_reward']:.2f})")
print(f"Celulas abertas (media): {random_results['avg_cells_opened']:.1f}")

print("\nEste e nosso BASELINE. O agente treinado precisa ser melhor que isso!")


# =============================================================================
# SECAO 4: TREINAMENTO COM PPO (Proximal Policy Optimization)
# =============================================================================
# PPO e um dos algoritmos de RL mais populares e estaveis.
#
# COMO O PPO FUNCIONA? (Explicacao simplificada)
# -----------------------------------------------
# 1. O agente joga varias partidas e coleta experiencias
# 2. Usa essas experiencias para atualizar sua "politica" (estrategia)
# 3. A atualizacao e LIMITADA (proximal = proximo) para nao mudar demais
#    de uma vez — isso evita que o agente "desaprenda" o que ja sabia
#
# POR QUE PPO?
# - E estavel: nao "explode" durante o treinamento como outros algoritmos
# - E eficiente: aprende razoavelmente rapido
# - E amplamente usado: a maioria dos projetos de RL comeca com PPO
#
# ACTION MASKING
# O MaskablePPO e uma versao especial do PPO que respeita a mascara de acoes.
# Ou seja, o agente NUNCA vai tentar clicar numa celula ja aberta.
# Isso acelera MUITO o aprendizado.

print("\n" + "=" * 60)
print("  SECAO 4: TREINAMENTO COM PPO")
print("=" * 60)

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker


def mask_fn(env):
    """
    Funcao que retorna a mascara de acoes.
    O MaskablePPO chama essa funcao antes de cada acao para saber
    quais acoes sao validas.
    """
    return env.action_masks()


class TrainingMetricsCallback(BaseCallback):
    """
    Callback customizado para coletar metricas DURANTE o treinamento.

    POR QUE UM CALLBACK?
    O treinamento roda por milhares de passos. O callback nos permite
    "espiar" o progresso sem interromper o treinamento.
    Coletamos metricas a cada N episodios para depois plotar graficos.
    """

    def __init__(self, eval_env, eval_freq=5000, n_eval_episodes=100, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.results_history = []
        self.timesteps_history = []

    def _on_step(self):
        if self.num_timesteps % self.eval_freq == 0:
            wins = 0
            total_rewards = []

            for _ in range(self.n_eval_episodes):
                obs, info = self.eval_env.reset()
                done = False
                ep_reward = 0

                while not done:
                    valid_actions = np.where(info["action_mask"] == 1)[0]
                    if len(valid_actions) == 0:
                        break
                    # Usa o modelo para prever a melhor acao
                    action_masks = info["action_mask"]
                    action, _ = self.model.predict(
                        obs, deterministic=True,
                        action_masks=action_masks
                    )
                    obs, reward, terminated, truncated, info = self.eval_env.step(int(action))
                    done = terminated or truncated
                    ep_reward += reward

                total_rewards.append(ep_reward)
                if ep_reward > 10:
                    wins += 1

            win_rate = wins / self.n_eval_episodes * 100
            avg_reward = np.mean(total_rewards)

            self.results_history.append({
                "win_rate": win_rate,
                "avg_reward": avg_reward,
            })
            self.timesteps_history.append(self.num_timesteps)

            print(f"  [{self.num_timesteps:>7} steps] Win rate: {win_rate:5.1f}% | Recompensa media: {avg_reward:+7.2f}")

        return True


# --- Configuracao do treinamento PPO ---
# Estes sao os HIPERPARAMETROS do PPO. Cada um controla um aspecto do aprendizado:

PPO_HYPERPARAMS = {
    "learning_rate": 3e-4,    # Velocidade de aprendizado (muito alto = instavel, muito baixo = lento)
    "n_steps": 256,           # Quantos passos coletar antes de cada atualizacao
    "batch_size": 64,         # Tamanho do mini-batch para atualizacao
    "n_epochs": 4,            # Quantas vezes reusar os dados coletados
    "gamma": 0.99,            # Fator de desconto (importancia de recompensas futuras)
    "ent_coef": 0.01,         # Coeficiente de entropia (incentiva exploracao)
    "clip_range": 0.2,        # Limita o quanto a politica pode mudar por atualizacao
}

TOTAL_TIMESTEPS_PPO = 500_000  # Total de passos de treinamento

print(f"\nHiperparametros do PPO:")
for key, value in PPO_HYPERPARAMS.items():
    print(f"  {key}: {value}")
print(f"\nTotal de timesteps: {TOTAL_TIMESTEPS_PPO:,}")


# Cria o ambiente com action masking
def make_env():
    env = MinesweeperEnv(board_size=5, num_mines=3)
    env = ActionMasker(env, mask_fn)
    return env


print("\nIniciando treinamento PPO...")
print("(Metricas serao exibidas a cada 5000 passos)\n")

train_env_ppo = make_env()
eval_env_ppo = MinesweeperEnv(board_size=5, num_mines=3)

ppo_callback = TrainingMetricsCallback(
    eval_env=eval_env_ppo,
    eval_freq=5000,
    n_eval_episodes=100,
)

start_time_ppo = time.time()

model_ppo = MaskablePPO(
    "CnnPolicy",
    train_env_ppo,
    verbose=0,
    **PPO_HYPERPARAMS
)

model_ppo.learn(total_timesteps=TOTAL_TIMESTEPS_PPO, callback=ppo_callback)

time_ppo = time.time() - start_time_ppo
print(f"\nTreinamento PPO concluido em {time_ppo:.1f} segundos ({time_ppo/60:.1f} min)")

# Salva o modelo treinado
os.makedirs("models", exist_ok=True)
model_ppo.save("models/ppo_minesweeper")
print("Modelo PPO salvo em models/ppo_minesweeper.zip")


# =============================================================================
# SECAO 5: TREINAMENTO COM A2C (Advantage Actor-Critic)
# =============================================================================
# A2C e outro algoritmo de policy gradient, similar ao PPO mas mais simples.
#
# DIFERENCA ENTRE PPO E A2C:
# - PPO: atualiza a politica de forma "segura" (clipping), mais estavel
# - A2C: atualiza a politica diretamente, mais rapido mas pode ser instavel
#
# POR QUE COMPARAR?
# O professor sugere comparar algoritmos e seus hiperparametros.
# Comparar PPO vs A2C mostra que entendemos as diferencas entre eles.

print("\n" + "=" * 60)
print("  SECAO 5: TREINAMENTO COM A2C")
print("=" * 60)

# NOTA IMPORTANTE: sb3-contrib so tem MaskablePPO, NAO tem MaskableA2C.
# Por isso, usaremos o A2C padrao do stable-baselines3.
# O A2C nao tera action masking — ele pode tentar clicar em celulas ja abertas.
# Nosso ambiente da penalidade -1.0 nesses casos, entao o A2C aprende a evitar.
# Isso tambem serve como COMPARACAO: action masking (PPO) vs sem masking (A2C).

from stable_baselines3 import A2C


class TrainingMetricsCallbackStandard(BaseCallback):
    """
    Callback para agentes SEM action masking (como A2C padrao).
    A diferenca e que nao passamos action_masks no predict().
    """

    def __init__(self, eval_env, eval_freq=5000, n_eval_episodes=100, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.results_history = []
        self.timesteps_history = []

    def _on_step(self):
        if self.num_timesteps % self.eval_freq == 0:
            wins = 0
            total_rewards = []

            for _ in range(self.n_eval_episodes):
                obs, info = self.eval_env.reset()
                done = False
                ep_reward = 0

                steps = 0
                while not done and steps < 100:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self.eval_env.step(int(action))
                    done = terminated or truncated
                    ep_reward += reward
                    steps += 1

                total_rewards.append(ep_reward)
                if ep_reward > 10:
                    wins += 1

            win_rate = wins / self.n_eval_episodes * 100
            avg_reward = np.mean(total_rewards)

            self.results_history.append({
                "win_rate": win_rate,
                "avg_reward": avg_reward,
            })
            self.timesteps_history.append(self.num_timesteps)

            print(f"  [{self.num_timesteps:>7} steps] Win rate: {win_rate:5.1f}% | Recompensa media: {avg_reward:+7.2f}")

        return True


A2C_HYPERPARAMS = {
    "learning_rate": 7e-4,    # A2C geralmente usa learning rate mais alto
    "n_steps": 256,           # Passos antes de cada atualizacao
    "gamma": 0.99,            # Fator de desconto
    "ent_coef": 0.01,         # Entropia para exploracao
}

TOTAL_TIMESTEPS_A2C = 500_000

print(f"\nHiperparametros do A2C:")
for key, value in A2C_HYPERPARAMS.items():
    print(f"  {key}: {value}")
print(f"\nTotal de timesteps: {TOTAL_TIMESTEPS_A2C:,}")
print("NOTA: A2C nao usa action masking - comparacao justa com PPO.\n")

print("Iniciando treinamento A2C...")
print("(Metricas serao exibidas a cada 5000 passos)\n")

# A2C usa ambiente SEM action masking wrapper
train_env_a2c = MinesweeperEnv(board_size=5, num_mines=3)
eval_env_a2c = MinesweeperEnv(board_size=5, num_mines=3)

a2c_callback = TrainingMetricsCallbackStandard(
    eval_env=eval_env_a2c,
    eval_freq=5000,
    n_eval_episodes=100,
)

start_time_a2c = time.time()

model_a2c = A2C(
    "CnnPolicy",
    train_env_a2c,
    verbose=0,
    **A2C_HYPERPARAMS
)

model_a2c.learn(total_timesteps=TOTAL_TIMESTEPS_A2C, callback=a2c_callback)

time_a2c = time.time() - start_time_a2c
print(f"\nTreinamento A2C concluido em {time_a2c:.1f} segundos ({time_a2c/60:.1f} min)")

model_a2c.save("models/a2c_minesweeper")
print("Modelo A2C salvo em models/a2c_minesweeper.zip")


# =============================================================================
# SECAO 6: TUNING DE HIPERPARAMETROS COM OPTUNA
# =============================================================================
# Ate agora, usamos hiperparametros "padrao". Mas sera que sao os MELHORES?
#
# O OPTUNA faz busca automatica dos melhores hiperparametros:
# 1. Escolhe uma combinacao aleatoria de hiperparametros
# 2. Treina o modelo com essa combinacao
# 3. Avalia o desempenho
# 4. Repete muitas vezes, ficando cada vez mais inteligente na busca
#
# POR QUE OPTUNA?
# - O professor mencionou explicitamente como diferencial
# - Mostra que nao estamos "chutando" hiperparametros
# - E muito mais eficiente que grid search (busca exaustiva)

print("\n" + "=" * 60)
print("  SECAO 6: TUNING COM OPTUNA")
print("=" * 60)

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial):
    """
    Funcao objetivo para o Optuna.
    O Optuna chama esta funcao varias vezes com diferentes hiperparametros.
    Nosso trabalho e treinar o modelo e retornar o desempenho (win rate).
    """
    # O Optuna SUGERE hiperparametros para testar
    # suggest_float = valor decimal entre min e max
    # suggest_categorical = escolhe de uma lista de opcoes
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [64, 128, 256, 512])
    gamma = trial.suggest_float("gamma", 0.9, 0.9999, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.001, 0.1, log=True)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
    n_epochs = trial.suggest_int("n_epochs", 3, 10)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    # Garante que batch_size <= n_steps (requisito do PPO)
    if batch_size > n_steps:
        batch_size = n_steps

    # Cria e treina o modelo com os hiperparametros sugeridos
    opt_env = make_env()
    try:
        model = MaskablePPO(
            "CnnPolicy",
            opt_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            ent_coef=ent_coef,
            clip_range=clip_range,
            n_epochs=n_epochs,
            batch_size=batch_size,
            verbose=0,
        )

        # Treina com menos timesteps para agilizar a busca
        model.learn(total_timesteps=50_000)

    except Exception as e:
        # Se der erro com alguma combinacao, retorna 0
        print(f"  Trial {trial.number} falhou: {e}")
        return 0.0

    # Avalia o modelo treinado
    eval_env = MinesweeperEnv(board_size=5, num_mines=3)
    wins = 0
    n_eval = 200

    for _ in range(n_eval):
        obs, info = eval_env.reset()
        done = False
        ep_reward = 0

        while not done:
            valid_actions = np.where(info["action_mask"] == 1)[0]
            if len(valid_actions) == 0:
                break
            action, _ = model.predict(obs, deterministic=True, action_masks=info["action_mask"])
            obs, reward, terminated, truncated, info = eval_env.step(int(action))
            done = terminated or truncated
            ep_reward += reward

        if ep_reward > 10:
            wins += 1

    win_rate = wins / n_eval * 100
    print(f"  Trial {trial.number}: win_rate={win_rate:.1f}% | lr={learning_rate:.6f}, gamma={gamma:.4f}, n_steps={n_steps}")

    return win_rate


# Roda a otimizacao
N_TRIALS = 20  # Numero de combinacoes a testar (aumente para resultados melhores)
print(f"\nIniciando busca com {N_TRIALS} trials...")
print("Cada trial treina um modelo com hiperparametros diferentes.\n")

study = optuna.create_study(direction="maximize")  # Queremos MAXIMIZAR win rate
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

print(f"\n--- Resultado da Otimizacao ---")
print(f"Melhor win rate encontrado: {study.best_value:.1f}%")
print(f"Melhores hiperparametros:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")


# --- Treina o modelo FINAL com os melhores hiperparametros ---
print("\n\nTreinando modelo FINAL com os melhores hiperparametros...")

best_params = study.best_params.copy()
# Garante que batch_size <= n_steps
if best_params.get("batch_size", 64) > best_params.get("n_steps", 256):
    best_params["batch_size"] = best_params["n_steps"]

train_env_best = make_env()
eval_env_best = MinesweeperEnv(board_size=5, num_mines=3)

best_callback = TrainingMetricsCallback(
    eval_env=eval_env_best,
    eval_freq=5000,
    n_eval_episodes=100,
)

start_time_best = time.time()

model_best = MaskablePPO(
    "CnnPolicy",
    train_env_best,
    verbose=0,
    **best_params
)

TOTAL_TIMESTEPS_BEST = 500_000
model_best.learn(total_timesteps=TOTAL_TIMESTEPS_BEST, callback=best_callback)

time_best = time.time() - start_time_best
print(f"\nTreinamento do modelo otimizado concluido em {time_best:.1f}s ({time_best/60:.1f} min)")

model_best.save("models/ppo_minesweeper_optimized")
print("Modelo otimizado salvo em models/ppo_minesweeper_optimized.zip")


# =============================================================================
# SECAO 7: AVALIACAO FINAL E COMPARACAO
# =============================================================================
# Hora de comparar TODOS os agentes lado a lado!
# Cada agente joga 1000 partidas e medimos o desempenho.

print("\n" + "=" * 60)
print("  SECAO 7: AVALIACAO FINAL")
print("=" * 60)


def evaluate_trained_agent(model, env, num_episodes=1000, use_mask=True):
    """
    Avalia um agente treinado jogando num_episodes partidas.
    Similar ao evaluate_random_agent, mas usa o modelo para escolher acoes.
    """
    wins = 0
    total_rewards = []
    cells_opened_list = []

    for _ in range(num_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0
        cells = 0

        while not done and cells < 100:  # Limite de 100 passos para evitar loops
            valid_actions = np.where(info["action_mask"] == 1)[0]
            if len(valid_actions) == 0:
                break

            if use_mask:
                action, _ = model.predict(obs, deterministic=True, action_masks=info["action_mask"])
            else:
                action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            ep_reward += reward
            cells += 1

        total_rewards.append(ep_reward)
        cells_opened_list.append(cells)
        if ep_reward > 10:
            wins += 1

    return {
        "win_rate": wins / num_episodes * 100,
        "avg_reward": np.mean(total_rewards),
        "std_reward": np.std(total_rewards),
        "avg_cells_opened": np.mean(cells_opened_list),
    }


print("\nAvaliando todos os agentes em 1000 partidas cada...\n")

eval_env = MinesweeperEnv(board_size=5, num_mines=3)

# Avalia cada agente
print("Avaliando agente aleatorio...")
random_final = evaluate_random_agent(eval_env, 1000)

print("Avaliando PPO...")
ppo_final = evaluate_trained_agent(model_ppo, eval_env, 1000)

print("Avaliando A2C...")
a2c_final = evaluate_trained_agent(model_a2c, eval_env, 1000, use_mask=False)

print("Avaliando PPO Otimizado (Optuna)...")
best_final = evaluate_trained_agent(model_best, eval_env, 1000)

# --- Tabela Comparativa ---
print("\n" + "=" * 75)
print("  TABELA COMPARATIVA FINAL")
print("=" * 75)
print(f"{'Agente':<25} {'Win Rate':>10} {'Reward Medio':>15} {'Celulas Abertas':>16}")
print("-" * 75)
print(f"{'Aleatorio (Baseline)':<25} {random_final['win_rate']:>9.1f}% {random_final['avg_reward']:>+14.2f} {random_final['avg_cells_opened']:>15.1f}")
print(f"{'PPO (padrao)':<25} {ppo_final['win_rate']:>9.1f}% {ppo_final['avg_reward']:>+14.2f} {ppo_final['avg_cells_opened']:>15.1f}")
print(f"{'A2C':<25} {a2c_final['win_rate']:>9.1f}% {a2c_final['avg_reward']:>+14.2f} {a2c_final['avg_cells_opened']:>15.1f}")
print(f"{'PPO Otimizado (Optuna)':<25} {best_final['win_rate']:>9.1f}% {best_final['avg_reward']:>+14.2f} {best_final['avg_cells_opened']:>15.1f}")
print("=" * 75)


# =============================================================================
# SECAO 8: GRAFICOS E VISUALIZACOES
# =============================================================================
# Graficos sao essenciais para demonstrar aprendizado visualmente.

print("\n" + "=" * 60)
print("  SECAO 8: GRAFICOS")
print("=" * 60)

os.makedirs("graficos", exist_ok=True)

# --- Grafico 1: Comparacao de Win Rate ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

agentes = ['Aleatorio', 'PPO', 'A2C', 'PPO\nOtimizado']
win_rates = [random_final['win_rate'], ppo_final['win_rate'],
             a2c_final['win_rate'], best_final['win_rate']]
avg_rewards = [random_final['avg_reward'], ppo_final['avg_reward'],
               a2c_final['avg_reward'], best_final['avg_reward']]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

bars1 = axes[0].bar(agentes, win_rates, color=colors, edgecolor='black', width=0.6)
axes[0].set_title('Taxa de Vitoria por Agente (%)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Win Rate (%)', fontsize=12)
axes[0].set_ylim(0, max(win_rates) * 1.3 + 5)
for bar, val in zip(bars1, win_rates):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

bars2 = axes[1].bar(agentes, avg_rewards, color=colors, edgecolor='black', width=0.6)
axes[1].set_title('Recompensa Media por Agente', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Recompensa Media', fontsize=12)
for bar, val in zip(bars2, avg_rewards):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + (0.5 if val >= 0 else -1.5),
                 f'{val:+.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.suptitle('Comparacao: Agentes de RL no Campo Minado (5x5, 3 minas)',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('graficos/comparacao_agentes.png', dpi=150, bbox_inches='tight')
print("Grafico salvo: graficos/comparacao_agentes.png")
plt.close()


# --- Grafico 2: Curvas de Aprendizado ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# PPO
if ppo_callback.timesteps_history:
    ppo_win_rates = [r["win_rate"] for r in ppo_callback.results_history]
    ppo_rewards = [r["avg_reward"] for r in ppo_callback.results_history]
    axes[0].plot(ppo_callback.timesteps_history, ppo_win_rates, 'o-',
                 color='#4ECDC4', label='PPO', linewidth=2, markersize=4)
    axes[1].plot(ppo_callback.timesteps_history, ppo_rewards, 'o-',
                 color='#4ECDC4', label='PPO', linewidth=2, markersize=4)

# A2C
if a2c_callback.timesteps_history:
    a2c_win_rates = [r["win_rate"] for r in a2c_callback.results_history]
    a2c_rewards = [r["avg_reward"] for r in a2c_callback.results_history]
    axes[0].plot(a2c_callback.timesteps_history, a2c_win_rates, 's-',
                 color='#45B7D1', label='A2C', linewidth=2, markersize=4)
    axes[1].plot(a2c_callback.timesteps_history, a2c_rewards, 's-',
                 color='#45B7D1', label='A2C', linewidth=2, markersize=4)

# PPO Otimizado
if best_callback.timesteps_history:
    best_win_rates = [r["win_rate"] for r in best_callback.results_history]
    best_rewards = [r["avg_reward"] for r in best_callback.results_history]
    axes[0].plot(best_callback.timesteps_history, best_win_rates, 'D-',
                 color='#96CEB4', label='PPO Otimizado', linewidth=2, markersize=4)
    axes[1].plot(best_callback.timesteps_history, best_rewards, 'D-',
                 color='#96CEB4', label='PPO Otimizado', linewidth=2, markersize=4)

# Baseline aleatorio (linha horizontal)
axes[0].axhline(y=random_final['win_rate'], color='#FF6B6B', linestyle='--',
                label=f'Aleatorio ({random_final["win_rate"]:.1f}%)', alpha=0.7)
axes[1].axhline(y=random_final['avg_reward'], color='#FF6B6B', linestyle='--',
                label=f'Aleatorio ({random_final["avg_reward"]:.1f})', alpha=0.7)

axes[0].set_title('Curva de Aprendizado: Win Rate', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Timesteps', fontsize=12)
axes[0].set_ylabel('Win Rate (%)', fontsize=12)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

axes[1].set_title('Curva de Aprendizado: Recompensa', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Timesteps', fontsize=12)
axes[1].set_ylabel('Recompensa Media', fontsize=12)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.suptitle('Evolucao do Aprendizado ao Longo do Treinamento',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('graficos/curvas_aprendizado.png', dpi=150, bbox_inches='tight')
print("Grafico salvo: graficos/curvas_aprendizado.png")
plt.close()


# --- Grafico 3: Optuna - Importancia dos Hiperparametros ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Historico de trials
trial_numbers = [t.number for t in study.trials]
trial_values = [t.value if t.value is not None else 0 for t in study.trials]
axes[0].bar(trial_numbers, trial_values, color='#4ECDC4', edgecolor='black', alpha=0.8)
axes[0].axhline(y=study.best_value, color='red', linestyle='--',
                label=f'Melhor: {study.best_value:.1f}%')
axes[0].set_title('Optuna: Win Rate por Trial', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Numero do Trial', fontsize=12)
axes[0].set_ylabel('Win Rate (%)', fontsize=12)
axes[0].legend(fontsize=10)

# Importancia dos hiperparametros
try:
    importances = optuna.importance.get_param_importances(study)
    params = list(importances.keys())
    values = list(importances.values())
    axes[1].barh(params, values, color='#96CEB4', edgecolor='black')
    axes[1].set_title('Importancia dos Hiperparametros', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Importancia', fontsize=12)
except Exception:
    axes[1].text(0.5, 0.5, 'Importancia nao disponivel\n(poucos trials)',
                 ha='center', va='center', fontsize=14, transform=axes[1].transAxes)
    axes[1].set_title('Importancia dos Hiperparametros', fontsize=14, fontweight='bold')

plt.suptitle('Analise da Otimizacao de Hiperparametros (Optuna)',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('graficos/optuna_analise.png', dpi=150, bbox_inches='tight')
print("Grafico salvo: graficos/optuna_analise.png")
plt.close()


# =============================================================================
# SECAO 9: DEMONSTRACAO VISUAL DE PARTIDAS
# =============================================================================
# Vamos assistir o melhor agente jogar para entender sua estrategia.

print("\n" + "=" * 60)
print("  SECAO 9: DEMONSTRACAO DE PARTIDAS")
print("=" * 60)

demo_env = MinesweeperEnv(board_size=5, num_mines=3, render_mode="human")

for game_num in range(3):
    print(f"\n{'='*40}")
    print(f"  PARTIDA {game_num + 1} (Agente PPO Otimizado)")
    print(f"{'='*40}")

    obs, info = demo_env.reset()
    done = False
    total_reward = 0
    step = 0

    print("\nTabuleiro inicial:")
    demo_env.render()

    while not done:
        action, _ = model_best.predict(obs, deterministic=True, action_masks=info["action_mask"])
        action = int(action)
        x, y = action // demo_env.board_size, action % demo_env.board_size

        obs, reward, terminated, truncated, info = demo_env.step(action)
        done = terminated or truncated
        total_reward += reward
        step += 1

        print(f"\nPasso {step}: clicou em ({x}, {y}) -> recompensa: {reward:+.1f}")
        demo_env.render()

    resultado = "VITORIA!" if total_reward > 10 else "DERROTA!"
    print(f"\n-> {resultado} Recompensa total: {total_reward:.1f}")

demo_env.close()


# =============================================================================
# SECAO 10: CONCLUSOES
# =============================================================================
print("\n" + "=" * 60)
print("  CONCLUSOES")
print("=" * 60)

print(f"""
RESULTADOS PRINCIPAIS:
---------------------
1. BASELINE (Aleatorio):  {random_final['win_rate']:.1f}% de vitorias
2. PPO (padrao):          {ppo_final['win_rate']:.1f}% de vitorias
3. A2C:                   {a2c_final['win_rate']:.1f}% de vitorias
4. PPO Otimizado (Optuna): {best_final['win_rate']:.1f}% de vitorias

MELHORIA SOBRE O BASELINE:
- PPO padrao:    {ppo_final['win_rate'] - random_final['win_rate']:+.1f} pontos percentuais
- A2C:           {a2c_final['win_rate'] - random_final['win_rate']:+.1f} pontos percentuais
- PPO Otimizado: {best_final['win_rate'] - random_final['win_rate']:+.1f} pontos percentuais

TEMPO DE TREINAMENTO:
- PPO: {time_ppo:.1f}s | A2C: {time_a2c:.1f}s | PPO Otimizado: {time_best:.1f}s

MELHORES HIPERPARAMETROS (Optuna):""")

for key, value in study.best_params.items():
    print(f"  {key}: {value}")

print(f"""
ANALISE:
--------
O Campo Minado e um jogo de informacao incompleta — o agente nao sabe
onde estao as minas ate pisar nelas. Isso torna o problema MUITO desafiador
para RL, pois ha um componente aleatorio inevitavel.

O agente aprendeu a:
- Preferir celulas no centro/bordas com base nos numeros revelados
- Evitar areas com numeros altos (indicam muitas minas vizinhas)
- Usar cascatas (celulas com 0 vizinhas) para revelar grandes areas

LIMITACOES:
- O Campo Minado tem aleatoriedade inerente (minas sao randomicas)
- Mesmo um jogador perfeito nao pode vencer 100% das vezes
- Tabuleiros maiores exigiriam redes neurais mais complexas (CNN)

TRABALHOS FUTUROS:
- Testar com tabuleiros maiores (8x8, 10x10)
- Usar CNN em vez de MLP para capturar padroes espaciais
- Implementar DQN para comparar com policy gradient
- Usar curriculum learning (comecar facil, aumentar dificuldade)
""")

print("=" * 60)
print("  FIM DO PROJETO")
print("=" * 60)
print("\nArquivos gerados:")
print("  models/ppo_minesweeper.zip")
print("  models/a2c_minesweeper.zip")
print("  models/ppo_minesweeper_optimized.zip")
print("  graficos/comparacao_agentes.png")
print("  graficos/curvas_aprendizado.png")
print("  graficos/optuna_analise.png")
