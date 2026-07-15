# Campo Minado com Aprendizado por Reforco

Projeto Final - Inteligencia Computacional (7o Bloco)

## Objetivo

Treinar um agente autonomo via Reinforcement Learning para jogar Campo Minado,
comparando algoritmos (PPO vs A2C) e otimizando hiperparametros com Optuna.

## Estrutura do Projeto

```
Campo minado/
├── minesweeper_env.py      # Ambiente Gymnasium do Campo Minado
├── campo_minado_rl.py      # Script principal (treinamento + avaliacao)
├── requirements.txt        # Dependencias
├── README.md               # Este arquivo
├── models/                 # Modelos salvos (gerado apos treinamento)
└── graficos/               # Graficos gerados (gerado apos treinamento)
```

## Como Executar

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Testar o ambiente

```bash
python minesweeper_env.py
```

### 3. Rodar o treinamento completo

```bash
python campo_minado_rl.py
```

O script executa automaticamente:
1. Avaliacao do baseline aleatorio
2. Treinamento com PPO
3. Treinamento com A2C
4. Tuning de hiperparametros com Optuna
5. Avaliacao comparativa final
6. Geracao de graficos

## Tecnologias

- **Gymnasium**: Modelagem do ambiente
- **Stable-Baselines3 + sb3-contrib**: Implementacao dos algoritmos PPO e A2C
- **Optuna**: Otimizacao de hiperparametros
- **Matplotlib**: Visualizacoes e graficos

## Abordagem

### Ambiente
- Tabuleiro 5x5 com 3 minas
- Observacao normalizada [0, 1]
- Action masking para acoes validas
- Reward shaping progressivo

### Agentes Comparados
- **Aleatorio** (baseline): escolhe acoes validas aleatoriamente
- **PPO** (Proximal Policy Optimization): com hiperparametros padrao
- **A2C** (Advantage Actor-Critic): comparacao com outro algoritmo
- **PPO Otimizado**: com hiperparametros encontrados pelo Optuna
