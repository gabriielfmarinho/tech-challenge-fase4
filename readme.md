Tech Challenge - Fase 4 (IADT)

Descricao breve
CLI para analise de video com reconhecimento facial, analise de emocoes,
detecao de atividades e geracao de resumo automatico.

Arquitetura e linguagem
- Linguagem: Python
- Arquitetura: Pipeline + modulos (clean light)
- Aplicacao nao web

Estrutura do projeto
- `readme.md`: visao geral do projeto
- `prompts/system/system.md`: prompt do sistema e diretrizes tecnicas
- `src/`: codigo fonte da aplicacao
  - `src/pipeline`: orquestracao do fluxo
  - `src/modules`: modulos de analise
  - `src/config`: configuracoes do projeto
  - `src/utils`: utilitarios compartilhados

Como rodar localmente
1) Instale o pacote do virtualenv:
   - Ubuntu/Debian: `sudo apt install -y python3.12-venv`
2) Crie e ative o ambiente virtual:
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
3) Instale as dependencias:
   - `pip install -r requirements.txt`
4) Execute a aplicacao pelo entrypoint do projeto.

Como usar (CLI)
Execucao padrao (processa todo o video):
```bash
PYTHONPATH=src python src/main.py --input "prompts/examples/Unlocking Facial Recognition_ Diverse Activities Analysis.mp4"
```

Execucao rapida (amostragem):
```bash
PYTHONPATH=src python src/main.py --input "prompts/examples/Unlocking Facial Recognition_ Diverse Activities Analysis.mp4" --max-frames 30 --frame-step 3 --resize-width 640
```

Execucao com melhor qualidade (mais lenta):
```bash
PYTHONPATH=src python src/main.py --input "prompts/examples/Unlocking Facial Recognition_ Diverse Activities Analysis.mp4" --face-model cnn --upsample 2 --min-face-size 50 --face-padding 0.2 
```

Evitar `__pycache__`
Para nao gerar pastas `__pycache__`, use:
```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python src/main.py --input "prompts/examples/Unlocking Facial Recognition_ Diverse Activities Analysis.mp4"
```

Parametros principais
- `--input`: caminho do video de entrada
- `--output-dir`: pasta de saida (padrao: `outputs/analysis`)
- `--output-video`: nome do video anotado (padrao: `annotated.mp4`)
- `--metadata-file`: nome do metadata (padrao: `metadata.jsonl`)
- `--frame-step`: processa a cada N frames (amostragem)
- `--max-frames`: limita numero de frames processados
- `--resize-width`: redimensiona o frame para acelerar
- `--face-model`: `hog` ou `cnn` (cnn e mais lento e sensivel)
- `--upsample`: aumenta a deteccao de faces em resolucoes baixas
- `--face-fallback`: `haar` ou `none` (fallback quando nao acha rostos)
- `--haar-scale`: fator de escala do Haar Cascade
- `--haar-neighbors`: numero de vizinhos do Haar Cascade
- `--min-face-size`: tamanho minimo da face (filtro de falsos positivos)
- `--face-padding`: padding no recorte da face para emocao
- `--full-metadata`: grava registros por frame no `metadata.jsonl`

O que a aplicacao faz
- Detecta rostos e desenha caixas no video
- Analisa emocao dominante por rosto
- Detecta atividade por frame
- Gera resumo automatico com contagens de emocoes, atividades e anomalias

Saidas geradas
- Video anotado: `outputs/analysis/annotated.mp4`
- Metadados: `outputs/analysis/metadata.jsonl`
 
Observacao sobre `__pycache__`
- O Python pode criar pastas `__pycache__` automaticamente durante a execucao.
- Recomenda-se adicionar `__pycache__/` ao `.gitignore`.

Formato do `metadata.jsonl`
- Quando `--full-metadata` nao e usado:
  - Contem apenas um registro final com o resumo
- Quando `--full-metadata` e usado:
  - Registros por frame com:
    - `frame_index`, `timestamp`
    - `face_count`, `boxes`
    - `emotions` (em portugues)
    - `activity`
    - `motion_score`, `is_anomaly`
  - Ultima linha contem `summary` com:
    - `frames_processed`, `faces_detected`, `anomalies_detected`
    - `activities`, `emotions`
    - `top_activities`, `top_emotions`
