SYSTEM PROMPT - DIRETRIZES TECNICAS

Objetivo
Definir as diretrizes tecnicas para a construcao da aplicacao, garantindo padrao de arquitetura e stack prioritaria.

Stack principal
- Linguagem: Python
- Arquitetura: Pipeline + modulos (clean light)
- Aplicacao desktop (CLI)
- Unico ponto de entrada para a aplicacao

Bibliotecas priorizadas
- cv2 (OpenCV): leitura de video, extracao de frames e anotacoes visuais
- face_recognition: deteccao e reconhecimento facial
- deepface: analise de expressoes emocionais
- mediapipe: suporte a deteccao de pontos faciais e/ou atividades
- os: manipulacao de arquivos e paths
- speech_recognition: transcricao e processamento de audio/texto
- moviepy: manipulacao de video e audio
- tqdm: barra de progresso no processamento

Diretrizes de arquitetura (Pipeline + modulos)
- Pipeline: orquestra o fluxo ponta a ponta
- Modulos: unidades isoladas para deteccao e analises
- Saidas: geracao de artefatos finais (video anotado e resumo)
- Metadados por frame apenas quando solicitado

Padroes de implementacao
- Separar pipeline em etapas claras
- Evitar logica de negocio diretamente na View
- Manter scripts de execucao simples e objetivos
- Usar apenas paradigma funcional (sem orientacao a objetos)

Boas praticas de codigo
- Seguir Clean Code e SOLID
- Nao incluir comentarios no codigo
- Escrever todo o codigo em ingles
- Evitar gerar arquivos desnecessarios no repositorio (ex.: `__pycache__`)
- Para execucao local, use `PYTHONDONTWRITEBYTECODE=1` para evitar `__pycache__`

Diretriz de comunicacao
- Todas as respostas devem ser em portugues
