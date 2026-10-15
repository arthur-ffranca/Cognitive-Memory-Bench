# Cognitive Memory Benchmark 🧠

Repositorio **independente** para comparar estrategias de memoria de longo prazo em agentes de IA. Inspirado no [FadeMem](https://arxiv.org/pdf/2601.18642v2). Esta **nao e a implementacao oficial** do artigo.

## Situacao atual

- **Existente e executavel:** `cmb.py` (simulacao offline e deterministica), `locomo_retrieval.py` (adaptador de recuperacao de evidencias), e `tests/test_benchmark.py`.
- **Organizado, ainda nao implementado:** `src/` e `benchmarks/`, que receberao o RAG real, FadeMem, chamada aos dois LLMs e avaliador. Estes arquivos sao **placeholders**, nao executam o benchmark completo.
- **Configuracao pronta:** `docker-compose.yml` para iniciar Qdrant local; `.env.example` sem credenciais.
- **Nao incluso:** dataset LoCoMo, chaves de API e resultados de execucoes.

## Arvore

```text
cognitive-memory-benchmark/
├── .env.example               # modelos, URL local do Qdrant (sem chave real)
├── .gitignore
├── docker-compose.yml         # Qdrant em Docker local
├── requirements.txt           # dependencias futuras
├── README.md
├── cmb.py                     # prototipo offline EXISTENTE
├── locomo_retrieval.py        # adaptador LoCoMo EXISTENTE
├── src/
│   ├── __init__.py
│   ├── fademem.py             # placeholder
│   ├── rag.py                 # placeholder
│   ├── embeddings.py          # placeholder
│   └── llm.py                 # placeholder
├── benchmarks/
│   ├── __init__.py
│   ├── run.py                 # placeholder
│   ├── evaluator.py           # placeholder
│   └── metrics.py             # placeholder
├── tests/
│   └── test_benchmark.py      # testes do prototipo offline
├── data/locomo/              # colocar dataset aqui no futuro
└── results/                  # saidas locais (ignoradas no Git)
```

## Testar o que ja existe

Requer Python 3.10+; sem dependencias adicionais:

```bash
python cmb.py --profiles 24 --seeds 3 --k 8 --fifo 250
python -m unittest discover -s tests -v
```

Para iniciar apenas o Qdrant local (Docker instalado):

```bash
docker compose up -d
```

Dashboard local: http://localhost:6333/dashboard

> A imagem do Qdrant esta configurada como `latest` para facilitar a estrutura inicial. Antes do benchmark reproduzivel, fixar uma versao explicita.

## Plano de experimentacao (nao implementado)

Comparar **Full Context**, **RAG tradicional** e **FadeMem + RAG**, cada um com **GPT-5 Mini** e **GPT-5.4 Mini** via API, resultando em seis combinacoes. Infraestrutura, dataset, embeddings e avaliacao no Dell, com **a inferencia dos GPTs realizada externamente pela API**. Sem Ollama.

A comparacao devera medir qualidade de respostas, recuperacao de evidencias, retencao, tokens, custo e latencia. Tambem devera avaliar ablacoes dos mecanismos do FadeMem.

Dataset: https://github.com/snap-research/LoCoMo

## Seguranca e versionamento

Copiar `.env.example` para `.env` localmente. Nunca versionar a chave de API. O `.gitignore` exclui o dataset, arquivos gerados em `results/`, caches e segredos.
