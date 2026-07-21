# NVIDIA Jetson Orin Nano — Inferência SAT-6 com GPU embarcada

Implementação da classificação de imagens multiespectrais do dataset **SAT-6** na NVIDIA Jetson Orin Nano, utilizando TensorFlow/Keras e a API C do TensorFlow.

## Resultados principais

| Métrica | Valor |
|---|---:|
| Acurácia | **99,34%** |
| Imagens classificadas corretamente | 9.934 de 10.000 |
| Melhor configuração adotada | modo 15 W, batch 32 |
| Latência média | 0,665477 ms/imagem |
| Vazão | aproximadamente 1.502,68 imagens/s |
| Potência média do sistema | 6,346327 W |
| Energia estimada por imagem | aproximadamente 4,223 mJ/imagem |
| Eficiência energética | aproximadamente 236,78 imagens/J |

> A potência foi obtida pelo campo `VDD_IN` do `tegrastats`. > A potência média foi obtida a partir do campo `VDD_IN` registrado pelo `tegrastats`. A energia por imagem foi estimada pela multiplicação da potência média pela latência média de inferência.

## Organização dos arquivos

```text
jetson-orin-nano/
├── README.md
├── src/
│   └── sat6_jetson_tf_c.c
├── scripts/
│   ├── export_sat6.py
│   └── gerar_graficos_benchmark_sat6.py
├── models/
│   ├── sat6_tf_28_best.keras
│   └── sat6_savedmodel_for_c/
├── data/
│   └── y_test_sat6_tf_labels.csv
├── results/
│   ├── execution_log_sat6_jetson.csv
│   ├── metadata.json
│   ├── resumo_resultados_sat6_jetson.csv
│   ├── training_history.csv
│   ├── logs_7w/
│   ├── logs_15w/
│   └── graficos/
└── docs/
    └── resultados_sat6_7w_15w.png
```

## Fluxo da implementação

1. O modelo CNN foi treinado com TensorFlow/Keras para entradas `28 × 28 × 4` e seis classes.
2. O modelo final foi salvo em formato `.keras` e exportado como TensorFlow SavedModel.
3. A aplicação `sat6_jetson_tf_c.c` carregou o SavedModel pela API C do TensorFlow.
4. Os testes foram executados com batches 1, 32, 256 e 1024 nos modos de potência 7 W e 15 W.
5. O `tegrastats` foi executado em paralelo para registrar potência e uso da GPU.
6. O script `gerar_graficos_benchmark_sat6.py` extraiu as métricas dos logs e gerou os gráficos finais.

## Compilação do código C

A aplicação depende da API C do TensorFlow instalada na Jetson.

```bash
gcc src/sat6_jetson_tf_c.c -ltensorflow -o model_complete_sat6
```

Exemplo de execução a partir da raiz desta pasta:

```bash
./model_complete_sat6   32   10000   models/sat6_savedmodel_for_c/   X_test_sat6_tf_hwc.csv   data/y_test_sat6_tf_labels.csv   serve_input_sat6   StatefulPartitionedCall   none   0
```

## Geração do CSV de entrada

O arquivo `X_test_sat6_tf_hwc.csv` não foi incluído porque possui aproximadamente 331 MB e pode ser regenerado a partir do dataset SAT-6 por meio de `scripts/export_sat6.py`.

O dataset completo também não é armazenado neste repositório devido ao tamanho.

## Dependências principais

- NVIDIA Jetson Orin Nano;
- JetPack e suporte à GPU;
- TensorFlow 2.16.1, conforme o ambiente registrado;
- API C do TensorFlow para compilação da aplicação em C;
- Python com NumPy, pandas e Matplotlib;
- `tegrastats` para coleta dos dados de potência e utilização da GPU.

