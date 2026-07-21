# PYNQ-Z2 — Acelerador FINN para classificação SAT-6

Esta pasta reúne os artefatos da configuração final **t2w8_500fps** executada
fisicamente na PYNQ-Z2, além dos resultados preservados do treinamento de
50 épocas, validação em Brevitas/QONNX e relatórios do fluxo FINN/Vivado.

## Resultado físico final

| Métrica | Valor |
|---|---:|
| Amostras | 1.000 |
| Acertos | 935 |
| Acurácia | 93,50% |
| Tempo total | 2,258213 s |
| Latência média | 2,258213 ms/imagem |
| Vazão | 442,827969 imagens/s |
| Potência média | 1,58 W |
| Energia estimada | 3,567977 mJ/imagem |

## Organização

```text
pynq-z2/
├── README.md
├── requirements-pynq.txt
├── THIRD_PARTY_NOTICES.md
├── SHA256SUMS.txt
├── data/
│   └── sat6_test_1000_uint8.npz
├── deploy/
│   ├── bitfile/
│   │   ├── finn-accel.bit
│   │   └── finn-accel.hwh
│   └── driver/
├── docs/
│   └── FLUXO_FINN.md
├── scripts/
│   └── validar_sat6_pynq.py
└── results/
    ├── treinamento-validacao-50ep/
    ├── sintese-finn/
    └── validacao-fisica/
```

## Fluxo de desenvolvimento

O fluxo foi baseado no projeto `ArthurEly/finn`, pasta
`notebooks/sat6_cnn`, e adaptado para o estudo apresentado no TCC. A sequência
completa está documentada em [`docs/FLUXO_FINN.md`](docs/FLUXO_FINN.md).

## Entrada e saída da implantação final

```text
Entrada:  (N, 32, 32, 4), UINT8
Saída:    (N, 1), UINT8
```

A saída contém diretamente o índice da classe prevista.

## Execução na placa

O script foi escrito para o ambiente Python 3.6 utilizado com a imagem PYNQ da
placa.

```bash
python3 scripts/validar_sat6_pynq.py
```

Teste rápido:

```bash
python3 scripts/validar_sat6_pynq.py --limit 10
```

Os resultados da execução são gravados em:

```text
results/validacao-fisica/nova-execucao/
```

## Artefatos principais

- `deploy/bitfile/finn-accel.bit`: bitstream t2w8 implantado;
- `deploy/bitfile/finn-accel.hwh`: descrição de hardware da implantação;
- `deploy/driver/`: driver FINN/PYNQ e dependências locais;
- `data/sat6_test_1000_uint8.npz`: conjunto físico de validação;
- `results/treinamento-validacao-50ep/`: histórico, métricas e matrizes;
- `results/sintese-finn/`: estimativas, RTLSIM e relatórios de recursos;
- `results/validacao-fisica/`: resultado final de 93,50%.

## Ferramentas utilizadas

- PyTorch e Brevitas;
- QONNX;
- FINN;
- Vivado/Vitis 2023.2;
- PYNQ-Z2.
