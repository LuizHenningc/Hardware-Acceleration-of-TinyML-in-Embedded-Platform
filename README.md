# Análise de TinyML com Aceleração em Hardware em Sistemas Embarcados

Estudo comparativo de soluções de TinyML aplicadas ao sensoriamento remoto, avaliando três plataformas embarcadas na classificação de imagens multiespectrais do dataset **SAT-6**.

Trabalho de Conclusão de Curso de Engenharia de Computação da Universidade do Vale do Itajaí — UNIVALI.

## Contexto

O projeto compara três abordagens de hardware para inferência embarcada:

- GPU embarcada;
- microcontrolador com acelerador neural dedicado;
- SoC-FPGA com acelerador gerado pelo FINN.

As três plataformas executam o mesmo problema de classificação do dataset SAT-6, porém utilizam modelos, ferramentas e fluxos de implantação adaptados às características de cada hardware.

A avaliação considera acurácia, latência, vazão, consumo energético e complexidade de implementação.

## Resultados principais

| Plataforma | Tipo | Acurácia | Latência média | Vazão | Potência média |
|---|---|---:|---:|---:|---:|
| **NVIDIA Jetson Orin Nano** | GPU embarcada | 99,34% | 0,665477 ms/imagem | 1.502,68 imagens/s | 6,346327 W |
| **MAX78000 — sistema HIL** | MCU com acelerador CNN | 94,70% | 342,808 ms/imagem | 2,917 imagens/s | aproximadamente 0,060 W |
| **MAX78000 — processamento embarcado** | MCU com acelerador CNN | 94,70% | 0,611 ms/imagem | 1.636,650 imagens/s | aproximadamente 0,060 W |
| **PYNQ-Z2 com FINN** | SoC-FPGA | 93,50% | 2,258213 ms/imagem | 442,83 imagens/s | 1,58 W |

> Os métodos de medição não são totalmente equivalentes entre as plataformas. Na MAX78000, o sistema HIL completo inclui a preparação no computador, comunicação UART, execução na placa e leitura da resposta. O processamento embarcado representa apenas as etapas executadas no firmware após o recebimento dos dados.

## Estrutura do repositório

```text
Hardware-Acceleration-of-TinyML-in-Embedded-Platform/
├── README.md
├── LICENSE
├── max78000/
│   └── README.md
├── jetson-orin-nano/
│   └── README.md
├── pynq-z2/
│   └── README.md
└── docs/
    └── TCC_Analise_TinyML_Aceleracao_Hardware.pdf