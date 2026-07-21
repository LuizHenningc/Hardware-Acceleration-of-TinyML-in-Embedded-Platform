# MAX78000 — Inferência TinyML com acelerador CNN dedicado

Implementação da classificação de imagens multiespectrais do dataset **SAT-6** em um microcontrolador de ultra baixo consumo com acelerador CNN integrado.

## Resultado

| Métrica | Valor |
|---|---:|
| Acurácia HIL | **94,70%** |
| Imagens classificadas corretamente | 947 de 1.000 |
| Latência do processamento embarcado sem UART | 0,611 ms/imagem |
| Vazão do processamento embarcado | 1.636,650 imagens/s |
| Latência média do sistema HIL completo | 342,808 ms/imagem |
| Vazão do sistema HIL completo | 2,917 imagens/s |
| Potência medida em inferência contínua | aproximadamente 60 mW |
| Energia estimada por imagem no processamento embarcado | 0,03666 mJ/imagem |

## Sobre a implementação

Diferente da Jetson (GPU embarcada em Linux) e da PYNQ-Z2 (lógica programável em FPGA), a MAX78000 executa a inferência por meio de um acelerador neural dedicado, integrado ao próprio microcontrolador. O fluxo aqui envolveu: conversão do modelo quantizado para firmware em C, compilação, gravação na placa e validação via **hardware-in-the-loop (HIL)**.

**Fluxo:**
1. **Conversão do modelo** — o modelo quantizado foi convertido para arquivos em C que representam e controlam a rede no acelerador, incluindo o `cnn.h` gerado automaticamente para a rede `sat6`:

```c
int cnn_enable(uint32_t clock_source, uint32_t clock_divider);
int cnn_init(void);
int cnn_load_weights(void);
int cnn_load_bias(void);
int cnn_configure(void);
int cnn_start(void);
int cnn_unload(uint32_t *out_buf);
```

2. **Geração e gravação do firmware** — firmware em C responsável por inicializar a placa, configurar o acelerador, organizar os dados na memória, acionar a inferência e disponibilizar as saídas. Controle rigoroso de rastreabilidade entre pesos, artefatos compilados e binário gravado (importante já que múltiplos ciclos de treino/conversão/compilação geram muitos arquivos intermediários).
3. **Validação hardware-in-the-loop** — a placa aguarda o envio de uma amostra via UART pelo computador hospedeiro, executa a inferência no acelerador CNN e retorna a classe prevista:

```c
#define IMG_BYTES (28 * 28 * 4)
#define SAT6_NUM_CLASSES 6

cnn_enable(...);
cnn_init();
cnn_load_weights();
cnn_load_bias();
cnn_configure();

while (1) {
    if (uart_getc() != 'S') continue;
    uart_putc('K');
    uart_read_bytes((uint8_t*) image_buf32, IMG_BYTES);
    load_cnn_input();
    cnn_start();
    while (cnn_time == 0) { __WFI(); }
    cnn_unload(cnn_out_words);
    // ... argmax e envio do resultado pela serial
}
```

4. **Registro das predições** — rótulo real, classe prevista e valores de saída da rede foram armazenados no script do computador hospedeiro, para cálculo posterior de acurácia, matriz de confusão e acurácia por classe.
5. **Medição de energia** — modo de inferência contínua para medição de consumo, com energia por imagem estimada a partir da potência e da latência.

## Organização dos arquivos

```text
max78000/
├── ai8x-training/
│   ├── sat6.py
│   ├── sat6_net.py
│   └── train.py
│
├── ai8x-synthesis/
│   ├── sat6.yaml
│   ├── cnn.c
│   ├── cnn.h
│   ├── main.c
│   ├── sampledata.h
│   ├── sampleoutput.h
│   ├── softmax.c
│   └── weights.h
│
├── comparar_csv_pc.py
├── comparar_pc_vs_placa.py
├── converter_imagem.py
├── disparador_hil.py
├── disparador_hil_matriz.py
├── gerar_csv_sat6_4c.py
├── gerar_graficos_treino_max78000.py
├── gerar_validacao_max78000.py
├── Pre_tcc3.py
├── test_image.h
└── teste_sanidade.py
```

### Arquivos de treinamento

A pasta `ai8x-training` contém os arquivos específicos do SAT-6 utilizados no fluxo de treinamento:

- `sat6.py`: carregamento e preparação do dataset;
- `sat6_net.py`: definição da arquitetura da rede neural;
- `train.py`: script utilizado no processo de treinamento.

### Arquivos de síntese e firmware

A pasta `ai8x-synthesis` contém a configuração da rede e os arquivos utilizados ou gerados para execução na MAX78000:

- `sat6.yaml`: configuração da rede para o processo de síntese;
- `cnn.c` e `cnn.h`: implementação e interface da CNN geradas para o acelerador;
- `weights.h`: pesos quantizados da rede;
- `main.c`: firmware principal utilizado na placa;
- `sampledata.h` e `sampleoutput.h`: dados e saídas de referência;
- `softmax.c`: função auxiliar para tratamento das saídas.

### Scripts no computador hospedeiro

Os scripts Python localizados na raiz da pasta foram utilizados para:

- preparar e converter as imagens do SAT-6;
- gerar arquivos de entrada compatíveis com a MAX78000;
- realizar testes de sanidade;
- enviar amostras para a placa pela comunicação serial;
- receber as classes previstas;
- executar a validação hardware-in-the-loop;
- gerar matrizes de confusão, métricas e gráficos;
- comparar as predições do computador com as predições da placa.

## Dependências externas

Esta pasta não contém cópias completas das ferramentas da Analog Devices. Para reproduzir o processo de treinamento, síntese, compilação e gravação do firmware, são necessários:

- `ai8x-training`;
- `ai8x-synthesis`;
- Maxim Microcontrollers SDK, também chamado de MSDK;
- Python com as bibliotecas utilizadas pelos scripts de preparação e validação;
- placa MAX78000FTHR;
- conexão serial UART configurada para a porta utilizada no computador.

Foram incluídos neste repositório somente os arquivos específicos do projeto SAT-6 e os arquivos gerados ou adaptados para a implementação realizada neste trabalho.

## Observações para reprodução

As portas seriais configuradas nos scripts, como `COM8`, devem ser alteradas de acordo com o computador utilizado.

Os arquivos completos do dataset SAT-6 não são armazenados neste repositório devido ao tamanho. Os scripts esperam que os arquivos de dados e rótulos sejam disponibilizados localmente nos caminhos configurados.

Os comandos e parâmetros exatos utilizados no treinamento e na síntese devem ser executados dentro dos ambientes oficiais `ai8x-training` e `ai8x-synthesis`.