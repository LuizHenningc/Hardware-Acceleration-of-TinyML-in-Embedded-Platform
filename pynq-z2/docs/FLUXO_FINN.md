# Fluxo utilizado com FINN

A implementação da PYNQ-Z2 foi desenvolvida com base na pasta
`notebooks/sat6_cnn` do fork `ArthurEly/finn`, derivado do projeto oficial
Xilinx/FINN.

O fluxo seguido no trabalho foi:

1. preparação e serialização do dataset SAT-6;
2. definição das topologias quantizadas em PyTorch e Brevitas;
3. treinamento e seleção do melhor modelo;
4. exportação para QONNX;
5. transformação do grafo pelo FINN;
6. definição do folding e da meta de desempenho;
7. simulação RTL, síntese OOC e análise de recursos;
8. geração do bitstream, arquivo HWH e driver PYNQ;
9. implantação física da configuração `t2w8_500fps` na PYNQ-Z2;
10. validação com 1.000 imagens do SAT-6.

## Projeto-base

- Repositório: `ArthurEly/finn`
- Pasta de referência: `notebooks/sat6_cnn`
- Projeto original do compilador: `Xilinx/FINN`

Os notebooks do projeto-base não foram copiados para esta pasta porque as
células e parâmetros públicos não representam exatamente a execução final do
TCC. O repositório preserva os artefatos finais da implantação t2w8 e os
resultados disponíveis do fluxo de 50 épocas.
