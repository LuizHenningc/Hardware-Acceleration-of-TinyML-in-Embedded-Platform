import numpy as np

# Substitua pelo caminho de uma imagem real do seu dataset
# Exemplo: imagem = np.load('caminho/para/uma/imagem_sat6.npy')
# Se não tiver o .npy fácil, vamos criar uma imagem sintética colorida para testar:
imagem = np.random.randint(0, 255, (28, 28, 4), dtype=np.uint8)


def salvar_para_c(data, filename="test_image.h"):
    with open(filename, "w") as f:
        f.write("#ifndef TEST_IMAGE_H\n#define TEST_IMAGE_H\n\n")
        f.write(f"const uint8_t test_image_data[] = {{\n")

        # O MAX78000 espera os dados achatados (HWC ou CHW dependendo da config)
        flat_data = data.flatten()
        for i, val in enumerate(flat_data):
            f.write(f"0x{val:02x}, ")
            if (i + 1) % 12 == 0:
                f.write("\n  ")

        f.write("\n};\n\n#endif")


salvar_para_c(imagem)
print("Arquivo test_image.h gerado com sucesso!")