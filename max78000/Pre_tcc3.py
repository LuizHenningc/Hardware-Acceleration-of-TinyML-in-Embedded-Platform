import scipy.io
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SAT6Dataset(Dataset):
    def __init__(self, mat_file_path, is_train=True):
        print(f"Carregando dados de: {mat_file_path}...")

        # Carrega o arquivo .mat
        mat_data = scipy.io.loadmat(mat_file_path)

        # Seleciona treino ou teste
        if is_train:
            images = mat_data['train_x']  # Formato original: (28, 28, 4, N)
            labels = mat_data['train_y']  # Formato original: (6, N) em one-hot
        else:
            images = mat_data['test_x']
            labels = mat_data['test_y']

        # --- O PULO DO GATO PARA A MAX78000 ---
        # 1. Isolar os canais RGB (Pegamos do índice 0 ao 2 e ignoramos o 3 que é o NIR)
        images_rgb = images[:, :, :3, :]  # O formato passa a ser (28, 28, 3, N)

        # 2. Reorganizar as dimensões para o padrão do PyTorch (Batch, Channels, Height, Width)
        # Transpomos de (H, W, C, N) para (N, C, H, W)
        self.images = np.transpose(images_rgb, (3, 2, 0, 1))

        # 3. Converter as labels de one-hot (ex: [0,0,1,0,0,0]) para índice inteiro (ex: 2)
        self.labels = np.argmax(labels, axis=0)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        # Converte para tensor e normaliza os pixels de 0-255 para 0.0-1.0
        image = torch.tensor(self.images[idx], dtype=torch.float32) / 255.0
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label


# --- Testando o nosso DataLoader ---
if __name__ == "__main__":
    # Substitua pelo caminho onde seu arquivo sat-6-full.mat está salvo
    caminho_arquivo = 'SAT-6/sat-6-full.mat'

    # Cria o dataset de treino
    dataset_treino = SAT6Dataset(caminho_arquivo, is_train=True)

    # Cria o DataLoader (que vai alimentar a rede neural em lotes)
    dataloader_treino = DataLoader(dataset_treino, batch_size=64, shuffle=True)

    # Pega um lote para verificar se tudo deu certo
    imagens_lote, labels_lote = next(iter(dataloader_treino))

    print(f"\nSucesso! Formato do lote de imagens: {imagens_lote.shape}")
    print(f"Formato do lote de labels: {labels_lote.shape}")