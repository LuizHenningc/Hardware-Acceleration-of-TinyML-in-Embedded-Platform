// Compilação sugerida:
// gcc sat6_jetson_tf_c.c -ltensorflow -o model_complete_sat6
//
// Exemplo simples:
// ./model_complete_sat6 256 1000
//
// Exemplo completo:
// ./model_complete_sat6 256 1000 sat6_saved/ X_test_sat6_tf_hwc.csv y_test_sat6_tf_labels.csv serving_default_input_sat6 StatefulPartitionedCall none 0
//
// Argumentos:
// 1) batch_size
// 2) num_samples
// 3) model_path            opcional, padrão: sat6_saved/
// 4) x_csv                 opcional, padrão: X_test_sat6_tf_hwc.csv
// 5) y_csv                 opcional, padrão: y_test_sat6_tf_labels.csv
// 6) input_op              opcional, padrão: serving_default_input_sat6
// 7) output_op             opcional, padrão: StatefulPartitionedCall
// 8) config_pb             opcional, use "none" para ignorar
// 9) normalize_255         opcional, 0 ou 1. Use 1 se o CSV estiver em 0..255 e o modelo esperar 0..1.

#include <tensorflow/c/c_api.h>

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define SAT6_HEIGHT 28
#define SAT6_WIDTH 28
#define SAT6_CHANNELS 4
#define FEATURE_COUNT (SAT6_HEIGHT * SAT6_WIDTH * SAT6_CHANNELS)
#define NUM_CLASSES 6
#define MAX_LINE_LENGTH 100000

static long long get_elapsed_ns(struct timespec start, struct timespec end) {
    long long start_ns = ((long long)start.tv_sec * 1000000000LL) + start.tv_nsec;
    long long end_ns = ((long long)end.tv_sec * 1000000000LL) + end.tv_nsec;

    return end_ns - start_ns;
}

static void tensor_deallocator(void* data, size_t len, void* arg) {
    (void)len;
    (void)arg;
    free(data);
}

static void check_status(TF_Status* status) {
    if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "Erro TensorFlow: %s\n", TF_Message(status));
        exit(EXIT_FAILURE);
    }
}

static int file_exists(const char* filename) {
    FILE* file = fopen(filename, "rb");

    if (file == NULL) {
        return 0;
    }

    fclose(file);
    return 1;
}

static TF_SessionOptions* create_session_options(const char* config_file_path) {
    TF_SessionOptions* options = TF_NewSessionOptions();

    if (config_file_path == NULL) {
        return options;
    }

    if (strcmp(config_file_path, "none") == 0) {
        return options;
    }

    if (!file_exists(config_file_path)) {
        fprintf(stderr, "Aviso: config_pb não encontrado (%s). Continuando sem config customizada.\n", config_file_path);
        return options;
    }

    FILE* file = fopen(config_file_path, "rb");

    if (file == NULL) {
        fprintf(stderr, "Aviso: não foi possível abrir config_pb (%s). Continuando sem config customizada.\n", config_file_path);
        return options;
    }

    fseek(file, 0, SEEK_END);
    long config_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    void* config_data = malloc((size_t)config_size);

    if (config_data == NULL) {
        fclose(file);
        fprintf(stderr, "Erro: falha ao alocar memória para config_pb.\n");
        exit(EXIT_FAILURE);
    }

    size_t read_count = fread(config_data, 1, (size_t)config_size, file);
    fclose(file);

    if (read_count != (size_t)config_size) {
        free(config_data);
        fprintf(stderr, "Erro: falha ao ler config_pb completo.\n");
        exit(EXIT_FAILURE);
    }

    TF_Status* status = TF_NewStatus();
    TF_SetConfig(options, config_data, (size_t)config_size, status);
    free(config_data);
    check_status(status);
    TF_DeleteStatus(status);

    return options;
}

static TF_Tensor* create_input_tensor(float* input_buffer, int batch_size) {
    int64_t dims[4] = {batch_size, SAT6_HEIGHT, SAT6_WIDTH, SAT6_CHANNELS};
    size_t total_bytes = (size_t)batch_size * FEATURE_COUNT * sizeof(float);

    return TF_NewTensor(TF_FLOAT, dims, 4, input_buffer, total_bytes, tensor_deallocator, NULL);
}

static float** read_x_test(const char* filename, int64_t num_samples, int normalize_255) {
    FILE* file = fopen(filename, "r");

    if (file == NULL) {
        fprintf(stderr, "Erro ao abrir X CSV (%s): %s\n", filename, strerror(errno));
        return NULL;
    }

    float** data = malloc((size_t)num_samples * sizeof(float*));

    if (data == NULL) {
        fclose(file);
        fprintf(stderr, "Erro: falha ao alocar matriz X.\n");
        exit(EXIT_FAILURE);
    }

    char line[MAX_LINE_LENGTH];
    int64_t sample_idx = 0;

    while (fgets(line, MAX_LINE_LENGTH, file) != NULL && sample_idx < num_samples) {
        data[sample_idx] = malloc(FEATURE_COUNT * sizeof(float));

        if (data[sample_idx] == NULL) {
            fclose(file);
            fprintf(stderr, "Erro: falha ao alocar amostra %ld.\n", sample_idx);
            exit(EXIT_FAILURE);
        }

        char* token = strtok(line, ",");
        int feature_idx = 0;

        while (token != NULL && feature_idx < FEATURE_COUNT) {
            float value = strtof(token, NULL);

            if (normalize_255) {
                value = value / 255.0f;
            }

            data[sample_idx][feature_idx] = value;
            feature_idx++;
            token = strtok(NULL, ",");
        }

        if (feature_idx != FEATURE_COUNT) {
            fprintf(stderr, "Erro: amostra %ld tem %d features. Esperado: %d.\n", sample_idx, feature_idx, FEATURE_COUNT);
            fclose(file);
            exit(EXIT_FAILURE);
        }

        sample_idx++;
    }

    fclose(file);

    if (sample_idx != num_samples) {
        fprintf(stderr, "Erro: esperado %ld amostras no X CSV, mas foram lidas %ld.\n", num_samples, sample_idx);
        exit(EXIT_FAILURE);
    }

    return data;
}

static int* read_y_test(const char* filename, int64_t num_samples) {
    FILE* file = fopen(filename, "r");

    if (file == NULL) {
        fprintf(stderr, "Erro ao abrir y CSV (%s): %s\n", filename, strerror(errno));
        return NULL;
    }

    int* labels = malloc((size_t)num_samples * sizeof(int));

    if (labels == NULL) {
        fclose(file);
        fprintf(stderr, "Erro: falha ao alocar labels.\n");
        exit(EXIT_FAILURE);
    }

    char line[128];
    int64_t idx = 0;

    while (fgets(line, sizeof(line), file) != NULL && idx < num_samples) {
        labels[idx] = atoi(line);
        idx++;
    }

    fclose(file);

    if (idx != num_samples) {
        fprintf(stderr, "Erro: esperado %ld labels, mas foram lidos %ld.\n", num_samples, idx);
        exit(EXIT_FAILURE);
    }

    return labels;
}

static TF_Output get_graph_output(TF_Graph* graph, const char* op_name, const char* fallback_names[], int fallback_count) {
    TF_Output output = {TF_GraphOperationByName(graph, op_name), 0};

    if (output.oper != NULL) {
        return output;
    }

    for (int i = 0; i < fallback_count; i++) {
        output.oper = TF_GraphOperationByName(graph, fallback_names[i]);

        if (output.oper != NULL) {
            fprintf(stderr, "Aviso: operação '%s' não encontrada. Usando fallback '%s'.\n", op_name, fallback_names[i]);
            return output;
        }
    }

    return output;
}

static void print_usage(const char* executable_name) {
    printf("Uso: %s <batch_size> <num_samples> [model_path] [x_csv] [y_csv] [input_op] [output_op] [config_pb|none] [normalize_255]\n", executable_name);
    printf("Exemplo: %s 256 1000 sat6_saved/ X_test_sat6_tf_hwc.csv y_test_sat6_tf_labels.csv serving_default_input_sat6 StatefulPartitionedCall none 0\n", executable_name);
}

static void free_x_data(float** x_data, int64_t num_samples) {
    if (x_data == NULL) {
        return;
    }

    for (int64_t i = 0; i < num_samples; i++) {
        free(x_data[i]);
    }

    free(x_data);
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    int batch_size = atoi(argv[1]);
    int64_t total_samples = atoll(argv[2]);

    if (batch_size <= 0) {
        fprintf(stderr, "Erro: batch_size precisa ser maior que zero.\n");
        return EXIT_FAILURE;
    }

    if (total_samples <= 0) {
        fprintf(stderr, "Erro: num_samples precisa ser maior que zero.\n");
        return EXIT_FAILURE;
    }

    const char* model_path = argc > 3 ? argv[3] : "sat6_saved/";
    const char* x_csv = argc > 4 ? argv[4] : "X_test_sat6_tf_hwc.csv";
    const char* y_csv = argc > 5 ? argv[5] : "y_test_sat6_tf_labels.csv";
    const char* input_op_name = argc > 6 ? argv[6] : "serving_default_input_sat6";
    const char* output_op_name = argc > 7 ? argv[7] : "StatefulPartitionedCall";
    const char* config_file_path = argc > 8 ? argv[8] : "none";
    int normalize_255 = argc > 9 ? atoi(argv[9]) : 0;

    setenv("TF_CPP_MIN_LOG_LEVEL", "0", 1);

    TF_Status* status = TF_NewStatus();
    TF_Graph* graph = TF_NewGraph();
    TF_SessionOptions* session_opts = create_session_options(config_file_path);
    const char* tags = "serve";

    printf("\nCarregando modelo: %s\n", model_path);
    TF_Session* session = TF_LoadSessionFromSavedModel(session_opts, NULL, model_path, &tags, 1, graph, NULL, status);
    check_status(status);

    const char* input_fallbacks[] = {
        "serving_default_input_sat6",
        "input_sat6",
        "serving_default_input_1",
        "input_1"
    };

    const char* output_fallbacks[] = {
        "StatefulPartitionedCall",
        "StatefulPartitionedCall_1",
        "Identity",
        "output_0"
    };

    TF_Output input_op = get_graph_output(graph, input_op_name, input_fallbacks, 4);
    TF_Output output_op = get_graph_output(graph, output_op_name, output_fallbacks, 4);

    if (input_op.oper == NULL) {
        fprintf(stderr, "Erro: operação de entrada não encontrada: %s\n", input_op_name);
        fprintf(stderr, "Confira com: saved_model_cli show --dir %s --tag_set serve --signature_def serving_default\n", model_path);
        exit(EXIT_FAILURE);
    }

    if (output_op.oper == NULL) {
        fprintf(stderr, "Erro: operação de saída não encontrada: %s\n", output_op_name);
        fprintf(stderr, "Confira com: saved_model_cli show --dir %s --tag_set serve --signature_def serving_default\n", model_path);
        exit(EXIT_FAILURE);
    }

    printf("Entrada: %s\n", TF_OperationName(input_op.oper));
    printf("Saída: %s\n", TF_OperationName(output_op.oper));
    printf("Formato SAT-6 esperado: [%d, %d, %d, %d]\n", batch_size, SAT6_HEIGHT, SAT6_WIDTH, SAT6_CHANNELS);
    printf("Features por amostra: %d\n", FEATURE_COUNT);
    printf("Classes: %d\n", NUM_CLASSES);
    printf("Normalização 0..255 para 0..1: %s\n", normalize_255 ? "sim" : "não");

    float** x_data = read_x_test(x_csv, total_samples, normalize_255);
    int* y_labels = read_y_test(y_csv, total_samples);

    if (x_data == NULL || y_labels == NULL) {
        fprintf(stderr, "Erro: falha ao carregar CSVs.\n");
        exit(EXIT_FAILURE);
    }

    int64_t correct = 0;
    struct timespec t_start;
    struct timespec t_end;

    printf("\nIniciando inferência\n");
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    for (int64_t i = 0; i < total_samples; i += batch_size) {
        int current_batch_size = batch_size;

        if (i + batch_size > total_samples) {
            current_batch_size = (int)(total_samples - i);
        }

        float* input_data = malloc((size_t)current_batch_size * FEATURE_COUNT * sizeof(float));

        if (input_data == NULL) {
            fprintf(stderr, "Erro: falha ao alocar input batch.\n");
            exit(EXIT_FAILURE);
        }

        for (int b = 0; b < current_batch_size; b++) {
            memcpy(&input_data[(size_t)b * FEATURE_COUNT], x_data[i + b], FEATURE_COUNT * sizeof(float));
        }

        TF_Tensor* input_tensor = create_input_tensor(input_data, current_batch_size);

        if (input_tensor == NULL) {
            fprintf(stderr, "Erro: input_tensor está NULL.\n");
            free(input_data);
            exit(EXIT_FAILURE);
        }

        TF_Tensor* output_tensor = NULL;

        TF_SessionRun(
            session,
            NULL,
            &input_op,
            &input_tensor,
            1,
            &output_op,
            &output_tensor,
            1,
            NULL,
            0,
            NULL,
            status
        );

        check_status(status);

        if (output_tensor == NULL) {
            fprintf(stderr, "Erro: output_tensor está NULL.\n");
            TF_DeleteTensor(input_tensor);
            exit(EXIT_FAILURE);
        }

        size_t expected_output_bytes = (size_t)current_batch_size * NUM_CLASSES * sizeof(float);
        size_t output_bytes = TF_TensorByteSize(output_tensor);

        if (output_bytes < expected_output_bytes) {
            fprintf(
                stderr,
                "Erro: saída do modelo menor que o esperado. Recebido: %zu bytes. Esperado pelo menos: %zu bytes.\n",
                output_bytes,
                expected_output_bytes
            );
            TF_DeleteTensor(output_tensor);
            TF_DeleteTensor(input_tensor);
            exit(EXIT_FAILURE);
        }

        float* output_data = (float*)TF_TensorData(output_tensor);

        for (int b = 0; b < current_batch_size; b++) {
            float max_value = output_data[(size_t)b * NUM_CLASSES];
            int predicted_class = 0;

            for (int class_idx = 1; class_idx < NUM_CLASSES; class_idx++) {
                float value = output_data[((size_t)b * NUM_CLASSES) + class_idx];

                if (value > max_value) {
                    max_value = value;
                    predicted_class = class_idx;
                }
            }

            if (predicted_class == y_labels[i + b]) {
                correct++;
            }
        }

        TF_DeleteTensor(output_tensor);
        TF_DeleteTensor(input_tensor);
    }

    clock_gettime(CLOCK_MONOTONIC, &t_end);

    long long elapsed_ns = get_elapsed_ns(t_start, t_end);
    double elapsed_seconds = (double)elapsed_ns / 1000000000.0;
    double latency_ms_per_image = (elapsed_seconds * 1000.0) / (double)total_samples;
    double accuracy = (100.0 * (double)correct) / (double)total_samples;

    printf("\nFinalizado\n");
    printf("Tempo total: %.6f segundos\n", elapsed_seconds);
    printf("Latência média: %.6f ms/imagem\n", latency_ms_per_image);
    printf("Acurácia: %.2f%% (%ld/%ld corretas)\n", accuracy, correct, total_samples);

    FILE* logfile = fopen("execution_log_sat6_jetson.csv", "a");

    if (logfile != NULL) {
        fprintf(logfile, "%d,%ld,%.6f,%.6f,%.2f,%ld\n", batch_size, total_samples, elapsed_seconds, latency_ms_per_image, accuracy, correct);
        fclose(logfile);
    }

    free_x_data(x_data, total_samples);
    free(y_labels);

    TF_DeleteSession(session, status);
    TF_DeleteSessionOptions(session_opts);
    TF_DeleteGraph(graph);
    TF_DeleteStatus(status);

    return EXIT_SUCCESS;
}
