#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 7     // Размер матриц (N x N)
#define ROOT 0  // Номер главного процесса

// Функция расчёта одной или нескольких строк произведения матриц
void multiply_matrices(int *A, int *B, int *C, int rows) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = 0;
            for (int k = 0; k < N; k++) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}

// Функция вывода матрицы на экран
void print_matrix(int *matrix) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d\t", matrix[i * N + j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]) {
    int rank, numprocs;
    long double starttime, endtime;
    int *sendcounts = NULL;
    int *displs = NULL;

    // Инициализация библиотеки MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    starttime = MPI_Wtime();

    // Инициализация матриц A и B
    int A[N * N], B[N * N], C[N * N];

    if (rank == ROOT) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i * N + j] = (4 + i * j) + j;
            }
        }

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                B[i * N + j] = (12 - i * j) + i;
            }
        }
    }

    // Рассылка матрицы B всем процессам
    MPI_Bcast(B, N * N, MPI_INT, ROOT, MPI_COMM_WORLD);

    // Нахождение размера матриц для каждого процесса
    int rows_per_process = N / numprocs;  // Базовое количество строк, которые получит каждый процесс
    int remainder = N % numprocs;         // Остаток строк

    // Определение количества строк для каждого процесса
    sendcounts = (int *)malloc(numprocs * sizeof(int));  // Количество чисел для каждого процесса
    displs = (int *)malloc(numprocs * sizeof(int));      // Смещение от начала матрицы А
    for (int i = 0; i < numprocs; i++) {
        sendcounts[i] = N * (rows_per_process + (i < remainder ? 1 : 0));
        displs[i] = ((i == 0) ? 0 : displs[i - 1] + sendcounts[i - 1]);
    }

    int local_A[sendcounts[rank]];  // Локальная матрица A для каждого процесса
    int local_C[sendcounts[rank]];  // Локальная матрица C для каждого процесса

    // Распределение строк матрицы A между процессами
    MPI_Scatterv(A, sendcounts, displs, MPI_INT, local_A, sendcounts[rank], MPI_INT, ROOT, MPI_COMM_WORLD);

    // Умножение матриц
    multiply_matrices(local_A, B, local_C, (sendcounts[rank] / N));

    // Сбор результатов в матрицу C
    MPI_Gatherv(local_C, sendcounts[rank], MPI_INT, C, sendcounts, displs, MPI_INT, ROOT, MPI_COMM_WORLD);

    endtime = MPI_Wtime();

    // Вывод результата только на процессе 0
    if (rank == ROOT) {
        printf("Матрица A:\n");
        print_matrix(A);
        printf("\nМатрица B:\n");
        print_matrix(B);
        printf("\nРезультат умножения матриц:\n");
        print_matrix(C);
        printf("\nВыполнение заняло %Lf секунд\n", endtime - starttime);
    }

    // Очищение выделенной памяти
    free(sendcounts);
    free(displs);

    MPI_Finalize();
    return 0;
}