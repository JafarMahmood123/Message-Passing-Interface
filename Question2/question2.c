#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ARRAY_SIZE 1000

int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    double start_time = MPI_Wtime();

    // Find out rank, size
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    double *X = NULL;
    double *local_X = NULL;
    int *sendcounts = NULL;
    int *displs = NULL;

    if (world_rank == 0) {
        // Generate random array
        X = (double*)malloc(ARRAY_SIZE * sizeof(double));
        srand(time(NULL));
        for (int i = 0; i < ARRAY_SIZE; i++) {
            X[i] = (double)rand() / RAND_MAX * 100.0; // Random between 0 and 100
        }

        // Calculate sendcounts and displacements for uneven distribution
        sendcounts = (int*)malloc(world_size * sizeof(int));
        displs = (int*)malloc(world_size * sizeof(int));
        int base_count = ARRAY_SIZE / world_size;
        int remainder = ARRAY_SIZE % world_size;
        int offset = 0;
        for (int i = 0; i < world_size; i++) {
            sendcounts[i] = base_count + (i < remainder ? 1 : 0);
            displs[i] = offset;
            offset += sendcounts[i];
        }
    }

    // Broadcast sendcounts to all processes
    int local_count;
    if (world_rank != 0) {
        sendcounts = (int*)malloc(world_size * sizeof(int));
    }
    MPI_Bcast(
      /* data         = */ sendcounts,
      /* count        = */ world_size,
      /* datatype     = */ MPI_INT,
      /* root         = */ 0,
      /* communicator = */ MPI_COMM_WORLD);
    local_count = sendcounts[world_rank];

    // Allocate local array
    local_X = (double*)malloc(local_count * sizeof(double));

    // Scatter the array
    MPI_Scatterv(
      /* sendbuf     = */ X,
      /* sendcounts  = */ sendcounts,
      /* displs      = */ displs,
      /* sendtype    = */ MPI_DOUBLE,
      /* recvbuf     = */ local_X,
      /* recvcount   = */ local_count,
      /* recvtype    = */ MPI_DOUBLE,
      /* root        = */ 0,
      /* comm        = */ MPI_COMM_WORLD);

    // Print received elements
    printf("Process %d received %d elements: ", world_rank, local_count);
    for (int i = 0; i < local_count; i++) {
        printf("%.2f ", local_X[i]);
    }
    printf("\n");

    // Calculate local sum and average
    double local_sum = 0.0;
    for (int i = 0; i < local_count; i++) {
        local_sum += local_X[i];
    }
    double local_avg = local_sum / local_count;
    printf("Process %d: Local sum = %.2f, Local average = %.6f\n", world_rank, local_sum, local_avg);

    // Compute global sum and total count using Allreduce for correct global average
    double global_sum;
    int total_count;
    MPI_Allreduce(
      /* sendbuf    = */ &local_sum,
      /* recvbuf    = */ &global_sum,
      /* count      = */ 1,
      /* datatype   = */ MPI_DOUBLE,
      /* op         = */ MPI_SUM,
      /* comm       = */ MPI_COMM_WORLD);
    MPI_Allreduce(
      /* sendbuf    = */ &local_count,
      /* recvbuf    = */ &total_count,
      /* count      = */ 1,
      /* datatype   = */ MPI_INT,
      /* op         = */ MPI_SUM,
      /* comm       = */ MPI_COMM_WORLD);

    double global_avg = global_sum / total_count;

    // Compute average of local averages (as per requirement)
    double sum_of_local_avgs;
    MPI_Allreduce(
      /* sendbuf    = */ &local_avg,
      /* recvbuf    = */ &sum_of_local_avgs,
      /* count      = */ 1,
      /* datatype   = */ MPI_DOUBLE,
      /* op         = */ MPI_SUM,
      /* comm       = */ MPI_COMM_WORLD);
    double avg_of_avgs = sum_of_local_avgs / world_size;

    printf("Process %d: Correct global average = %.6f, Average of local averages = %.6f\n", world_rank, global_avg, avg_of_avgs);

    double end_time = MPI_Wtime();
    printf("Process %d: Execution time = %.6f seconds\n", world_rank, end_time - start_time);

    // Free memory
    if (world_rank == 0) {
        free(X);
        free(sendcounts);
        free(displs);
    } else {
        free(sendcounts);
    }
    free(local_X);

    MPI_Finalize();
    return 0;
}
