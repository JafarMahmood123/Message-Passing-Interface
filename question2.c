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
    int process_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
    int num_processes;
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    double *X = NULL;
    double *local_X = NULL;
    int *sendcounts = NULL;
    int *displs = NULL;

    if (process_rank == 0) {
        // Generate random array
        X = (double*)malloc(ARRAY_SIZE * sizeof(double));
        srand(time(NULL));
        for (int i = 0; i < ARRAY_SIZE; i++) {
            X[i] = (double)rand() / RAND_MAX * 100.0; // Random between 0 and 100
        }

        // Calculate sendcounts and displacements for uneven distribution
        sendcounts = (int*)malloc(num_processes * sizeof(int));
        displs = (int*)malloc(num_processes * sizeof(int));
        int base_count = ARRAY_SIZE / num_processes;
        int remainder = ARRAY_SIZE % num_processes;
        int offset = 0;
        for (int i = 0; i < num_processes; i++) {
            sendcounts[i] = base_count + (i < remainder ? 1 : 0);
            displs[i] = offset;
            offset += sendcounts[i];
        }
    }

    // Broadcast sendcounts to all processes
    int local_count;
    if (process_rank != 0) {
        sendcounts = (int*)malloc(num_processes * sizeof(int));
    }
    MPI_Bcast(
      /* data         = */ sendcounts,
      /* count        = */ num_processes,
      /* datatype     = */ MPI_INT,
      /* root         = */ 0,
      /* communicator = */ MPI_COMM_WORLD);
    local_count = sendcounts[process_rank];

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
    printf("Process %d received %d elements: ", process_rank, local_count);
    for (int i = 0; i < local_count; i++) {
        printf("%.15g ", local_X[i]);
    }
    printf("\n");

    // Calculate local sum and average
    double local_sum = 0.0;
    for (int i = 0; i < local_count; i++) {
        local_sum += local_X[i];
    }
    double local_avg = local_sum / local_count;
    printf("Process %d: Local sum = %.15g, Local average = %.15g\n", process_rank, local_sum, local_avg);

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
    double avg_of_avgs = sum_of_local_avgs / num_processes;

    printf("Process %d: Correct global average = %.15g, Average of local averages = %.15g\n", process_rank, global_avg, avg_of_avgs);

    double end_time = MPI_Wtime();
    double execution_time = end_time - start_time;
    printf("Process %d: Execution time = %.15g seconds\n", process_rank, execution_time);

    // Compute total time as sum of all execution times
    double total_time;
    MPI_Allreduce(&execution_time, &total_time, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (process_rank == 0) {
        printf("Total time (sum of all process times) = %.15g seconds\n", total_time);
    }

    // Free memory
    if (process_rank == 0) {
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
