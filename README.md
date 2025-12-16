# MPI Question 2 Implementation

This folder contains the implementation of the second question from the MPI assignment.

## Description

The program generates an array of 1000 random double-precision floating-point numbers in the master process (rank 0). It then distributes these elements among all participating processes using MPI_Scatterv to handle uneven distributions. Each process calculates the average of its received elements locally.

The correct global average is computed by summing all local sums from each process and dividing by the total count of all elements across all processes. This ensures accurate calculation even with uneven data distribution. Additionally, the average of local averages is computed to demonstrate the mathematical note that these two methods produce different results.

## Compilation

To compile the program, use:

```
mpicc question2.c -o question2
```

## Running

To run the program with N processes:

```
mpirun -np N ./question2
```

Replace N with the desired number of processes (e.g., 1, 2, 4, 8).

## Output

Each process will print:
- The elements it received
- Its local sum and average
- Both the correct global average and the average of local averages
- Execution time

## Performance Testing

Run the program with different numbers of processes (e.g., 1, 2, 4, 8) to observe execution times and how performance scales. The program includes timing to measure the total execution time, allowing comparison of performance with varying number of processes. Note the difference between the correct global average and the average of local averages, illustrating the concept from the question's note.
