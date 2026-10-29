CC      = gcc
CFLAGS  = -Wall -Wextra -O3 -std=c11

ifeq ($(OS),Windows_NT)
    LDFLAGS = -lpsapi
    TARGETS = dense_benchmark.exe sparse_benchmark.exe
else
    LDFLAGS = -lm
    TARGETS = dense_benchmark sparse_benchmark
endif

all: $(TARGETS)

dense_benchmark.exe: dense_benchmark.c
	$(CC) $(CFLAGS) dense_benchmark.c -o dense_benchmark.exe $(LDFLAGS)

sparse_benchmark.exe: sparse_benchmark.c
	$(CC) $(CFLAGS) sparse_benchmark.c -o sparse_benchmark.exe $(LDFLAGS)

dense_benchmark: dense_benchmark.c
	$(CC) $(CFLAGS) dense_benchmark.c -o dense_benchmark $(LDFLAGS)

sparse_benchmark: sparse_benchmark.c
	$(CC) $(CFLAGS) sparse_benchmark.c -o sparse_benchmark $(LDFLAGS)

clean:
	rm -f *.exe *.o dense_benchmark sparse_benchmark
