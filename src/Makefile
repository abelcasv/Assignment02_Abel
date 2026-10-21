CC      = gcc
CFLAGS  = -Wall -Wextra -O3 -std=c11
LDFLAGS = -lpsapi

all: dense_benchmark.exe sparse_benchmark.exe

dense_benchmark.exe: dense_benchmark.c
	$(CC) $(CFLAGS) dense_benchmark.c -o dense_benchmark.exe $(LDFLAGS)

sparse_benchmark.exe: sparse_benchmark.c
	$(CC) $(CFLAGS) sparse_benchmark.c -o sparse_benchmark.exe $(LDFLAGS)

clean:
	del /f /q *.exe *.o 2>nul || rm -f *.exe *.o
