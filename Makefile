CC      = gcc
MPICC   = mpicc
CFLAGS  = -O2 -Wall -Wextra -Wpedantic -std=c11 -fopenmp
SIMDFLAGS = -mavx2
LDFLAGS = -fopenmp -lm
SRCDIR  = src
BINDIR  = bin
TARGETS = $(addprefix $(BINDIR)/,ex1 ex2 ex3)

.PHONY: all clean

all: $(BINDIR) $(TARGETS)

$(BINDIR):
	mkdir -p $(BINDIR)

$(BINDIR)/ex1: $(SRCDIR)/ex1.c
	$(MPICC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BINDIR)/ex2: $(SRCDIR)/ex2.c
	$(MPICC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BINDIR)/ex3: $(SRCDIR)/ex3.c
	$(CC) $(CFLAGS) $(SIMDFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -rf $(BINDIR)
