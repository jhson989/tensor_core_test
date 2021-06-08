CC = /usr/local/cuda/bin/nvcc
PROGS = sgemm
OBJS =
INC =
COMPILE_OPT = -lcublas -O3 -Xcompiler -fopenmp

all: ${PROGS}

%: %.cu ${OBJS} ${INC} Makefile
	${CC} ${COMPILE_OPT} -o $@ ${OBJS} $<


clean :
	rm ${PROGS}


