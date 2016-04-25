CC=g++
MKLROOT = /opt/intel/composer_xe_2013_sp1.2.144/mkl

CFLAGS= -I ../eigen3.2.1 -DNDEBUG -std=c++0x -fopenmp -m64 -I$(MKLROOT)/include

LFLAGS= -Wl,--no-as-needed -L$(MKLROOT)/lib/intel64 -lgomp -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -ldl -lpthread -lm -lstdc++

DEPS=ckmeans.h RedSVD.h
  
OBJS=main.o ckmeans.o

all: ckmeans_train

ckmeans_train: $(OBJS)
	$(CC) $^ $(LFLAGS) -g -o $@

%.o: %.cpp $(DEPS)
	$(CC) $(CFLAGS) -fPIC -O3 -Wall -c $< -o $@ 


clean:
	rm -f *.o ckmeans_train

install:


