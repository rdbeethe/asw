all:
	nvcc -g `pkg-config opencv --cflags` asw.cu `pkg-config opencv --libs`

debug:
	cuda-gdb --args ./a.out l.png r.png 64 5 50

run:
	./a.out l.png r.png 64 5 50 | less
